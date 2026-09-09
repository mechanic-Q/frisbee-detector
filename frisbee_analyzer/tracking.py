"""球员检测 + BoT-SORT 跟踪（worker 侧，需 GPU 环境）。

移植自 E2 实测原型 data/bili_final_test/track_players.py。注意：方案文档 §2.1
曾写"固定机位 → gmc_method=none"，但 E2 实测机位为边线摇镜，默认配置的
sparseOptFlow 全局运动补偿对摇镜有效，因此保留 ultralytics 默认 botsort.yaml。
"""

from __future__ import annotations

from pathlib import Path

import cv2


class WorkerCancelled(Exception):
    """GUI 侧请求取消时由 cancel_check 触发。"""


def probe_video(video_path: str | Path) -> dict:
    """读视频元信息（不解码全片）。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def make_tracker_config(conf: float, base: str = "botsort_players.yaml") -> str:
    """生成临时 tracker YAML，把检出阈值与 --conf 对齐。

    根因：BoT-SORT 的 track_high_thresh/new_track_thresh 默认 0.25，会先于推理的 conf
    丢弃低分检测——只传 --conf 无效（2026-09-09 实测确认）。这里按 conf 覆写后返回路径。
    """
    import tempfile
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "configs" / "trackers" / base
    text = src.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        key = line.split(":")[0].strip()
        if key in ("track_high_thresh", "new_track_thresh"):
            lines.append(f"{key}: {conf}")
        elif key == "track_low_thresh":
            lines.append(f"{key}: {min(0.1, conf)}")
        else:
            lines.append(line)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="botsort_conf_")
    with open(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def tiled_detect(model, frame, conf: float, imgsz: int, classes, grid: int = 2,
                 overlap: float = 0.2, nms_iou: float = 0.5):
    """切片检测：把画面切成 grid×grid（带 overlap）分别推理，合并去重。

    动机（2026-09-09 实测）：1080P 远场球员仅几十像素，整帧推理漏检严重；
    2×2 切片把远场目标放大 ~2 倍后召回 +29~33%（整帧 14-30 → 切片 18-40 框/帧）。
    返回 [(bbox, conf, cls), ...]（像素坐标，已跨片 NMS）。
    """
    import numpy as np

    h, w = frame.shape[:2]
    step_y, step_x = int(h / grid), int(w / grid)
    pad_y, pad_x = int(step_y * overlap), int(step_x * overlap)
    boxes, scores, clss = [], [], []
    for i in range(grid):
        for j in range(grid):
            y1 = max(0, i * step_y - pad_y)
            y2 = min(h, (i + 1) * step_y + pad_y)
            x1 = max(0, j * step_x - pad_x)
            x2 = min(w, (j + 1) * step_x + pad_x)
            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            res = model.predict(tile, conf=conf, imgsz=imgsz, classes=list(classes),
                                verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for box, cf, cl in zip(res.boxes.xyxy.cpu().numpy(),
                                   res.boxes.conf.cpu().tolist(),
                                   res.boxes.cls.int().cpu().tolist()):
                boxes.append([box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1])
                scores.append(float(cf))
                clss.append(int(cl))
    if not boxes:
        return []
    keep = _nms(np.asarray(boxes), np.asarray(scores), nms_iou)
    return [(boxes[i], scores[i], clss[i]) for i in keep]


def _nms(boxes, scores, iou_thr: float):
    import numpy as np

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def iter_player_tracks(
    video_path: str | Path,
    weights: str = "yolo26x.pt",
    conf: float = 0.25,
    imgsz: int = 1280,
    tracker: str | None = None,
    cancel_check=None,
    max_frames: int | None = None,
    classes: tuple[int, ...] = (0,),
    tile_grid: int = 0,
    tile_conf: float = 0.3,
):
    """逐帧产出 (frame_idx, detections)。detection = {track_id, bbox, conf, cls}。

    零样本用 COCO person 预训练权重（classes=(0,)）；players_e4 微调权重就绪后传
    classes=(0,1,2)（player-red/player-blue/referee），类别即队伍（见 pipeline
    --team-from-cls），观众从检测端被类别排除。

    tile_grid>0 时启用切片检测补漏（远场小目标召回 +29~33%，见 tiled_detect）：
    整帧推理驱动 BoT-SORT 保持轨迹连续，切片结果中与已有框 IoU<0.5 的作为补框
    （track_id 取 -1，由上层/统计侧按需处理，不参与跟踪）。
    """
    from ultralytics import YOLO  # 惰性导入：模块本身可在无 torch 环境做静态检查

    if tracker is None:
        tracker = make_tracker_config(conf)
    model = YOLO(str(weights))
    frame_idx = 0
    for res in model.track(
        source=str(video_path),
        conf=conf,
        imgsz=imgsz,
        classes=list(classes),
        tracker=tracker,
        persist=True,
        stream=True,
        verbose=False,
    ):
        if cancel_check is not None and cancel_check():
            raise WorkerCancelled(f"cancelled at frame {frame_idx}")
        dets = []
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().cpu().tolist()
            confs = res.boxes.conf.cpu().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.int().cpu().tolist()
            for tid, c, box, cbin in zip(ids, confs, boxes, clss):
                dets.append({
                    "track_id": int(tid),
                    "bbox": [round(float(v), 1) for v in box],
                    "conf": round(float(c), 3),
                    "cls": int(cbin),
                })

        if tile_grid > 0 and res.orig_img is not None:
            # 切片补漏：与整帧/已跟踪框 IoU<0.5 的切片结果作为补框（track_id=-1）
            for box, cf, cl in tiled_detect(model, res.orig_img, tile_conf, imgsz,
                                            classes, grid=tile_grid):
                if cf < tile_conf:
                    continue
                if any(_iou_xyxy(box, d["bbox"]) > 0.5 for d in dets):
                    continue
                dets.append({
                    "track_id": -1,
                    "bbox": [round(float(v), 1) for v in box],
                    "conf": round(float(cf), 3),
                    "cls": int(cl),
                })

        yield frame_idx, dets
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break


def _iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
