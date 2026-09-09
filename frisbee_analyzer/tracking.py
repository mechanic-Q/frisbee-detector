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


def iter_player_tracks(
    video_path: str | Path,
    weights: str = "yolo26x.pt",
    conf: float = 0.25,
    imgsz: int = 1280,
    tracker: str | None = None,
    cancel_check=None,
    max_frames: int | None = None,
    classes: tuple[int, ...] = (0,),
):
    """逐帧产出 (frame_idx, detections)。detection = {track_id, bbox, conf, cls}。

    零样本用 COCO person 预训练权重（classes=(0,)）；players_e4 微调权重就绪后传
    classes=(0,1,2)（player-red/player-blue/referee），类别即队伍（见 pipeline
    --team-from-cls），观众从检测端被类别排除。
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
        yield frame_idx, dets
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break
