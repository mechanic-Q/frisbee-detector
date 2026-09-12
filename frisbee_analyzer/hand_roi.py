"""手部 ROI 高分辨率复检（F1 第二检测通道，§13.12）。

根因（G4 终判遗留，§13.11）：持盘中的盘贴身遮挡，整帧检测器看不见 →
融合轨迹降解 predicting/lost，持盘状态机 0 产出，得分只能靠端区慢速走段候选规则。

思路：融合状态里非 tracking 的帧（盘"消失"时段），取离最后观测锚点最近的
K 个球员，把手部区（躯干横带向两侧扩展）crop 出来放大复检；检出映射回原帧
坐标（source="hand_roi"）并入逐帧检测序列，二次过 fuse_disc_detections。
健康 tracking 帧不复检——主通道已看见的盘不需要第二通道，也杜绝 ROI 误检
干扰好轨迹。

纯几何函数（ROI 推导/持盘人选/检出合并）只依赖标准库，可单测；
视频复检入口 recheck_video 需要 ultralytics+cv2（GPU/CPU 均可）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── 可调常量（默认即生产配置；pipeline 以 CLI 参数覆盖）──
HAND_ROI_PAD_W = 0.60     # ROI 左右各扩 0.6×框宽（手臂伸展区）
HAND_ROI_TOP = 0.15       # ROI 顶部下移 0.15×框高（躲开头部——白帽是域外头号误检形态）
HAND_ROI_BOTTOM = 0.25    # ROI 底部下探 0.25×框高（腰间/低持盘）
HAND_ROI_NEAR_FRAC = 1.2  # 锚点-球员框中心距离 < 1.2×框高 才算"可能持盘人"
HAND_ROI_K = 2            # 每帧最多复检的球员数（按距离取最近）
HAND_ROI_CONF = 0.30      # ROI 复检测出阈值（crop 放大后检出应更自信，卡紧防误检）
HAND_ROI_IMGSZ = 640      # crop 送检分辨率（480p 素材球员框 ~40-80px，640 足够）
HAND_ROI_MAX_GAP = 120    # 距最后观测 ≤ 此帧数才复检（4s@30fps；持盘很少更久）
HAND_ROI_MIN_CROP = 24    # crop 边长下限（px），更小没有检测意义
ROI_DEDUP_IOU = 0.5       # 与整帧检出重叠超过此 IoU 的 ROI 检出丢弃（同一物理盘）


@dataclass
class HandRoiStats:
    frames_with_anchor: int = 0   # 有锚点的候选帧数（可能触发复检）
    frames_rechecked: int = 0     # 实际跑了推理的帧数
    rois_checked: int = 0         # 实际送检的 crop 数
    dets_added: int = 0           # 复检产出检出数
    dets_deduped: int = 0         # 与整帧检出重叠被丢弃数
    details: dict = field(default_factory=dict)


def hand_roi_box(bbox, pad_w: float = HAND_ROI_PAD_W,
                 top: float = HAND_ROI_TOP, bottom: float = HAND_ROI_BOTTOM,
                 frame_w: int | None = None, frame_h: int | None = None) -> tuple:
    """球员框 → 手部区 ROI（躯干横带 + 双侧手臂伸展区），钳到画面内。"""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    rx1 = x1 - pad_w * bw
    rx2 = x2 + pad_w * bw
    ry1 = y1 + top * bh
    ry2 = y2 + bottom * bh
    if frame_w is not None:
        rx1, rx2 = max(0.0, rx1), min(float(frame_w), rx2)
    if frame_h is not None:
        ry1, ry2 = max(0.0, ry1), min(float(frame_h), ry2)
    return rx1, ry1, rx2, ry2


def select_holders(players: list[dict], anchor: tuple[float, float],
                   k: int = HAND_ROI_K, near_frac: float = HAND_ROI_NEAR_FRAC,
                   frame_w: int | None = None, frame_h: int | None = None) -> list[dict]:
    """选可能持盘的球员：手部 ROI 含锚点，或框中心距锚点 < near_frac×框高。

    返回按距离升序的前 k 个（元素为原 det dict，附 "hr_dist"）。
    """
    out = []
    ax, ay = anchor
    for det in players:
        x1, y1, x2, y2 = (float(v) for v in det["bbox"])
        bh = max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.hypot(cx - ax, cy - ay)
        roi = hand_roi_box(det["bbox"], frame_w=frame_w, frame_h=frame_h)
        contains = roi[0] <= ax <= roi[2] and roi[1] <= ay <= roi[3]
        if contains or dist < near_frac * bh:
            out.append({**det, "hr_dist": dist})
    out.sort(key=lambda d: d["hr_dist"])
    return out[:k]


def merge_hand_roi_detections(seq: list[list[dict]],
                              extra: dict[int, list[dict]]) -> tuple[list[list[dict]], HandRoiStats]:
    """把 ROI 复检出（原帧坐标）并入逐帧检测序列，返回 (seq2, stats)。

    与该帧已有整帧检出 IoU > ROI_DEDUP_IOU 的 ROI 检出丢弃——同一物理盘
    不重复喂给融合层。
    """
    stats = HandRoiStats()
    seq2 = [list(dets) for dets in seq]

    def _iou(a, b) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / ua if ua > 0 else 0.0

    for fi, dets in extra.items():
        if fi < 0 or fi >= len(seq2):
            continue
        for d in dets:
            stats.dets_added += 1
            if any(_iou(d["bbox"], e["bbox"]) > ROI_DEDUP_IOU for e in seq2[fi]):
                stats.dets_deduped += 1
                continue
            seq2[fi].append(d)
    return seq2, stats


def anchor_candidates(fused, max_gap: int = HAND_ROI_MAX_GAP) -> dict[int, tuple[float, float]]:
    """融合结果 → {候选帧号: 最后观测锚点}。

    候选帧 = 非 tracking 且距最近一次 tracking ≤ max_gap。
    锚点固定在最后观测位置（不随 Kalman 外推漂移——持盘人站着不动，
    外推点会漂走）。
    """
    candidates: dict[int, tuple[float, float]] = {}
    last_obs: tuple[float, float] | None = None
    last_trk = -10**9
    for i, f in enumerate(fused):
        if f.status == "tracking":
            last_obs = (f.cx, f.cy)
            last_trk = i
        elif last_obs is not None and i - last_trk <= max_gap:
            candidates[i] = last_obs
    return candidates


def recheck_video(video_path, fused, players_by_frame: dict[int, list[dict]],
                  disc_weights: str, k: int = HAND_ROI_K,
                  conf: float = HAND_ROI_CONF, imgsz: float = HAND_ROI_IMGSZ,
                  max_gap: int = HAND_ROI_MAX_GAP,
                  width: int | None = None, height: int | None = None,
                  cancel_check=None, log=None) -> tuple[dict[int, list[dict]], HandRoiStats]:
    """逐帧读视频，对候选帧的持盘人手部 ROI 做放大复检。

    fused: fuse_disc_detections 输出的 list[DiscFrame]（与帧号 0..N-1 对齐）。
    players_by_frame: 帧号 → 球员 det 列表（未过滤的原始检出即可）。
    返回 ({帧号: [{bbox, conf, source:"hand_roi"}, ...]}, stats)。
    """
    import cv2

    candidates = anchor_candidates(fused, max_gap=max_gap)
    stats = HandRoiStats(frames_with_anchor=len(candidates))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    if width is None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if height is None:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    from ultralytics import YOLO
    model = YOLO(str(disc_weights))

    extra: dict[int, list[dict]] = {}
    idx = 0
    while True:
        if cancel_check is not None and cancel_check():
            break
        ok, frame = cap.read()
        if not ok:
            break
        anchor = candidates.get(idx)
        if anchor is not None:
            holders = select_holders(players_by_frame.get(idx, []), anchor, k=k,
                                     frame_w=width, frame_h=height)
            crops, offsets = [], []
            for h in holders:
                rx1, ry1, rx2, ry2 = hand_roi_box(h["bbox"], frame_w=width, frame_h=height)
                cx1, cy1 = int(round(rx1)), int(round(ry1))
                cx2, cy2 = int(round(rx2)), int(round(ry2))
                if cx2 - cx1 < HAND_ROI_MIN_CROP or cy2 - cy1 < HAND_ROI_MIN_CROP:
                    continue
                crops.append(frame[cy1:cy2, cx1:cx2])
                offsets.append((cx1, cy1))
            if crops:
                stats.frames_rechecked += 1
                stats.rois_checked += len(crops)
                results = model.predict(crops, conf=conf, imgsz=imgsz, verbose=False)
                frame_dets = []
                for res, (ox, oy) in zip(results, offsets):
                    if res.boxes is None or len(res.boxes) == 0:
                        continue
                    for box, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                       res.boxes.conf.cpu().tolist()):
                        frame_dets.append({
                            "bbox": [round(float(box[0]) + ox, 1), round(float(box[1]) + oy, 1),
                                     round(float(box[2]) + ox, 1), round(float(box[3]) + oy, 1)],
                            "conf": round(float(cf), 4),
                            "source": "hand_roi",
                        })
                if frame_dets:
                    extra[idx] = frame_dets
            if log is not None and stats.frames_rechecked % 300 == 0:
                log(f"hand-roi: {stats.frames_rechecked} frames rechecked, "
                    f"{sum(len(v) for v in extra.values())} dets so far")
        idx += 1
    cap.release()
    return extra, stats
