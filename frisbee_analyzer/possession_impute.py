"""持盘补全（possession imputation，F1 第三通道，§13.14）。

根因（§13.11/13.12）：持盘中的盘贴身遮挡，融合轨迹在持盘段降解/断轨，
事件引擎拿到 disc=None → 持盘状态机在持盘段"失明"（G4 f836 前后 possession
_start 后 8 帧即 possession_end lost）。

算法优先方案（Stage 2 后自纠正，替代盲区 crop 分类器——世界坐标近距判别
噪声 ~7m 见 §13.13，外观分类器信号弱）：融合 degraded 段若**像素锚点稳定
落在唯一球员框内**（同 track_id 连续、锚点不动），则该球员在持盘——把
"手部点"（脚点世界坐标 + 手高）反投影回像素，合成盘观测（source=
"possession_imputed"）并入序列二次融合，状态机因此能持续看见盘。

与 hand_roi 的区别：hand_roi 是"再测一次"（有真实图像证据），本模块是
"有据推断"（几何+身份连续性证据），合成观测永远带 source 标记可审计，
streak 上限防长程伪造。

纯几何，只依赖标准库；merge 复用 hand_roi.merge_hand_roi_detections。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── 可调常量（pipeline CLI 可覆盖）──
IMPUTE_BOX_PAD = 0.10      # 球员框外扩比例（锚点容差）
IMPUTE_MAX_GAP = 150       # 距最后观测 ≤ 此帧数才补全（5s@30fps；持盘很少更久）
IMPUTE_STREAK = 90         # 单段最多连续补全帧数（3s，防长程伪造）
IMPUTE_MIN_STREAK = 3      # 至少连续 N 帧同球员含锚点才补全（防闪帧）
IMPUTE_HAND_HEIGHT = 1.3   # 手部点离地高度 m（= events.py 1.8m×0.72）
IMPUTE_BBOX_SIZE = 16.0    # 合成盘 bbox 边长 px
IMPUTE_CONF = 0.50         # 合成观测置信度（搜索态 MIN_SCORE 0.3 之上）
IMPUTE_MAX_SPEED = 6.0     # 持有者框中心速度上限 px/帧（≈步行；跑动扫过锚点不算持有）


@dataclass
class ImputeStats:
    anchor_frames: int = 0       # degraded 且有锚点的帧
    holder_frames: int = 0       # 唯一持有者帧（框含锚点且 track_id 唯一）
    streaks: int = 0             # 开出的补全段数
    frames_imputed: int = 0      # 实际合成观测的帧数
    details: dict = field(default_factory=dict)


def _anchor_candidates(fused, max_gap: int) -> dict[int, tuple[float, float]]:
    """与 hand_roi.anchor_candidates 同逻辑（解耦避免跨模块状态）。"""
    cands: dict[int, tuple[float, float]] = {}
    last_obs = None
    last_trk = -10**9
    for i, f in enumerate(fused):
        if f.status == "tracking":
            last_obs = (f.cx, f.cy)
            last_trk = i
        elif last_obs is not None and i - last_trk <= max_gap:
            cands[i] = last_obs
    return cands


def impute_possession(fused, players_by_frame: dict[int, list[dict]],
                      world_to_pixel=None, pixel_to_world=None,
                      max_gap: int = IMPUTE_MAX_GAP, max_streak: int = IMPUTE_STREAK,
                      min_streak: int = IMPUTE_MIN_STREAK,
                      box_pad: float = IMPUTE_BOX_PAD,
                      hand_height_m: float = IMPUTE_HAND_HEIGHT,
                      max_speed: float = IMPUTE_MAX_SPEED,
                      follow_holder: bool = True,
                      ) -> tuple[dict[int, list[dict]], ImputeStats]:
    """识别持盘段并产出合成盘观测（原帧像素坐标）。

    fused: fuse_disc_detections 输出（与帧号对齐）；players_by_frame: 帧号→球员
    det（含 track_id/bbox）。world_to_pixel/pixel_to_world 提供时，合成点用手部
    世界坐标反投影（校正地平面视差）；缺省退化为框内手高比例点。

    follow_holder（默认开）：静态锚点只覆盖站立持盘——走动持盘者的框会离开
    最后观测点。开启后锚点逐帧跟随持有者手部点，身份连续+速度门+段上限
    约束漂移（§13.14 迭代2）。
    返回 ({帧号: [{bbox, conf, source:"possession_imputed"}]}, stats)。
    """
    stats = ImputeStats()
    anchors = _anchor_candidates(fused, max_gap)
    stats.anchor_frames = len(anchors)

    def hand_pixel(det) -> tuple[float, float] | None:
        x1, y1, x2, y2 = (float(v) for v in det["bbox"])
        if world_to_pixel is None or pixel_to_world is None:
            return (x1 + x2) / 2, y1 + 0.30 * (y2 - y1)  # 框内手高比例点（近似）
        wx, wy = pixel_to_world((x1 + x2) / 2, y2)        # 脚点→世界
        return world_to_pixel(wx, wy + hand_height_m)      # 手部点→像素（视差校正）

    extra: dict[int, list[dict]] = {}
    i = 0
    n = len(fused)
    while i < n:
        if i not in anchors:
            i += 1
            continue
        # 向后收集同 track_id 连续含锚点的段（含速度门：跑动者扫过锚点不算持有）
        anchor = anchors[i]
        holder_tid, streak_frames = None, []
        identity_changed = False
        j = i
        while j < n and j in anchors:
            hit = None
            for det in players_by_frame.get(j, []):
                x1, y1, x2, y2 = (float(v) for v in det["bbox"])
                pad_w, pad_h = (x2 - x1) * box_pad, (y2 - y1) * box_pad
                if (x1 - pad_w) <= anchor[0] <= (x2 + pad_w) and \
                        (y1 - pad_h) <= anchor[1] <= (y2 + pad_h):
                    if hit is not None and det.get("track_id") != hit.get("track_id"):
                        hit = None  # 两名球员同时含锚点 → 身份歧义，弃
                        break
                    hit = det
            if hit is None:
                break
            # 速度门：与上一帧同 track 框中心位移超阈值 → 非站立持盘，段终止
            if streak_frames:
                px1, pdet = streak_frames[-1]
                bx1, by1, bx2, by2 = (float(v) for v in pdet["bbox"])
                cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
                cx0, cy0 = (bx1 + bx2) / 2, (by1 + by2) / 2
                if math.hypot(cx1 - cx0, cy1 - cy0) > max_speed * max(j - px1, 1):
                    break
            tid = hit.get("track_id")
            if holder_tid is None:
                holder_tid = tid
            elif tid != holder_tid:
                identity_changed = True  # 换人须由真实观测接手，整段跳过不重开
                break
            streak_frames.append((j, hit))
            if follow_holder:
                hp = hand_pixel(hit)
                if hp is not None:
                    anchor = hp  # 锚点跟随持有者手部点（漂移受身份+速度门约束）
            j += 1
        if identity_changed:
            # 整段锚点区全部作废：同一静止锚点先后落进两名球员框，归属不清，
            # 从头到尾都不补（换人须由真实观测接手）
            while j < n and j in anchors:
                j += 1
            i = j
            continue
        if len(streak_frames) >= min_streak:
            stats.streaks += 1
            for fi, det in streak_frames[:max_streak]:
                hp = hand_pixel(det)
                if hp is None:
                    continue
                s = IMPUTE_BBOX_SIZE / 2
                extra[fi] = [{
                    "bbox": [round(hp[0] - s, 1), round(hp[1] - s, 1),
                             round(hp[0] + s, 1), round(hp[1] + s, 1)],
                    "conf": IMPUTE_CONF,
                    "source": "possession_imputed",
                }]
                stats.frames_imputed += 1
            stats.holder_frames += len(streak_frames)
            i = j
        else:
            i += 1
    return extra, stats
