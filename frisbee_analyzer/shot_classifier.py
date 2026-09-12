"""镜头类型分类器（§13.19：科学化识别率口径的前置）。

动机：WFDF 转播大量特写/半身镜头——盘不在全景画面里，把这些时间计入
"识别率分母"是不科学的。全景/特写的判别不需要训练模型：球员框的
**数量与尺寸分布**即是镜头语言的特征（全景 10+ 个小框；特写 0-3 个大框；
观众席 0 个框但有密集小目标……用框统计即可）。

口径定义（stats 消费）：
  wide   — 全景比赛画面：≥WIDE_MIN_PLAYERS 个球员框且中位框高 ≤WIDE_MAX_H_FRAC
  close  — 特写/半身：其余（含少量大框、无人框）
识别率 = 盘在管帧 / wide 帧（"可见时段识别率"）。

纯函数，只吃单帧球员检测列表。
"""
from __future__ import annotations

# ── 可调常量 ──
WIDE_MIN_PLAYERS = 6       # 全景最少球员框数（14 球员+2 裁判的战场，特写 0-3 人）
WIDE_MAX_H_FRAC = 0.30     # 全景球员框高上限（帧高占比；特写半身框高 >30%）
CLOSE_MIN_H_FRAC = 0.18    # 特写判定的大框下限（辅助）


def classify_shot(player_dets: list[dict], frame_h: float) -> str:
    """单帧镜头分类：'wide' | 'close' | 'empty'。

    empty：无球员框（导播切观众席/广告位——既非全景也非特写）。
    """
    if not player_dets:
        return "empty"
    n = len(player_dets)
    heights = []
    for det in player_dets:
        x1, y1, x2, y2 = det["bbox"]
        heights.append((y2 - y1) / max(frame_h, 1))
    heights.sort()
    med_h = heights[len(heights) // 2]
    max_h = heights[-1]
    if n >= WIDE_MIN_PLAYERS and med_h <= WIDE_MAX_H_FRAC:
        return "wide"
    if n <= 3 and max_h >= CLOSE_MIN_H_FRAC:
        return "close"
    # 中间态：4-5 人或框偏大——按人数就近归 wide（宁进分母，不虚高识别率）
    return "wide" if n >= 4 else "close"


def classify_sequence(per_frame_dets: dict[str, list], frame_h: float) -> dict:
    """整段分类统计：wide/close/empty 帧数与占比（provenance 用）。"""
    from collections import Counter
    counts = Counter()
    wide_frames = []
    for k, dets in per_frame_dets.items():
        shot = classify_shot(dets, frame_h)
        counts[shot] += 1
        if shot == "wide":
            wide_frames.append(int(k))
    total = sum(counts.values()) or 1
    return {
        "counts": dict(counts),
        "wide_frac": round(counts["wide"] / total, 3),
        "wide_frames": sorted(wide_frames),
        "constants": {"WIDE_MIN_PLAYERS": WIDE_MIN_PLAYERS,
                      "WIDE_MAX_H_FRAC": WIDE_MAX_H_FRAC},
    }
