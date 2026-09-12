"""Gate-A: disc_fusion 融合模块单元测试（纯合成数据，无 I/O 无模型）。

判据（F1 方案 Phase A）:
- T1 合成匀速轨迹+干扰框，关联准确率 100%
- T2 注入 >3σ 野值全被拒（gated）且轨迹不中断（降级纯预测后恢复）
- T3 速度拒绝边界（>25 m/s 拒 / ≤20 m/s 留）
- T4 10 帧遮挡后重关联成功率 ≥90%
- T5 模块不 import cv2/torch
"""

from __future__ import annotations

import importlib
import sys

import pytest

sys.path.insert(0, ".")

from frisbee_analyzer import disc_fusion
from frisbee_analyzer.disc_fusion import (
    MIN_TRACK_QUALITY,
    DiscFrame,
    FusionStats,
    Kalman2D,
    fuse_disc_detections,
    score_candidates,
)


# ── 合成数据工具 ──

def make_trajectory(n=50, start=(100.0, 200.0), vel=(6.0, 3.0), jitter=0.0, seed=0):
    """生成匀速运动真值检测序列（每帧一个候选）。"""
    import random
    rng = random.Random(seed)
    out = []
    x, y = start
    for _ in range(n):
        jx = rng.uniform(-jitter, jitter) if jitter else 0.0
        jy = rng.uniform(-jitter, jitter) if jitter else 0.0
        cx, cy = x + jx, y + jy
        out.append([{"bbox": [cx - 10, cy - 10, cx + 10, cy + 10], "conf": 0.8}])
        x += vel[0]
        y += vel[1]
    return out


# ── T5: 纯度 ──

def test_t5_no_cv2_torch_import():
    """AST 级检查：模块的 import 语句不得含 cv2/torch/ultralytics（docstring 提及不算）。"""
    import ast
    tree = ast.parse(open("frisbee_analyzer/disc_fusion.py", encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"cv2", "torch", "ultralytics"}
    assert not (imported & banned), f"发现禁止依赖: {imported & banned}"
    allowed = {"__future__", "math", "collections", "dataclasses", "numpy"}
    assert imported <= allowed | {"frisbee_analyzer", "utils"}, f"意外 import: {imported - allowed}"


# ── T1: 关联准确率 ──

def test_t1_clean_trajectory_tracked_throughout():
    """50 帧匀速轨迹（有 3 个干扰框/帧）应全程 tracking，位置误差 ≤ 感知阈值。"""
    dets = []
    for i, frame in enumerate(make_trajectory(50)):
        # 干扰框：静止的帽子/草纹（远离轨迹）
        frame = frame + [
            {"bbox": [900, 900, 920, 920], "conf": 0.6},
            {"bbox": [400, 800, 430, 830], "conf": 0.7},
            {"bbox": [1200, 300, 1230, 320], "conf": 0.5},
        ]
        dets.append(frame)
    frames, stats = fuse_disc_detections(dets, fps=25.0)
    tracking = [f for f in frames if f.status == "tracking"]
    assert len(tracking) >= 48, f"tracking 帧数不足: {len(tracking)}/50"
    # 位置连续性：相邻 tracking 帧位移应≈速度(6,3)→7.0px
    import math
    jumps = [math.hypot(b.cx - a.cx, b.cy - a.cy)
             for a, b in zip(tracking, tracking[1:])]
    assert max(jumps) < 15.0, f"轨迹跳变: {max(jumps):.1f}"
    assert stats.n_gated == 0 and stats.n_rejected_speed == 0


def test_t1_interference_never_hijacks():
    """高置信干扰框出现在轨迹附近 15px 内也不应劫持（评分偏向运动一致）。"""
    dets = make_trajectory(30)
    dets[10].insert(0, {"bbox": [dets[10][0]["bbox"][0] + 15, dets[10][0]["bbox"][1],
                                 dets[10][0]["bbox"][2] + 15, dets[10][0]["bbox"][3]], "conf": 0.95})
    frames, _ = fuse_disc_detections(dets, fps=25.0)
    # 被劫持的判据：第 10 帧后轨迹跳去干扰框位置（+15px 偏移）且不再回来
    t10 = frames[10]
    t12 = frames[12]
    assert t10.status == "tracking" and t12.status == "tracking"
    assert abs(t12.cx - t10.cx) < 30  # 仍在真轨迹邻域


# ── T2: 马氏门控 ──

def test_t2_outliers_gated_and_track_recovers():
    """50 帧轨迹中第 25 帧注入远端野值：应 gated（降级预测），第 26 帧恢复 tracking。"""
    dets = make_trajectory(50)
    # 野值：远端高置信
    dets[25] = [{"bbox": [1700, 1000, 1740, 1040], "conf": 0.9}]
    frames, stats = fuse_disc_detections(dets, fps=25.0)
    assert frames[25].status in ("gated", "predicting"), f"野值帧状态: {frames[25].status}"
    assert frames[26].status == "tracking", "野值后一帧应恢复 tracking"
    assert stats.n_gated >= 1
    # 轨迹未被污染：第 24/26 帧位置连续
    import math
    jump = math.hypot(frames[26].cx - frames[24].cx, frames[26].cy - frames[24].cy)
    assert jump < 25.0, f"野值污染了轨迹: {jump:.1f}"


def test_t2_gate_threshold_respected():
    """门控阈值参数化生效；连续门拒 ≥3 帧后圆判据接管（§13.18 门拒续接）。

    旧行为"极小阈值大量 gate"已被 §13.18 门拒续接取代——门拒 3 帧即触发
    最后观测点+外推圆判据，真盘候选（圆内）被接回 tracking。断言改为：
    确认仍有 gate 帧产生（机制存在），且轨迹存活（不破产）。
    """
    dets = make_trajectory(50)
    frames, stats = fuse_disc_detections(dets, fps=25.0, gate_threshold=0.0001)
    gated = [f for f in frames if f.status == "gated"]
    assert len(gated) >= 1, "极小阈值初期应产生 gate 帧"
    trk = [f for f in frames if f.status == "tracking"]
    assert len(trk) >= 20, f"门拒续接应保住轨迹: {len(trk)} tracking"
    assert stats.longest_track_frames >= 20


# ── T3: 速度拒绝 ──

def test_t3_speed_rejection_boundary():
    """带投影函数时：>25 m/s 的瞬移观测被 rejected，≤20 m/s 正常接受。

    §13.16 开轨确认：MIN_TRACK_QUALITY=3 帧内输出 searching，故断言限定
    确认窗之后的帧（轨迹前 2 帧不产生 tracking 输出）。
    """
    scale = 1.0  # 1px = 1m：0.8px/帧 @25fps = 20 m/s（保留）；200px/帧 = 200 m/s（拒绝）
    # proj 映到场内（§13.19 场线先验默认关，但显式投影时应给场内坐标）
    proj = lambda cx, cy: (cx * scale, min(cy * scale, 20.0))
    # 正常轨迹 0.8px/帧 → 20 m/s（边界内保留）
    dets_ok = make_trajectory(20, vel=(0.8, 0.0))
    frames, stats = fuse_disc_detections(dets_ok, fps=25.0, world_projector=proj)
    assert all(f.status == "tracking" for f in frames[MIN_TRACK_QUALITY - 1:]), \
        "20m/s 不应被拒（确认窗后全部 tracking）"
    assert stats.n_rejected_speed == 0
    # 瞬移：第 10 帧跳 200px → 200 m/s（必须拒绝）
    dets_tp = make_trajectory(20, vel=(0.8, 0.0))
    dets_tp[10] = [{"bbox": [2000, 200, 2020, 220], "conf": 0.9}]
    frames2, stats2 = fuse_disc_detections(dets_tp, fps=25.0, world_projector=proj)
    assert stats2.n_rejected_speed >= 1, "200m/s 瞬移必须被拒"
    assert frames2[10].status == "rejected"


# ── T4: 断轨续接 ──

def test_t4_occlusion_reacquire():
    """10 帧全空遮挡后目标重现：应从纯预测恢复 tracking，位置在预测邻域。"""
    base = make_trajectory(40, vel=(8.0, 4.0))
    dets = base[:15] + [[] for _ in range(10)] + base[25:]
    frames, stats = fuse_disc_detections(dets, fps=25.0)
    after = frames[25]
    assert after.status == "tracking", f"遮挡后应恢复: {after.status}"
    # 恢复点应接近真值轨迹第 25 帧位置（15+10 帧匀速外推）
    truth_x = 100.0 + 8.0 * 25
    truth_y = 200.0 + 4.0 * 25
    import math
    err = math.hypot(after.cx - truth_x, after.cy - truth_y)
    assert err < 30.0, f"重关联位置误差过大: {err:.1f}px"


def test_t4_multi_occlusion_90pct():
    """5 处 10 帧遮挡（间隔 ≥18 帧，不背靠背超 LOST 上限），重关联成功率 ≥90%。"""
    n = 140
    base = make_trajectory(n, vel=(6.0, 2.0), seed=3)
    occlusions = [10, 38, 66, 94, 122]  # 每段 10 帧空，间隔 18 帧
    dets = []
    for i in range(n):
        if any(o <= i < o + 10 for o in occlusions):
            dets.append([])
        else:
            dets.append(base[i])
    frames, stats = fuse_disc_detections(dets, fps=25.0)
    recovered = 0
    for o in occlusions:
        recovered += 1 if frames[o + 10].status == "tracking" else 0
    assert recovered / len(occlusions) >= 0.9, f"重关联成功率 {recovered}/{len(occlusions)}"
