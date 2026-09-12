"""Gate: segment_calib 段内自动标定（pipeline 内联版）单元测试。

判据:
- SC1 合成均匀场内脚点 → 标定通过门（内点率 ≥0.85）且矩阵可把脚点投回场内
- SC2 点数不足 → 拒绝并给原因
- SC3 角点退化（所有点一条线）→ 拒绝
- SC4 纯度：不 import cv2/torch
"""

from __future__ import annotations

import ast
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from frisbee_analyzer.segment_calib import (
    _pixel_to_world,
    auto_calibrate_segment,
    collect_foot_points,
)


def _synth_frames(n_players=14, n_frames=40, seed=0):
    """合成 frames：球员均匀分布在场地区域（像素↔世界用已知可逆映射模拟）。

    直接构造"世界→像素"为恒等缩放的简化场景（1px=0.1m），检验算法本身。
    """
    import random
    rng = random.Random(seed)
    frames = {}
    scale = 10.0  # 1m = 10px
    w_px, h_px = 1000, 400
    for f in range(n_frames):
        dets = []
        for p in range(n_players):
            wx = rng.uniform(5, 95)
            wy = rng.uniform(3, 34)
            cx, foot_y = wx * scale, wy * scale
            hgt = rng.uniform(25, 40)
            dets.append({"bbox": [cx - 10, foot_y - hgt, cx + 10, foot_y],
                         "conf": 0.7, "cls": "player"})
        frames[str(f)] = dets
    return frames, w_px, h_px, scale


def test_sc1_synthetic_field_passes_gate():
    frames, w, h, scale = _synth_frames()
    calib, report = auto_calibrate_segment(frames, w, h)
    assert calib is not None, f"应过门: {report}"
    assert report["passed"] is True
    assert report["optimized_inlier_ratio"] >= 0.85
    # 抽一个场内真值点验证矩阵方向正确
    m = np.array(calib["matrix"])
    r = _pixel_to_world(m, 50 * scale, 20 * scale)
    assert r is not None and 0 <= r[0] <= 100 and 0 <= r[1] <= 37


def test_sc2_too_few_points_rejected():
    frames, w, h, _ = _synth_frames(n_frames=3)  # 3*14=42 点 < 500
    calib, report = auto_calibrate_segment(frames, w, h)
    assert calib is None
    assert "too few" in report.get("reason", "")


def test_sc3_degenerate_line_rejected():
    # 所有点共线（y 固定）→ 角点初始化退化
    frames = {}
    for f in range(40):
        frames[str(f)] = [{"bbox": [float(10 * i % 900), 100.0, float(10 * i % 900) + 20, 200.0],
                           "conf": 0.7} for i in range(14)]
    calib, report = auto_calibrate_segment(frames, 1000, 400)
    # 共线时 near/far 带重合 → init 退化或内点率不过门，两者都算正确拒绝
    assert calib is None or report.get("passed") is False


def test_sc4_no_cv2_torch():
    tree = ast.parse(open("frisbee_analyzer/segment_calib.py", encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"cv2", "torch", "ultralytics"})


def test_collect_foot_points_filters_small():
    frames = {"0": [
        {"bbox": [0, 0, 20, 10], "conf": 0.5},    # 高 10 < 400*0.05=20 → 过滤
        {"bbox": [0, 0, 20, 30], "conf": 0.5},    # 高 30 → 保留
    ]}
    pts = collect_foot_points(frames, 400.0)
    assert len(pts) == 1
    assert pts[0][1] == 30.0


def test_sc5_recovery_finds_passing_window():
    """整段被场外离群簇拖 FAIL，滑窗恢复找到干净子窗（§13.15）。"""
    from frisbee_analyzer.segment_calib import auto_calibrate_segment_recover

    frames, w, h, scale = _synth_frames(n_frames=40, seed=3)
    # 前 120 帧"场外簇"：脚点远在画面下方，钳位比例的矩形盖不住两簇 → 整段 FAIL
    rng = np.random.default_rng(7)
    for f in range(120):
        dets = []
        for _ in range(20):
            cx, foot_y = rng.uniform(900, 1000), rng.uniform(2000, 2100)
            hgt = rng.uniform(25, 40)
            dets.append({"bbox": [cx - 10, foot_y - hgt, cx + 10, foot_y],
                         "conf": 0.7, "cls": "player"})
        frames[str(f)] = dets
    ordered = {f"{int(k) + 120}": v for k, v in frames.items()}  # 干净窗放后半段
    calib, report = auto_calibrate_segment_recover(ordered, w, h)
    assert calib is not None, f"恢复应找到过门子窗: {report}"
    assert report.get("recovery_window") is not None
    assert report["recovery_window"][0] >= 120  # 找到的是干净子窗
    assert report["optimized_inlier_ratio"] >= 0.85
