"""events_runner 适配层单测（纯 CPU，合成 tracks doc + 合成标定）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frisbee_analyzer.events_runner import (  # noqa: E402
    build_end_zones, compute_events, load_calibration, px_to_world,
)


def make_calibration(tmp_path: Path, scale: float = 10.0):
    """简易标定：像素 (0,0)-(100,37) 映射到世界 (0,0)-(100,37)（缩放 scale 倍）。"""
    # 像素点 (0,0)->(0,0), (1000,0)->(100,0), (1000,370)->(100,37)
    src = np.array([[0, 0], [1000, 0], [1000, 370], [0, 370]], dtype=np.float64)
    dst = np.array([[0, 0], [100, 0], [100, 37], [0, 37]], dtype=np.float64)
    H, _ = __import__("cv2").findHomography(src, dst, 0)
    cal = {
        "homography": np.asarray(H).reshape(3, 3).tolist(),
        "field_size_m": [100.0, 37.0],
        "end_zone_depth_m": 18.0,
    }
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(cal))
    return p


def px(x_world: float, y_world: float) -> tuple:
    """世界 → 像素（上面标定的逆：x*10, y*10）"""
    return x_world * 10.0, y_world * 10.0


def test_load_calibration_roundtrip(tmp_path):
    p = make_calibration(tmp_path)
    H, (fw, fh), ez = load_calibration(p)
    assert H.shape == (3, 3)
    assert (fw, fh) == (100.0, 37.0)
    assert ez == 18.0
    x, y = px_to_world(H, 500, 185)
    assert abs(x - 50) < 0.5 and abs(y - 18.5) < 0.5


def test_build_end_zones_geometry():
    ezs = build_end_zones(100.0, 37.0, 18.0, team_left=0)
    assert len(ezs) == 2
    assert ezs[0].team_attacking == 0 and ezs[1].team_attacking == 1
    assert ezs[0].contains(5, 18) and not ezs[0].contains(50, 18)
    assert ezs[1].contains(95, 18) and not ezs[1].contains(50, 18)


def test_compute_events_pass_produces_transfer(tmp_path):
    """0 队 101 持盘 → 传给同队 102：应产出 1 次 transfer、0 turnover。"""
    cal = make_calibration(tmp_path)
    frames = {}
    disc_frames = {}
    # 0-9 帧：101 在 (30,18) 持盘
    for i in range(10):
        x, y = px(30, 18)
        frames[str(i)] = [{"track_id": 101, "bbox": [x - 20, y - 60, x + 20, y], "conf": 0.9,
                           "cls": 0, "team_id": 0}]
        dx, dy = px(30, 19.3)
        disc_frames[str(i)] = {"cx": dx, "cy": dy, "conf": 0.9, "bbox": [dx - 5, dy - 5, dx + 5, dy + 5]}
    # 10-12 帧：盘飞向 102（位置 40,19）
    for i, (wx, wy) in enumerate([(32, 18.6), (36, 19.0), (40, 19.2)], start=10):
        for tid, (tx, ty) in ((101, (30, 18)), (102, (40, 19))):
            x, y = px(tx, ty)
            frames.setdefault(str(i), []).append(
                {"track_id": tid, "bbox": [x - 20, y - 60, x + 20, y], "conf": 0.9, "cls": 0, "team_id": 0})
        dx, dy = px(wx, wy)
        disc_frames[str(i)] = {"cx": dx, "cy": dy, "conf": 0.9, "bbox": [dx - 5, dy - 5, dx + 5, dy + 5]}
    # 13-25 帧：102 持盘
    for i in range(13, 26):
        for tid, (tx, ty) in ((101, (30, 18)), (102, (40, 19))):
            x, y = px(tx, ty)
            frames.setdefault(str(i), []).append(
                {"track_id": tid, "bbox": [x - 20, y - 60, x + 20, y], "conf": 0.9, "cls": 0, "team_id": 0})
        dx, dy = px(40, 20.3)
        disc_frames[str(i)] = {"cx": dx, "cy": dy, "conf": 0.9, "bbox": [dx - 5, dy - 5, dx + 5, dy + 5]}

    doc = {"fps": 30.0, "frames": frames, "disc_frames": disc_frames, "team_overrides": {}}
    out = compute_events(doc, cal)
    types = [e["type"] for e in out["events"]]
    assert types.count("transfer") == 1, types
    assert types.count("turnover") == 0, types
    tr = next(e for e in out["events"] if e["type"] == "transfer")
    assert tr["from_track"] == 101 and tr["to_track"] == 102


def test_compute_events_no_disc_yields_no_false_possession(tmp_path):
    """无盘轨迹时不得产生持盘/交换手事件（验证报告的核心结论）。"""
    cal = make_calibration(tmp_path)
    frames = {str(i): [{"track_id": 101, "bbox": [px(30, 18)[0] - 20, px(30, 18)[1] - 60,
                                                  px(30, 18)[0] + 20, px(30, 18)[1]],
                        "conf": 0.9, "cls": 0, "team_id": 0}] for i in range(30)}
    doc = {"fps": 30.0, "frames": frames, "disc_frames": {}, "team_overrides": {}}
    out = compute_events(doc, cal)
    types = [e["type"] for e in out["events"]]
    assert "possession_start" not in types
    assert "transfer" not in types
    assert out["score"] == {0: 0, 1: 0}
