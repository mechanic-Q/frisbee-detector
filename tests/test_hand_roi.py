"""frisbee_analyzer 手部 ROI 复检单元测试（纯几何，无 CV 依赖）。

运行: python -m pytest tests/test_hand_roi.py -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.disc_fusion import DiscFrame, fuse_disc_detections  # noqa: E402
from frisbee_analyzer.hand_roi import (  # noqa: E402
    ROI_DEDUP_IOU,
    anchor_candidates,
    hand_roi_box,
    merge_hand_roi_detections,
    select_holders,
)


# ── hand_roi_box：躯干横带 + 双侧扩展 + 画面钳位 ──

def test_hand_roi_box_pads_and_band():
    # 球员框 (100,100)-(140,180)：bw=40 bh=80
    roi = hand_roi_box([100, 100, 140, 180])
    assert roi == (100 - 0.6 * 40, 100 + 0.15 * 80, 140 + 0.6 * 40, 180 + 0.25 * 80)


def test_hand_roi_box_clamps_to_frame():
    roi = hand_roi_box([0, 0, 50, 100], frame_w=1920, frame_h=480)
    assert roi[0] == 0.0                    # 左侧外扩被钳到 0
    assert roi[1] == 0.15 * 100             # 顶部下移不受钳位影响
    roi2 = hand_roi_box([1900, 400, 1960, 470], frame_w=1920, frame_h=480)
    assert roi2[2] == 1920.0 and roi2[3] == 480.0  # 右/下越界被钳回画面内


# ── select_holders：ROI 含锚点 或 中心距 < near_frac×框高 ──

def test_select_holders_by_containment_and_distance():
    players = [
        {"bbox": [90, 100, 130, 180]},    # ROI 含锚点 (110, 150)
        {"bbox": [500, 100, 540, 180]},   # 很远
        {"bbox": [190, 120, 230, 200]},   # 中心 (210,160) 距锚点 ~100 < 1.2*80=96? 否 → 排除
    ]
    picked = select_holders(players, (110, 150))
    assert len(picked) == 1
    assert picked[0]["bbox"] == [90, 100, 130, 180]

    # 放近第二个：中心 (210,160) 距 (150,160) = 60 < 96 → 入选；含锚点的球员 dist≈44.7 更近排前
    players[2]["bbox"] = [190, 120, 230, 200]
    picked = select_holders(players, (150, 160))
    assert [p["bbox"] for p in picked] == [[90, 100, 130, 180], [190, 120, 230, 200]]


def test_select_holders_k_limit():
    players = [{"bbox": [10 + 80 * i, 100, 30 + 80 * i, 180]} for i in range(3)]
    anchor = (20, 140)  # bh=80, near=96：dist 0 / 80 / 160 → 前两人入选
    picked = select_holders(players, anchor, k=2)
    assert len(picked) == 2
    assert picked[0]["bbox"] == [10, 100, 30, 180]  # 最近的排最前
    assert picked[1]["bbox"] == [90, 100, 110, 180]


# ── anchor_candidates：非 tracking 帧挂最后观测锚点，超 max_gap 丢弃 ──

def _df(status, cx=10.0, cy=20.0):
    return DiscFrame([cx - 5, cy - 5, cx + 5, cy + 5], 0.9, cx, cy, status)


def test_anchor_candidates_gaps():
    fused = [_df("tracking")] + [_df("searching")] * 3 + \
            [_df("tracking", cx=50, cy=60)] + [_df("predicting")] * 2
    cand = anchor_candidates(fused, max_gap=2)
    # 帧1-3 距帧0 的 tracking ≤2 只覆盖帧1,2；帧3 距帧0 =3 超 gap、距帧4 =1（帧4 在其后未发生）→ 不在
    assert cand == {1: (10.0, 20.0), 2: (10.0, 20.0),
                    5: (50.0, 60.0), 6: (50.0, 60.0)}


def test_anchor_candidates_no_track_no_candidates():
    fused = [_df("searching"), _df("predicting")]
    assert anchor_candidates(fused) == {}


def test_anchor_candidates_max_gap_expires():
    fused = [_df("tracking")] + [_df("searching")] * 5
    assert anchor_candidates(fused, max_gap=3) == {1: (10.0, 20.0), 2: (10.0, 20.0), 3: (10.0, 20.0)}


# ── merge_hand_roi_detections：并入 + IoU 去重 ──

def test_merge_adds_and_dedups():
    seq = [[{"bbox": [10, 10, 20, 20], "conf": 0.8}], []]
    extra = {
        0: [{"bbox": [11, 11, 21, 21], "conf": 0.9, "source": "hand_roi"}],  # IoU≈0.68 → 去重
        1: [{"bbox": [100, 100, 110, 110], "conf": 0.9, "source": "hand_roi"}],
    }
    seq2, stats = merge_hand_roi_detections(seq, extra)
    assert len(seq2[0]) == 1 and seq2[0][0]["conf"] == 0.8  # 原 det 保留，ROI 被丢
    assert len(seq2[1]) == 1 and seq2[1][0]["source"] == "hand_roi"
    assert stats.dets_added == 2 and stats.dets_deduped == 1
    # 原 seq 不被改写
    assert len(seq[1]) == 0


def test_merge_iou_boundary_exact():
    seq = [[{"bbox": [0, 0, 10, 10], "conf": 0.8}]]
    # IoU=0.5 恰好不大于阈值 → 保留
    extra = {0: [{"bbox": [0, 0, 10, 20], "conf": 0.9, "source": "hand_roi"}]}
    seq2, stats = merge_hand_roi_detections(seq, extra)
    assert len(seq2[0]) == 2 and stats.dets_deduped == 0
    assert ROI_DEDUP_IOU == 0.5


# ── 融合层 source 透传 ──

def test_fusion_passes_source_through():
    # 同位 ROI 检出成轨后 source 透传（默认 MIN_TRACK_QUALITY=1 无确认窗）
    dets = [[{"bbox": [100, 100, 110, 110], "conf": 0.7, "source": "hand_roi"}]] * 5
    fused, _ = fuse_disc_detections(dets, fps=30.0)
    assert all(f.status == "tracking" for f in fused)
    assert all(f.source == "hand_roi" for f in fused)


def test_fusion_main_channel_source_none():
    dets = [[{"bbox": [100, 100, 110, 110], "conf": 0.7}]] * 5
    fused, _ = fuse_disc_detections(dets, fps=30.0)
    assert all(f.source is None for f in fused)
