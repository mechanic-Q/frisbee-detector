"""Phase A §13.16 融合强化单元测试：门控重试 / tile 搜索态准入 / IOS-NMM。

运行: python -m pytest tests/test_disc_fusion_retry.py tests/test_tracking_tile.py -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.disc_fusion import fuse_disc_detections  # noqa: E402
from frisbee_analyzer.tracking import _nmm_ios, detect_disc_candidates  # noqa: E402


# ── 门控重试：评分最优候选被门拒后，次优真候选应胜出 ──

def test_gate_retry_rescues_true_candidate():
    """跟踪态：远端假盘 conf 更高，但贴预测真盘应因马氏距离序重试胜出。

    注：真盘必须贴预测（Kalman 稳态 P≈0.12，>5px 跳变即 d2 超阈——§13.7
    已知过信特性）。本测试验证的是"评分最优≠门控最优"的修复。
    """
    # 建轨迹：6 帧向右匀速 (5,0)/帧
    base = []
    for i in range(6):
        cx = 100.0 + 5 * i
        base.append([{"bbox": [cx - 8, 192, cx + 8, 208], "conf": 0.8}])
    # 第 7 帧：真盘贴预测（cx=130，d2≈0，conf 0.6）；远端假盘（conf 0.95）
    true_cx = 100.0 + 5 * 6  # =130 = 预测位置
    base.append([
        {"bbox": [true_cx - 8, 192, true_cx + 8, 208], "conf": 0.6},
        {"bbox": [true_cx + 300 - 8, 192, true_cx + 300 + 8, 208], "conf": 0.95},
    ])
    fused, _ = fuse_disc_detections(base, fps=25.0)
    last = fused[-1]
    assert last.status == "tracking"
    assert abs(last.cx - true_cx) < 10, f"真盘应胜出: got cx={last.cx}"


def test_all_gated_downgrades_with_status():
    """全部候选过不了门 → 帧降级 gated；n_gated 计帧级。"""
    base = []
    for i in range(5):
        cx = 100.0 + 5 * i
        base.append([{"bbox": [cx - 8, 192, cx + 8, 208], "conf": 0.8}])
    # 第 6 帧只有一个远端野值
    base.append([{"bbox": [1000 - 8, 192, 1000 + 8, 208], "conf": 0.9}])
    fused, stats = fuse_disc_detections(base, fps=25.0, gate_threshold=13.82)
    assert fused[-1].status == "gated"
    assert stats.n_gated == 1


# ── tile 搜索态准入 ──

def test_tile_search_state_high_admission():
    """搜索态（无轨迹）：低分 tile 候选被拒，≥TILE_MIN_SCORE 的 tile 可开轨。"""
    from frisbee_analyzer.disc_fusion import TILE_MIN_SCORE
    low = [[{"bbox": [50, 50, 66, 66], "conf": 0.4, "source": "tile"}]] * 4
    fused_low, _ = fuse_disc_detections(low, fps=30.0)
    assert all(f.status == "searching" for f in fused_low), "低分 tile 搜索态不得开轨"

    ok = [[{"bbox": [50, 50, 66, 66], "conf": TILE_MIN_SCORE, "source": "tile"}]] * 4
    fused_ok, _ = fuse_disc_detections(ok, fps=30.0)
    assert fused_ok[-1].status == "tracking", "达到准入线的 tile 应可开轨"


def test_full_frame_low_conf_still_admitted():
    """搜索态整帧低分候选不受 tile 准入约束（保持 MIN_SCORE=0.3 旧行为）。"""
    dets = [[{"bbox": [50, 50, 66, 66], "conf": 0.4, "source": "full"}]] * 4
    fused, _ = fuse_disc_detections(dets, fps=30.0)
    assert fused[-1].status == "tracking"


# ── 开轨确认 ──

def test_track_confirmation_window():
    """单帧假盘不得立即锁轨：前 MIN_TRACK_QUALITY-1 帧输出 searching。"""
    from frisbee_analyzer.disc_fusion import MIN_TRACK_QUALITY
    one = [[{"bbox": [50, 50, 66, 66], "conf": 0.9}]]  # 只出现 1 帧
    fused, _ = fuse_disc_detections(one, fps=30.0)
    assert all(f.status == "searching" for f in fused), "单帧不得成轨"

    three = [[{"bbox": [50, 50, 66, 66], "conf": 0.9}]] * MIN_TRACK_QUALITY
    fused3, _ = fuse_disc_detections(three, fps=30.0)
    assert fused3[-1].status == "tracking", "确认窗后应升级 tracking"


# ── IOS-NMM ──

def test_nmm_ios_merges_contained_boxes():
    """切片大框包含整帧小框（IOS 高）→ 合并保留高分；远框保留。"""
    boxes = [[100, 100, 120, 120], [98, 98, 124, 124], [500, 500, 520, 520]]
    scores = [0.6, 0.9, 0.8]
    keep = _nmm_ios(boxes, scores, ios_thr=0.5)
    assert keep == [1, 2], "包含框应被合并，高分者保留；远框独立"


def test_nmm_ios_keeps_distinct_small_boxes():
    """两个互不交叠的小框（IOS=0）都保留。"""
    keep = _nmm_ios([[10, 10, 26, 26], [60, 60, 76, 76]], [0.8, 0.7])
    assert len(keep) == 2
