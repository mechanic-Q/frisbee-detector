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

def test_track_confirmation_window(monkeypatch):
    """开轨确认：MIN_TRACK_QUALITY>1 时单帧假盘不成轨（§13.16 设计）。

    默认 =1（关闭）：chunk1 实测 =3 时短轨迹 tracking -42%，弊大于利（诚实账
    见 §13.16）；常量保留，monkeypatch 验证机制本身。
    """
    from frisbee_analyzer import disc_fusion

    one = [[{"bbox": [50, 50, 66, 66], "conf": 0.9}]]  # 只出现 1 帧
    fused, _ = fuse_disc_detections(one, fps=30.0)
    assert fused[0].status == "tracking", "默认确认=1 时单帧即成轨（旧行为）"

    monkeypatch.setattr(disc_fusion, "MIN_TRACK_QUALITY", 3)
    fused1, _ = fuse_disc_detections(one, fps=30.0)
    assert all(f.status == "searching" for f in fused1), "确认窗>1 时单帧不得成轨"

    three = [[{"bbox": [50, 50, 66, 66], "conf": 0.9}]] * 3
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


# ── §13.18 断轨续接强化 ──

def test_reacquire_after_gap():
    """断档 25 帧（>旧 LOST_TRACK_THRESHOLD=15，<新 REACQ_WINDOW=30）后盘重现 → 续接。"""
    base = []
    for i in range(10):
        cx = 100.0 + 3 * i
        base.append([{"bbox": [cx - 8, 192, cx + 8, 208], "conf": 0.8}])
    gap = [[] for _ in range(25)]                     # 旧逻辑 15 帧即破产
    reappear = [{"bbox": [130 - 8, 192, 130 + 8, 208], "conf": 0.7}]  # 预测/最后观测邻域
    fused, stats = fuse_disc_detections(base + gap + [reappear], fps=25.0)
    assert fused[-1].status == "tracking", f"断档重现应续接: {fused[-1].status}"
    assert stats.longest_track_frames >= 11


def test_reacquire_radius_hard_cap():
    """续接段候选超出搜索圆 → 不接（防漂移乱接）。"""
    base = []
    for i in range(10):
        cx = 100.0 + 3 * i
        base.append([{"bbox": [cx - 8, 192, cx + 8, 208], "conf": 0.8}])
    gap = [[] for _ in range(25)]
    far = [{"bbox": [1200 - 8, 192, 1200 + 8, 208], "conf": 0.9}]  # 距最后观测 ~1100px，圆外
    fused, _ = fuse_disc_detections(base + gap + [far], fps=25.0)
    assert fused[-1].status != "tracking", "远端候选不得借续接乱接"


def test_reacq_window_expiry():
    """断档超过 REACQ_WINDOW（30 帧）→ 轨迹破产 searching。"""
    base = [[{"bbox": [100 - 8, 192, 100 + 8, 208], "conf": 0.8}]] * 5
    long_gap = [[] for _ in range(40)]
    fused, stats = fuse_disc_detections(base + long_gap, fps=25.0)
    assert stats.n_lost >= 1, "超窗应破产"
    assert fused[-1].status == "searching"
