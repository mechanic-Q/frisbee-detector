"""team.py 纯函数单测：不依赖 GPU/SigLIP/视频，embedding 可注入。"""

import json

import numpy as np
import pytest

from frisbee_analyzer.team import (
    MIN_CROPS_FOR_CLUSTER,
    apply_team_from_cls,
    cluster_team_embeddings,
    crop_upper_body,
    label_from_color,
    label_from_crops,
    majority_vote,
    team_ids_by_x,
    write_team_overrides,
)


class FakeEmbedder:
    """用裁剪图均值当 embedding——同均值聚成一簇，可构造两支"球衣色"。"""

    def embed(self, crops):
        return np.array([[float(np.mean(c))] * 4 for c in crops], dtype=np.float32)


def _make_crops(n, value):
    return [np.full((16, 16, 3), value, dtype=np.uint8) for _ in range(n)]


# ── 聚类 ─────────────────────────────────────────────────

def test_cluster_separates_two_teams():
    crops = _make_crops(8, 10) + _make_crops(8, 200)  # 两簇均值差异明显
    labels = cluster_team_embeddings(FakeEmbedder().embed(crops))
    assert labels is not None
    assert len(labels) == 16
    # 同簇内标签一致、两簇标签不同
    assert len(set(labels[:8])) == 1 and len(set(labels[8:])) == 1
    assert labels[0] != labels[8]


def test_cluster_too_few_returns_none():
    crops = _make_crops(MIN_CROPS_FOR_CLUSTER - 1, 10)
    assert cluster_team_embeddings(FakeEmbedder().embed(crops)) is None


# ── 半场先验定队号 ────────────────────────────────────────

def test_team_ids_by_x_orders_clusters():
    labels = np.array([0, 0, 1, 1])
    xs = np.array([10.0, 20.0, 80.0, 90.0])
    assert team_ids_by_x(labels, xs) == {0: 0, 1: 1}
    # 簇号翻转后映射跟着翻转，team 0 始终是 x 较小一侧
    labels_flipped = np.array([1, 1, 0, 0])
    assert team_ids_by_x(labels_flipped, xs) == {1: 0, 0: 1}


def test_majority_vote():
    mapping = {0: 1, 1: 0}
    assert majority_vote({7: [0, 0, 1], 8: [1, 1, 1, 0], 9: []}, mapping) == {7: 1, 8: 0}


# ── 端到端（注入 embedding）──────────────────────────────

def test_label_from_crops_two_teams():
    track_ids = [1] * 8 + [2] * 8
    crops = _make_crops(8, 10) + _make_crops(8, 200)
    xs = [10.0] * 8 + [90.0] * 8
    mapping = label_from_crops(track_ids, crops, xs, FakeEmbedder())
    assert mapping == {1: 0, 2: 1}  # track1 在左半场 → team0


def test_label_from_crops_empty():
    assert label_from_crops([], [], [], FakeEmbedder()) == {}


# ── 颜色特征聚类（v0 主路）────────────────────────────────

def test_label_from_color_separates_red_blue():
    feats = {i: (0.30, 0.01) for i in range(1, 6)}
    feats.update({i: (0.01, 0.28) for i in range(6, 11)})
    mapping, colors = label_from_color(feats)
    assert mapping is not None and len(mapping) == 10
    assert len(colors) == 2
    # 不变量：每个 track 的队色必须与自身红蓝占比一致（与 KMeans 原始标号无关）
    for t, (rf, bf) in feats.items():
        expect = "red" if rf >= bf else "blue"
        assert colors[str(mapping[t])] == expect, f"track {t}: {colors} vs ({rf},{bf})"


def test_label_from_color_weak_separation_returns_none():
    feats = {i: (0.05, 0.04) for i in range(1, 20)}  # 全员无色（如白衫）
    assert label_from_color(feats) == (None, None)


def test_label_from_color_rejects_signal_less_clusters():
    # 深色衫素材反例：两簇"可分"但红蓝信号都趋零且无对比 → 信号/对比门拒绝
    feats = {i: ((0.02, 0.01) if i < 10 else (0.01, 0.02)) for i in range(1, 21)}
    mapping, colors = label_from_color(feats, min_sep=0.01)  # 强行放宽分离度门
    assert mapping is None and colors is None


def test_label_from_color_accepts_one_colored_one_dark():
    # 55-56min 实测正例：红队 rf=0.295 vs 深色队 rf=0.028（对比 ~10x）→ 应采信颜色通道
    feats = {i: (0.295, 0.002) for i in range(1, 11)}
    feats.update({i: (0.028, 0.016) for i in range(11, 21)})
    mapping, colors = label_from_color(feats)
    assert mapping is not None and len(colors) == 2
    # 关键断言：红衣队（team0=主色偏红序）与深色队分属不同 team，且队号稳定
    red_teams = {mapping[i] for i in range(1, 11)}
    dark_teams = {mapping[i] for i in range(11, 21)}
    assert red_teams == {0} and dark_teams == {1}  # rf-bf 降序 → 红簇=team0
    assert colors["0"] == "red"
    assert colors["1"] == "dark"  # 无信号簇诚实标 dark，不硬凑红蓝


def test_label_from_color_names_both_colors_when_both_signal():
    # 红蓝双有信号：两侧都按簇心真实颜色命名
    feats = {i: (0.30, 0.01) for i in range(1, 6)}
    feats.update({i: (0.01, 0.28) for i in range(6, 11)})
    _mapping, colors = label_from_color(feats)
    assert sorted(colors.values()) == ["blue", "red"]  # 仍须被信号门拒绝


def test_label_from_color_too_few_tracks():
    assert label_from_color({1: (0.3, 0.0)}) == (None, None)


# ── 裁剪 ─────────────────────────────────────────────────

def test_crop_upper_body():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop = crop_upper_body(frame, [100, 500, 200, 900])  # 框高 400 → 上部 45% ≈ 180px
    assert crop.shape[0] == pytest.approx(180, abs=2)
    assert crop.shape[1] == 100
    assert crop_upper_body(frame, [0, 0, 5, 5]) is None  # 太小


# ── 类别即队伍（players_e4 权重路径）──────────────────────

def test_apply_team_from_cls():
    frames = {"0": [{"track_id": 1, "cls": 0}, {"track_id": 2, "cls": 1}],
              "1": [{"track_id": 3, "cls": 2}, {"track_id": 4, "cls": 3}]}
    n = apply_team_from_cls(frames)
    assert n == 4
    assert frames["0"][0]["team_id"] == 0 and frames["0"][1]["team_id"] == 1
    assert frames["1"][0]["team_id"] == 2
    assert frames["1"][1]["team_id"] is None  # cls=3（观众/忽略）不判队


# ── 切片补框队伍继承 ─────────────────────────────────────

def test_inherit_team_for_untracked():
    from frisbee_analyzer.team import inherit_team_for_untracked

    frames = {"0": [
        {"track_id": 1, "bbox": [100, 500, 200, 800], "team_id": 0},
        {"track_id": -1, "bbox": [210, 500, 300, 800], "team_id": None},   # 近邻 team0
        {"track_id": -1, "bbox": [1600, 500, 1700, 800], "team_id": None}, # 无近邻
    ]}
    n = inherit_team_for_untracked(frames)
    assert n == 1
    assert frames["0"][1]["team_id"] == 0
    assert frames["0"][2]["team_id"] is None  # 超距离阈值不猜


def test_inherit_skips_when_no_labeled():
    from frisbee_analyzer.team import inherit_team_for_untracked

    frames = {"0": [{"track_id": -1, "bbox": [10, 10, 50, 90], "team_id": None}]}
    assert inherit_team_for_untracked(frames) == 0


# ── 改判落盘 ─────────────────────────────────────────────

def test_write_team_overrides_atomic(tmp_path):
    doc = {"video": "v.mp4", "team_overrides": {}, "frames": {}}
    p = tmp_path / "tracks.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    out = write_team_overrides(doc, 3, 1, p)
    assert out["team_overrides"] == {"3": 1}
    reloaded = json.loads(p.read_text(encoding="utf-8"))
    assert reloaded["team_overrides"] == {"3": 1}
    assert not p.with_suffix(".json.tmp").exists()  # 原子替换后无残留
