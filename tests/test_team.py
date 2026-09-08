"""team.py 纯函数单测：不依赖 GPU/SigLIP/视频，embedding 可注入。"""

import json

import numpy as np
import pytest

from frisbee_analyzer.team import (
    MIN_CROPS_FOR_CLUSTER,
    cluster_team_embeddings,
    crop_upper_body,
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


# ── 裁剪 ─────────────────────────────────────────────────

def test_crop_upper_body():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop = crop_upper_body(frame, [100, 500, 200, 900])  # 框高 400 → 上部 45% ≈ 180px
    assert crop.shape[0] == pytest.approx(180, abs=2)
    assert crop.shape[1] == 100
    assert crop_upper_body(frame, [0, 0, 5, 5]) is None  # 太小


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
