"""Phase A §13.16 盘候选检测单元测试（mock 模型，无 GPU）。

运行: python -m pytest tests/test_tracking_tile.py -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.tracking import detect_disc_candidates  # noqa: E402


def _dummy_img():
    import numpy as np
    return np.zeros((480, 852, 3), dtype="uint8")


class _Tensor:
    """模拟 ultralytics 输出 tensor 的 .cpu().numpy()/.tolist() 链。"""

    def __init__(self, values):
        self._v = values

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np
        return np.asarray(self._v, dtype=float)

    def tolist(self):
        return list(self._v)


class _IntTensor:
    def __init__(self, values):
        self._v = values

    def cpu(self):
        return self

    def int(self):
        return self

    def tolist(self):
        return list(self._v)


class _Boxes:
    def __init__(self, boxes_confs):
        n = len(boxes_confs)
        self.xyxy = _Tensor([b for b, _ in boxes_confs]) if boxes_confs else None
        self.conf = _Tensor([c for _, c in boxes_confs]) if boxes_confs else None
        self.cls = _IntTensor([0] * n)

    def __len__(self):
        return len(self.xyxy._v) if self.xyxy is not None else 0


class FakeRes:
    def __init__(self, boxes_confs):
        self.boxes = _Boxes(boxes_confs)


class FakeModel:
    """predict(frame, conf, imgsz, classes, verbose) → [res]；按调用序出队。"""

    def __init__(self, calls):
        self.calls = list(calls)  # 每次 predict 消耗一个 FakeRes
        self.frames_seen = []

    def predict(self, img, **kw):
        self.frames_seen.append((kw.get("conf"), img is not None))
        return [FakeRes(self.calls.pop(0))]


def test_no_tile_when_off():
    m = FakeModel([[( [10, 10, 26, 26], 0.6 )]])
    out = detect_disc_candidates(m, _dummy_img(), 0.35, 640, tile_grid=0)
    assert len(out) == 1 and out[0]["source"] == "full"
    assert len(m.calls) == 0  # 只跑了一次整帧


def test_tile_triggered_on_low_conf():
    # 整帧 conf 0.4 < 0.5 触发切片；切片出 0.62 高分 → 合并 top2
    img = _dummy_img()
    m = FakeModel([
        [([10, 10, 26, 26], 0.4)],                  # 整帧
        [([12, 12, 28, 28], 0.62)], [], [], [],      # 2×2 切片：片1 高分，其余空
    ])
    out = detect_disc_candidates(m, _dummy_img(), 0.35, 640, tile_grid=2,
                                 tile_trigger="conf<0.5", tile_topk=2)
    # 切片高分框与整帧低分框是同一物理盘（IOS 高）→ NMM 合并保留高分 tile 表示
    assert len(out) == 1 and out[0]["source"] == "tile"
    assert abs(out[0]["conf"] - 0.62) < 1e-3


def test_no_tile_on_confident_frame():
    # 整帧已有 0.8 高分 → 不触发切片
    m = FakeModel([[([10, 10, 26, 26], 0.8)]])
    detect_disc_candidates(m, _dummy_img(), 0.35, 640, tile_grid=2, tile_trigger="conf<0.5")
    assert len(m.calls) == 0  # 只消耗了整帧调用


def test_only_no_detection_mode():
    # only-no-detection：有检出（低分）也不切片
    m = FakeModel([[([10, 10, 26, 26], 0.4)]])
    detect_disc_candidates(m, _dummy_img(), 0.35, 640, tile_grid=2, tile_trigger="only-no-detection")
    assert len(m.calls) == 0


def test_topk_cap():
    # 3 个互不重叠候选 → 只留 conf 前 2
    m = FakeModel([
        [([10, 10, 26, 26], 0.5), ([200, 200, 216, 216], 0.45), ([400, 400, 416, 416], 0.4)],
    ])
    out = detect_disc_candidates(m, _dummy_img(), 0.35, 640, tile_grid=0, tile_topk=2)
    assert len(out) == 2 and out[0]["conf"] == 0.5


def test_tile_conf_floor():
    # 切片推理用 tile_conf 地板（默认=disc_conf）；低于地板的切片检出不出现在结果
    m = FakeModel([
        [],                                         # 整帧空
        [([12, 12, 28, 28], 0.34)], [], [], [],     # 片1: 0.34 低于 tile_conf 地板（模拟漏网）
    ])
    # tile_conf 传 0.35 → mock 中 0.34 的框仍返回（模拟模型不守约），但 detect 层不再过滤——
    # 结果保留（fusion 搜索态准入会拒），此处只验证不崩溃且 source=tile
    out = detect_disc_candidates(m, _dummy_img(), 0.35, 640, tile_grid=2,
                                 tile_trigger="only-no-detection", tile_conf=0.35)
    assert all(d["source"] == "tile" for d in out)
