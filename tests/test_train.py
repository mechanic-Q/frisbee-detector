from types import SimpleNamespace

import models.train as train_module


def test_train_frisbee_detector_can_disable_plots(monkeypatch, tmp_path):
    captured = {}

    class FakeYOLO:
        def __init__(self, model_spec):
            self.model_spec = model_spec

        def train(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(save_dir=tmp_path)

    monkeypatch.setattr(train_module, "YOLO", FakeYOLO)

    best_path, _ = train_module.train_frisbee_detector(
        data_yaml="configs/frisbee_merged.yaml",
        model_spec="yolov8s-p2.yaml",
        plots=False,
    )

    assert captured["plots"] is False
    assert best_path == str(tmp_path / "weights" / "best.pt")
