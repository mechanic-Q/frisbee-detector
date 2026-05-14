"""Tests for SigLIP zero-shot frisbee classifier."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pytest
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_model_forward_pass_output_shape():
    """SigLIP 模型前向传播输出形状为 (1, 1)."""
    from tools.classify_frisbee import load_model_and_processor
    from PIL import Image
    import torch

    model, processor = load_model_and_processor(device="cpu")
    dummy_img = Image.new("RGB", (384, 384), color=(128, 128, 128))
    inputs = processor(text=["a photo of a frisbee"], images=dummy_img,
                       padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits_per_image.shape == (1, 1)


def test_classify_image_returns_tuple():
    """单图分类返回 (is_frisbee: bool, probability: float)."""
    from tools.classify_frisbee import load_model_and_processor, classify_image
    from PIL import Image
    import tempfile

    model, processor = load_model_and_processor(device="cpu")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        dummy = Image.new("RGB", (100, 100), color=(0, 0, 0))
        dummy.save(f.name, format="JPEG")
        is_frisbee, prob = classify_image(f.name, model, processor, device="cpu")
    os.unlink(f.name)

    assert isinstance(is_frisbee, bool)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_classify_directory_writes_csv(tmp_path):
    """批量分类目录输出 CSV，包含所有文件."""
    from tools.classify_frisbee import load_model_and_processor, classify_directory
    from PIL import Image

    model, processor = load_model_and_processor(device="cpu")

    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(str(tmp_path / f"crop_{i}_c0.50_vid.jpg"), format="JPEG")

    output_csv = str(tmp_path / "results.csv")
    classify_directory(model, processor, str(tmp_path), output_csv, device="cpu")

    with open(output_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3
        for row in rows:
            assert "filename" in row
            assert "label" in row
            assert "confidence" in row
            assert row["label"] in ("frisbee", "not_frisbee")
            conf = float(row["confidence"])
            assert 0.0 <= conf <= 1.0
