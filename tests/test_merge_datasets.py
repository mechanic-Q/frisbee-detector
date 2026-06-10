"""Tests for product-driven dataset merge behavior."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import merge_datasets


def _write_pool_item(pool: Path, stem: str, label: str = "") -> None:
    (pool / "images").mkdir(parents=True, exist_ok=True)
    (pool / "labels").mkdir(parents=True, exist_ok=True)
    (pool / "images" / f"{stem}.jpg").write_bytes(b"fake image")
    (pool / "labels" / f"{stem}.txt").write_text(label)


def _write_product(product_path: Path, pool: Path, exclude: list[str]) -> None:
    exclude_block = "\n" + "\n".join(f"  - {source}" for source in exclude) if exclude else " []"
    product_path.write_text(
        f"""pool: {pool}
sources:
  - keep
  - drop
exclude:{exclude_block}
split:
  train: 1.0
  val: 0.0
  test: 0.0
"""
    )


def test_collect_from_pool_honors_excluded_sources(tmp_path):
    pool = tmp_path / "pool"
    _write_pool_item(pool, "keep_001", "0 0.5 0.5 0.1 0.1\n")
    _write_pool_item(pool, "drop_001")

    files = merge_datasets.collect_from_pool(pool, ["keep", "drop"], exclude={"drop"})

    assert [img.stem for img, _ in files] == ["keep_001"]


def test_merge_reruns_when_product_exclude_changes(tmp_path, monkeypatch):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "manifest.json").write_text("{}")
    _write_pool_item(pool, "keep_001", "0 0.5 0.5 0.1 0.1\n")
    _write_pool_item(pool, "drop_001")

    out_dir = tmp_path / "merged"
    monkeypatch.setattr(merge_datasets, "DST_DIR", out_dir)
    monkeypatch.setattr(merge_datasets, "CONFIG_OUT", tmp_path / "frisbee_merged.yaml")
    monkeypatch.setattr(merge_datasets, "HASH_FILE", out_dir / ".pool_hash")

    product = tmp_path / "product.yaml"
    _write_product(product, pool, exclude=[])
    merge_datasets.merge_from_product(product, seed=42)
    assert (out_dir / "images" / "train" / "drop_001.jpg").exists()

    _write_product(product, pool, exclude=["drop"])
    merge_datasets.merge_from_product(product, seed=42)

    train_images = sorted(p.stem for p in (out_dir / "images" / "train").glob("*.jpg"))
    assert train_images == ["keep_001"]
