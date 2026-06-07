"""Merge annotation export samples into frisbee_merged training split.

Usage:
    python3 tools/merge_annotation_exports.py --export-dir data/annotation/p2_shadow_fp_round1/export
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
del _Path

from configs.paths import PROJECT_ROOT

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "frisbee_merged"


def _copy_pairs(
    image_dir: Path,
    label_dir: Path,
    dst_images: Path,
    dst_labels: Path,
) -> int:
    copied = 0
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for export image: {image_path}")
        shutil.copy2(image_path, dst_images / image_path.name)
        shutil.copy2(label_path, dst_labels / label_path.name)
        copied += 1
    return copied


def merge_export(export_dir: str | Path, dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> dict[str, int]:
    export_path = Path(export_dir)
    dataset_path = Path(dataset_dir)
    if not export_path.exists():
        raise FileNotFoundError(f"Export directory not found: {export_path}")

    dst_images = dataset_path / "images" / "train"
    dst_labels = dataset_path / "labels" / "train"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    positive_copied = _copy_pairs(
        export_path / "images" / "positive",
        export_path / "labels" / "positive",
        dst_images,
        dst_labels,
    )
    hard_negative_copied = _copy_pairs(
        export_path / "images" / "hard_negative",
        export_path / "labels" / "hard_negative",
        dst_images,
        dst_labels,
    )

    return {
        "positive_copied": positive_copied,
        "hard_negative_copied": hard_negative_copied,
        "total_copied": positive_copied + hard_negative_copied,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge annotation exports into YOLO train split")
    parser.add_argument("--export-dir", required=True, help="Annotation export directory")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="YOLO dataset root")
    args = parser.parse_args()

    report = merge_export(args.export_dir, args.dataset_dir)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
