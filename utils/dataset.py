"""Shared dataset utilities: YAML config generation, dataset splitting."""

from pathlib import Path
import random
from typing import Optional


def generate_yaml_config(
    dataset_dir: str | Path,
    train_dir: str = "images/train",
    val_dir: str = "images/val",
    test_dir: str = "images/test",
    nc: int = 1,
    names: Optional[list[str]] = None,
) -> str:
    """Generate a YOLO dataset YAML config string."""
    if names is None:
        names = ["frisbee"]
    names_str = str(names).replace("'", "")  # ['frisbee'] -> [frisbee]
    return (
        f"path: {dataset_dir}\n"
        f"train: {train_dir}\n"
        f"val: {val_dir}\n"
        f"test: {test_dir}\n"
        f"nc: {nc}\n"
        f"names: {names_str}\n"
    )


def write_yaml_config(
    output_path: str | Path,
    dataset_dir: str | Path,
    train_dir: str = "images/train",
    val_dir: str = "images/val",
    test_dir: str = "images/test",
    nc: int = 1,
    names: Optional[list[str]] = None,
) -> Path:
    """Write a YOLO dataset YAML config file."""
    content = generate_yaml_config(dataset_dir, train_dir, val_dir, test_dir, nc, names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    return output_path


def split_items(items: list, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> dict[str, list]:
    """Shuffle and split items into train/val/test by ratios."""
    random.seed(seed)
    shuffled = list(items)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def ensure_split_dirs(base_dir: Path, split_name: str) -> tuple[Path, Path]:
    """Create and return (images/split, labels/split) directories."""
    img_dir = base_dir / "images" / split_name
    lbl_dir = base_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir
