"""Merge frisbee_pool into train/val/test splits based on a product YAML.

Usage:
    python3 tools/merge_datasets.py --product ../frisbee-data/products/v1.yaml
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from pathlib import Path as _Path

import yaml

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
del _Path

from configs.paths import PROJECT_ROOT  # noqa: E402
from utils.dataset import ensure_split_dirs, split_items, write_yaml_config  # noqa: E402

DST_DIR = PROJECT_ROOT / "data" / "datasets" / "frisbee_merged"
CONFIG_OUT = PROJECT_ROOT / "configs" / "frisbee_merged.yaml"
HASH_FILE = DST_DIR / ".pool_hash"


def load_product(yaml_path: Path) -> dict:
    """Parse a product YAML and return config dict."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Product YAML not found: {yaml_path}")
    with open(yaml_path) as f:
        product = yaml.safe_load(f)
    if not product:
        raise ValueError(f"Empty or invalid YAML: {yaml_path}")
    required = ["pool", "sources", "split"]
    for key in required:
        if key not in product:
            raise KeyError(f"Product YAML missing required key: {key}")
    return product


def collect_from_pool(
    pool: Path,
    sources: list[str],
    exclude: set[str] | None = None,
) -> list[tuple[Path, Path]]:
    """Scan pool and return sorted [(img, lbl), ...] filtered by source prefix."""
    files: list[tuple[Path, Path]] = []
    exclude_set = exclude or set()
    source_set = set(sources) - exclude_set
    for lbl in sorted((pool / "labels").glob("*.txt")):
        prefix = lbl.stem.split("_")[0]
        if prefix not in source_set:
            continue
        img = pool / "images" / f"{lbl.stem}.jpg"
        if img.exists():
            files.append((img, lbl))
    if not files:
        raise ValueError(f"No pool files matched sources: {sources}")
    return files


def _hash_manifest(pool: Path) -> str:
    """SHA256 of pool manifest for change detection."""
    manifest = pool / "manifest.json"
    if not manifest.exists():
        return ""
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _hash_merge_inputs(pool: Path, product_yaml: Path, seed: int) -> str:
    """Hash all inputs that affect the merged dataset contents."""
    h = hashlib.sha256()
    h.update(_hash_manifest(pool).encode())
    h.update(product_yaml.read_bytes())
    h.update(str(seed).encode())
    return h.hexdigest()


def merge_from_product(product_yaml: Path, seed: int = 42) -> None:
    """Merge dataset based on product YAML. Idempotent (skips if pool unchanged)."""
    product = load_product(product_yaml)

    pool = Path(product["pool"])
    if not pool.exists():
        raise FileNotFoundError(f"Pool not found: {pool}")

    # C2: Skip if pool and product config are unchanged.
    manifest_hash = _hash_merge_inputs(pool, product_yaml, seed)
    if HASH_FILE.exists() and HASH_FILE.read_text().strip() == manifest_hash:
        print(f"Pool unchanged (hash: {manifest_hash[:12]}...), merge skipped.")
        print(f"Output ready: {CONFIG_OUT}")
        return

    sources = product["sources"]
    exclude = set(product.get("exclude", []))
    train_only = set(product.get("train_only", [])) - exclude
    ratios = tuple(product["split"].values())  # {train, val, test}

    print(f"Collecting from pool: {pool}")
    if exclude:
        print(f"  Excluding sources: {sorted(exclude)}")
    all_files = collect_from_pool(pool, sources, exclude=exclude)
    print(f"  Total files: {len(all_files)}")

    # Separate train_only and regular files
    train_only_files = []
    regular_files = []
    for img, lbl in all_files:
        prefix = lbl.stem.split("_")[0]
        if prefix in train_only:
            train_only_files.append((img, lbl))
        else:
            regular_files.append((img, lbl))

    print(f"  train_only: {len(train_only_files)} (sources: {list(train_only)})")
    print(f"  regular: {len(regular_files)}")

    # Split regular files
    splits = split_items(regular_files, ratios, seed=seed)

    # Add train_only files to train split only
    splits["train"].extend(train_only_files)

    # C3: Clean old output
    for subdir in ["images", "labels"]:
        dst_sub = DST_DIR / subdir
        if dst_sub.exists():
            shutil.rmtree(dst_sub)

    print(f"\nWriting to: {DST_DIR}")
    total_positive = 0
    total_negative = 0

    for split_name, items in splits.items():
        img_dir, lbl_dir = ensure_split_dirs(DST_DIR, split_name)
        positive = 0
        negative = 0

        for img_file, lbl_file in items:
            # Pool stems already have source prefix
            new_img = img_dir / f"{img_file.stem}{img_file.suffix}"
            new_lbl = lbl_dir / f"{lbl_file.stem}.txt"

            shutil.copy2(str(img_file), str(new_img))
            if lbl_file.exists():
                shutil.copy2(str(lbl_file), str(new_lbl))
                if new_lbl.stat().st_size == 0 or not new_lbl.read_text().strip():
                    negative += 1
                else:
                    positive += 1
            else:
                new_lbl.touch()
                negative += 1

        total_positive += positive
        total_negative += negative
        print(f"  {split_name}: {len(items)} images ({positive} pos, {negative} neg)")

    total = total_positive + total_negative
    print(f"\nTotal: {total} images")
    print(f"  Positive: {total_positive} ({total_positive / total * 100:.1f}%)")
    print(f"  Negative: {total_negative} ({total_negative / total * 100:.1f}%)")

    # Write YAML config
    write_yaml_config(
        output_path=CONFIG_OUT,
        dataset_dir=str(DST_DIR),
    )
    print(f"\nConfig: {CONFIG_OUT}")

    # Save hash for future skip detection
    HASH_FILE.write_text(manifest_hash)
    print(f"Hash saved: {manifest_hash[:12]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge dataset from product YAML")
    parser.add_argument("--product", required=True, help="Product YAML path")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    args = parser.parse_args()
    merge_from_product(Path(args.product), args.seed)
