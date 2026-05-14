"""Shared utilities for frisbee classifier."""
import csv
import warnings
from pathlib import Path


def load_labeled_dataset(csv_path, img_dir):
    """Load labeled dataset from CSV and image directory.

    Returns:
        list of (Path, int): (image_path, label) where label is 0=FP, 1=TP.
    """
    samples = []
    img_dir = Path(img_dir)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            label = 1 if row["result"].strip().upper() == "TP" else 0
            img_path = img_dir / row["filename"]
            if img_path.is_file():
                samples.append((img_path, label))
            else:
                warnings.warn(f"Missing image: {img_path}")
    return samples


def split_train_val(samples, val_ratio=0.2, seed=42):
    """Stratified split into train and validation sets."""
    import random
    rng = random.Random(seed)

    tp_samples = [(p, l) for p, l in samples if l == 1]
    fp_samples = [(p, l) for p, l in samples if l == 0]

    rng.shuffle(tp_samples)
    rng.shuffle(fp_samples)

    tp_split = int(len(tp_samples) * (1 - val_ratio))
    fp_split = int(len(fp_samples) * (1 - val_ratio))

    train = tp_samples[:tp_split] + fp_samples[:fp_split]
    val = tp_samples[tp_split:] + fp_samples[fp_split:]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
