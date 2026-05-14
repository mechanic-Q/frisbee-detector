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


def create_model(num_classes=2, pretrained=True):
    """Create a ResNet18 binary classifier."""
    import torch.nn as nn
    import torchvision.models as models

    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_transforms(is_train=True):
    """Get image transforms for training or inference."""
    import torchvision.transforms as T

    if is_train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def classify_image(model, image_path, transform, device="cuda"):
    """Classify a single image. Returns (label, confidence)."""
    import torch
    import cv2

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = probs.max(dim=1)

    return pred.item(), conf.item()


def classify_directory(model, img_dir, output_csv, transform, device="cuda"):
    """Classify all images in a directory, write results to CSV."""
    import csv
    from pathlib import Path

    img_dir = Path(img_dir)
    results = []

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )

    for img_path in image_files:
        label, conf = classify_image(model, str(img_path), transform, device=device)
        results.append((img_path.name, label, conf))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence"])
        for filename, label, conf in results:
            writer.writerow([filename, label, f"{conf:.4f}"])

    return len(results)
