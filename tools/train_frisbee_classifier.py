"""Train a frisbee binary classifier on manually-labeled detection crops."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from tools.classifier_utils import (
    load_labeled_dataset, split_train_val, create_model, get_transforms
)


def train_one_epoch(model, samples, transform, optimizer, criterion, device, batch_size=16):
    """Train for one epoch. Uses oversampling: ensures batch has both classes."""
    model.train()
    tp_samples = [(p, l) for p, l in samples if l == 1]
    fp_samples = [(p, l) for p, l in samples if l == 0]

    total_loss = 0.0
    correct = 0
    total = 0

    num_batches = max(len(tp_samples), len(fp_samples)) // (batch_size // 2)
    if num_batches == 0:
        num_batches = 1

    for _ in range(num_batches):
        batch_tp = random.choices(tp_samples, k=batch_size // 2)
        batch_fp = random.choices(fp_samples, k=batch_size // 2)
        batch = batch_tp + batch_fp
        random.shuffle(batch)

        images = []
        labels = []
        for img_path, label in batch:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img)
            images.append(tensor)
            labels.append(label)

        x = torch.stack(images).to(device)
        y = torch.tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = output.max(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / num_batches, correct / total


def validate(model, samples, transform, criterion, device, batch_size=16):
    """Validate on a set of samples."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        images = []
        labels = []
        for img_path, label in batch:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img)
            images.append(tensor)
            labels.append(label)

        if not images:
            continue

        x = torch.stack(images).to(device)
        y = torch.tensor(labels, dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(x)
            loss = criterion(output, y)
            _, preds = output.max(dim=1)

        total_loss += loss.item()
        correct += (preds == y).sum().item()
        total += y.size(0)
        num_batches += 1

    return total_loss / max(num_batches, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/perbox_crops/review_results.csv")
    parser.add_argument("--img-dir", default="data/perbox_crops")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--name", default="frisbee_classifier_v1")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    samples = load_labeled_dataset(csv_path, img_dir)
    print(f"Loaded {len(samples)} samples ({sum(1 for _,l in samples if l==1)} TP, {sum(1 for _,l in samples if l==0)} FP)")

    train_samples, val_samples = split_train_val(samples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Train: {len(train_samples)} ({sum(1 for _,l in train_samples if l==1)} TP, {sum(1 for _,l in train_samples if l==0)} FP)")
    print(f"Val:   {len(val_samples)} ({sum(1 for _,l in val_samples if l==1)} TP, {sum(1 for _,l in val_samples if l==0)} FP)")

    model = create_model(num_classes=2, pretrained=True)
    model.to(device)

    tp_count = sum(1 for _, l in train_samples if l == 1)
    fp_count = sum(1 for _, l in train_samples if l == 0)
    tp_weight = fp_count / max(tp_count, 1)
    class_weights = torch.tensor([1.0, tp_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam([
        {"params": model.fc.parameters(), "lr": args.lr * 10},
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": args.lr},
    ])

    transform_train = get_transforms(is_train=True)
    transform_val = get_transforms(is_train=False)

    best_val_acc = 0.0
    save_path = Path("runs/classify") / args.name
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_samples, transform_train, optimizer, criterion, device, args.batch_size
        )
        val_loss, val_acc = validate(
            model, val_samples, transform_val, criterion, device, args.batch_size
        )

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, save_path / f"{args.name}.pt")
            print(f"  => saved (val_acc={val_acc:.4f})")

    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Model saved to {save_path / args.name}.pt")


if __name__ == "__main__":
    main()
