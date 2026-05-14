"""Classify detection crops as TP/FP using trained binary classifier."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import torch
from tools.classifier_utils import create_model, get_transforms, classify_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pt model file")
    parser.add_argument("--crop-dir", required=True, help="Directory with crop images")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <crop-dir>/classify_results.csv)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", "unknown")
    print(f"Model loaded (val_acc={val_acc})")

    transform = get_transforms(is_train=False)
    crop_dir = Path(args.crop_dir)
    output = Path(args.output) if args.output else crop_dir / "classify_results.csv"

    n = classify_directory(model, str(crop_dir), str(output), transform, device=device)
    print(f"Classified {n} images")
    print(f"Results saved to {output}")

    if output.exists():
        tp = fp = 0
        with open(output) as f:
            for row in csv.DictReader(f):
                if row["label"] == "1":
                    tp += 1
                else:
                    fp += 1
        total = tp + fp
        if total > 0:
            print(f"TP={tp} FP={fp} FP_rate={fp/total*100:.1f}%")


if __name__ == "__main__":
    main()
