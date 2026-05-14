"""SigLIP zero-shot frisbee classifier for YOLO detection crops."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import torch
from PIL import Image


def load_model_and_processor(device="cuda"):
    """Load SigLIP model and processor. First call downloads ~3.5GB."""
    from transformers import AutoModel, AutoProcessor
    model_name = "google/siglip-so400m-patch14-384"
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def classify_image(image_path, model, processor, device="cuda", threshold=0.5):
    """Classify a single crop. Returns (is_frisbee, probability)."""
    prompt = "a photo of a frisbee or flying disc"
    image = Image.open(str(image_path)).convert("RGB")

    inputs = processor(text=[prompt], images=image,
                       padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        prob = torch.sigmoid(logits)[0][0].item()

    return prob >= threshold, prob


def classify_directory(model, processor, img_dir, output_csv, device="cuda", threshold=0.5):
    """Classify all images in a directory, write results to CSV."""
    img_dir = Path(img_dir)
    results = []

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )

    for img_path in image_files:
        is_frisbee, prob = classify_image(str(img_path), model, processor, device=device, threshold=threshold)
        label = "frisbee" if is_frisbee else "not_frisbee"
        results.append((img_path.name, label, prob))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence"])
        for filename, label, prob in results:
            writer.writerow([filename, label, f"{prob:.4f}"])

    return len(results)


def main():
    parser = argparse.ArgumentParser(description="SigLIP zero-shot frisbee classifier")
    parser.add_argument("--crop-dir", required=True, help="Directory with crop images")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading SigLIP model (first time downloads ~3.5GB)...")
    model, processor = load_model_and_processor(device=device)

    crop_dir = Path(args.crop_dir)
    output = Path(args.output) if args.output else crop_dir / "siglip_results.csv"

    n = classify_directory(model, processor, str(crop_dir), str(output), device=device, threshold=args.threshold)
    print(f"Classified {n} images")
    print(f"Results saved to {output}")

    if output.exists():
        frisbee = 0
        not_frisbee = 0
        with open(output) as f:
            for row in csv.DictReader(f):
                if row["label"] == "frisbee":
                    frisbee += 1
                else:
                    not_frisbee += 1
        total = frisbee + not_frisbee
        if total > 0:
            print(f"frisbee={frisbee} not_frisbee={not_frisbee} "
                  f"detection_rate={frisbee/total*100:.1f}%")


if __name__ == "__main__":
    main()
