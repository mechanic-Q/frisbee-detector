"""Train a YOLOv8 frisbee detection model with small-object optimizations."""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
del _Path

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from configs.models import DEFAULT_MODEL_SIZE, DEFAULT_IMGSZ, DEFAULT_EPOCHS, DEFAULT_BATCH
from configs.paths import PROJECT_ROOT


def train_frisbee_detector(
    data_yaml: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    resume_from: str | None = None,
    device: int | str = 0,
    workers: int = 4,
    box: float = 7.5,
    close_mosaic: int = 10,
    patience: int = 20,
    run_name: str | None = None,
    model_spec: str | None = None,
    cache: bool = False,
) -> tuple[str, object]:
    """Train a YOLOv8 model. Returns (best_model_path, training_results)."""
    if resume_from and os.path.exists(resume_from):
        print(f"Loading model from: {resume_from}")
        model = YOLO(resume_from)
    elif model_spec:
        print(f"Loading model: {model_spec}")
        model = YOLO(model_spec)
    else:
        print(f"Loading base model: yolov8{model_size}.pt")
        model = YOLO(f"yolov8{model_size}.pt")

    name = run_name if run_name else f"frisbee_det_{model_size}"
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project="runs",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        degrees=15,
        translate=0.1,
        scale=0.9,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=patience,
        close_mosaic=close_mosaic,
        box=box,
        cache=cache,
        save=True,
        save_period=10,
        device=device,
        workers=workers,
        exist_ok=True,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest model saved to: {best_path}")
    return str(best_path), results


def validate_model(model_path: str, data_yaml: str, imgsz: int = DEFAULT_IMGSZ) -> object:
    """Validate a trained model and print metrics."""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz)
    print(f"\n{'='*40}")
    print(f"Validation Results: {model_path}")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print(f"{'='*40}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 frisbee detection model")
    parser.add_argument("--data", default="frisbee.yaml", help="Dataset YAML path")
    parser.add_argument("--model", default=None, help="Custom model YAML (e.g., yolov8s-p2.yaml)")
    parser.add_argument("--model-size", default=DEFAULT_MODEL_SIZE, choices=["n", "s", "m", "l"], help="Model size (used if --model not set)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Image size")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume/fine-tune from")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--device", default="0", help="CUDA device")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss weight")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic in last N epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--name", default=None, help="Training run name")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--model-path", default=None, help="Model path for --validate-only")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM")
    parser.add_argument("--product", default=None, help="Product YAML path (auto-merges before training)")
    args = parser.parse_args()

    if args.product:
        product_path = Path(args.product)
        if not product_path.exists():
            print(f"ERROR: Product YAML not found: {product_path}")
            exit(1)
        from tools.merge_datasets import merge_from_product
        merge_from_product(product_path)
        args.data = str(PROJECT_ROOT / "configs" / "frisbee_merged.yaml")

    if args.validate_only:
        if not args.model_path:
            print("ERROR: --model-path required with --validate-only")
            exit(1)
        validate_model(args.model_path, args.data, args.imgsz)
    else:
        best_path, _ = train_frisbee_detector(
            data_yaml=args.data,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            resume_from=args.resume,
            workers=args.workers,
            device=int(args.device) if args.device.isdigit() else args.device,
            box=args.box,
            close_mosaic=args.close_mosaic,
            patience=args.patience,
            run_name=args.name,
            model_spec=args.model,
            cache=args.cache,
        )
        print("\nRunning validation on best model...")
        validate_model(best_path, args.data, args.imgsz)
