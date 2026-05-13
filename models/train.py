from ultralytics import YOLO
import argparse
from pathlib import Path
import os


def train_frisbee_detector(data_yaml, model_size="s", epochs=100, imgsz=1280,
                           batch=8, resume_from=None, device=0, workers=4,
                           box=7.5, cls=0.5, close_mosaic=10, patience=20,
                           run_name=None):
    if resume_from and os.path.exists(resume_from):
        print(f"Loading model from: {resume_from}")
        model = YOLO(resume_from)
    else:
        print(f"Loading model: yolov8{model_size}.pt")
        model = YOLO(f"yolov8{model_size}.pt")

    name = run_name if run_name else f"frisbee_det_{model_size}"
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project="runs/detect",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        # Small-object optimization: frisbee bboxes avg ~0.01% of image area
        # Lower mosaic to prevent tiny objects from being destroyed by 4-image blending
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        degrees=15,
        translate=0.1,
        # Wider scale range to preserve small objects during augmentation
        scale=0.9,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=patience,
        close_mosaic=close_mosaic,
        box=box,
        cls=cls,
        save=True,
        save_period=10,
        device=device,
        workers=workers,
        exist_ok=True,
    )
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest model saved to: {best_path}")
    return str(best_path), results


def validate_model(model_path, data_yaml, imgsz=1280):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="frisbee.yaml", help="Dataset YAML path")
    parser.add_argument("--model-size", default="s", choices=["n", "s", "m", "l"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume/fine-tune from")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", default="0", help="CUDA device")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss weight (higher = better localization)")
    parser.add_argument("--cls", type=float, default=0.5, help="Classification loss weight (higher = fewer false positives)")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic augmentation in last N epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--name", default=None, help="Training run name (default: frisbee_det_{model_size})")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    if args.validate_only:
        if not args.model_path:
            print("--model-path required with --validate-only")
            exit(1)
        validate_model(args.model_path, args.data, args.imgsz)
    else:
        best_path, results = train_frisbee_detector(
            data_yaml=args.data,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            resume_from=args.resume,
            workers=args.workers,
            device=int(args.device) if args.device.isdigit() else args.device,
            box=args.box,
            cls=args.cls,
            close_mosaic=args.close_mosaic,
            patience=args.patience,
            run_name=args.name,
        )
        print(f"\nRunning validation on best model...")
        validate_model(best_path, args.data, args.imgsz)
