"""球员检测方案对比（零样本）：DFL足球权重 / yolov8x / yolo11x / yolo26x，全帧 vs SAHI(2x上采样切片)。
在 WSL 运行: python3 data/bili_final_test/compare_player_det.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

ROOT = Path("/mnt/e/frisbee-detector")
OUT_DIR = ROOT / "data/bili_final_test/out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DFL = str(ROOT / "models/pretrained/football-player-detection.pt")

PROBES = ["probe_60s", "probe_1200s"]
ROUGH_TRUTH = {"probe_60s": 25, "probe_1200s": 14}  # 粗略人工计数（场上球员+裁判）

VARIANTS = {
    "A_dfl_full": ("full", DFL, 0.10),
    "B_dfl_sahi": ("sahi", DFL, 0.25),
    "C_v8x_full": ("full", str(ROOT / "yolov8x.pt"), 0.25),
    "D_v8x_sahi": ("sahi", str(ROOT / "yolov8x.pt"), 0.25),
    "E_11x_full": ("full", str(ROOT / "yolo11x.pt"), 0.25),
    "F_11x_sahi": ("sahi", str(ROOT / "yolo11x.pt"), 0.25),
    "G_26x_full": ("full", str(ROOT / "yolo26x.pt"), 0.25),
    "H_26x_sahi": ("sahi", str(ROOT / "yolo26x.pt"), 0.25),
}


def predict_full(model, frame, conf, football=False):
    classes = [1, 2, 3] if football else [0]
    t0 = time.time()
    r = model.predict(frame, conf=conf, imgsz=1280, classes=classes, verbose=False)[0]
    dt = (time.time() - t0) * 1000
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), dt


def predict_sahi(model_path, frame, conf, football=False):
    # COCO person=0；DFL 中 player=2, goalkeeper=1, referee=3 → sahi 无 classes 过滤，后处理按类过滤
    keep = {1, 2, 3} if football else {0}
    m = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=model_path,
                                           confidence_threshold=conf, device="cuda:0")
    big = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    t0 = time.time()
    r = get_sliced_prediction(big, m, slice_height=512, slice_width=512,
                              overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                              postprocess_type="NMS", postprocess_match_metric="IOS",
                              postprocess_match_threshold=0.5, verbose=0)
    dt = (time.time() - t0) * 1000
    boxes, confs = [], []
    for o in r.object_prediction_list:
        if o.category.id not in keep:
            continue
        x1, y1, x2, y2 = o.bbox.to_xyxy()
        boxes.append([x1 / 2, y1 / 2, x2 / 2, y2 / 2])
        confs.append(o.score.value)
    return np.array(boxes).reshape(-1, 4), np.array(confs), dt


def draw_save(frame, boxes, confs, name):
    img = frame.copy()
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 160, 0), 2)
        cv2.putText(img, f"{c:.2f}", (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 160, 0), 1)
    cv2.putText(img, f"{name}: {len(boxes)} dets", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imwrite(str(OUT_DIR / f"cmp_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    frames = {n: cv2.imread(str(ROOT / f"data/bili_final_test/frames/{n}.jpg")) for n in PROBES}
    models = {}
    print("loading models...")
    for p in {v[1] for v in VARIANTS.values()}:
        models[p] = YOLO(p)
    print("done")

    summary = {}
    for name, frame in frames.items():
        print(f"\n=== {name} (rough truth ~{ROUGH_TRUTH[name]}) ===")
        for vname, (mode, mpath, conf) in VARIANTS.items():
            football = "dfl" in vname
            if mode == "full":
                b, c, dt = predict_full(models[mpath], frame, conf, football)
            else:
                b, c, dt = predict_sahi(mpath, frame, conf, football)
            draw_save(frame, b, c, f"{vname}_{name}")
            summary[f"{vname}_{name}"] = {"dets": len(b), "ms": round(dt)}
            print(f"{vname:14s}: {len(b):3d} dets  {dt:6.0f} ms")

    import json
    (OUT_DIR / "compare_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nsaved -> out/compare_summary.json + cmp_*.jpg")


if __name__ == "__main__":
    main()
