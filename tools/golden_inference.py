"""Step 4: 全模型在金标准 100 帧上推理（低阈值保存全部检测，离线扫阈值）。
运行: python tools/golden_inference.py [--models m1,m2]
输出: results/golden_inference/<model_name>.json
格式: {"model": name, "resolution": R, "frames": {"f000123.jpg": [{"bbox":[x1,y1,x2,y2],"conf":c}, ...]}, "fps": ...}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import os as _os
ROOT = Path(_os.environ.get("FRISBEE_ROOT", str(Path(__file__).resolve().parents[1])))
FRAMES = ROOT / "data/golden_set_100/frames"
OUT = ROOT / "results/golden_inference"
OUT.mkdir(parents=True, exist_ok=True)
LOW_CONF = 0.01   # 极低阈值：保存全部检测，离线扫阈值

# 候选矩阵：name -> (type, weights, resolution)
YOLO_MODELS = {
    "yolo_v8s":     ("ultra", ROOT / "runs/detect/bench_v8s/weights/best.pt", 1280),
    "yolo_11s":     ("ultra", ROOT / "runs/detect/bench_11s/weights/best.pt", 1280),
    "yolo_26s":     ("ultra", ROOT / "runs/detect/bench_26s/weights/best.pt", 1280),
    "prod_v8s_p2":  ("ultra", ROOT / "runs/detect/bili_prod_v8sp2/weights/best.pt", 1280),
    "shadow_v1":    ("ultra", ROOT / "runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt", 1280),
    "ctrl_v8s_p2":  ("ultra", ROOT / "runs/detect/bili_ctrl_v8sp2/weights/best.pt", 1280),
    "prod_v8s_p2_640": ("ultra", ROOT / "runs/detect/bili_prod_v8sp2/weights/best.pt", 640),
    "race_v8sp2_v3": ("ultra", ROOT / "runs/detect/race_v8sp2_v3/weights/best.pt", 1280),
    "race_v8sp2_v35": ("ultra", ROOT / "runs/detect/race_v8sp2_v35/weights/best.pt", 1280),
}


def load_frames():
    files = sorted(FRAMES.glob("*.jpg"))
    imgs = {}
    for f in files:
        im = cv2.imread(str(f))
        if im is not None:
            imgs[f.name] = im
    return imgs


def run_ultra(name, weights, imgsz, imgs):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    res = {}
    n = 0
    t0 = None
    for fname, img in imgs.items():
        if n == 3:
            t0 = time.time()
        r = m.predict(img, conf=LOW_CONF, imgsz=imgsz, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().tolist()):
                dets.append({"bbox": [round(float(v), 1) for v in b], "conf": round(float(c), 4)})
        res[fname] = dets
        n += 1
    fps = round((n - 3) / max(time.time() - t0, 1e-6), 1) if n > 3 else None
    return {"model": name, "type": "ultralytics", "resolution": imgsz,
            "weights": str(weights), "fps": fps, "frames": res}


def run_sahi(name, weights, imgsz, imgs, slice_wh=640, overlap=0.2):
    """SAHI 切片推理变体。"""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    m = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                           confidence_threshold=LOW_CONF, device="cuda:0")
    res = {}
    n = 0
    t0 = None
    for fname, img in imgs.items():
        if n == 3:
            t0 = time.time()
        r = get_sliced_prediction(img, m, slice_height=slice_wh, slice_width=slice_wh,
                                  overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                  postprocess_type="NMS", postprocess_match_metric="IOS",
                                  postprocess_match_threshold=0.5, verbose=0)
        dets = [{"bbox": [round(float(v), 1) for v in o.bbox.to_xyxy()],
                 "conf": round(float(o.score.value), 4)} for o in r.object_prediction_list]
        res[fname] = dets
        n += 1
    fps = round((n - 3) / max(time.time() - t0, 1e-6), 1) if n > 3 else None
    return {"model": name, "type": "sahi", "resolution": slice_wh,
            "weights": str(weights), "fps": fps, "frames": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="逗号分隔的模型名（默认全部 ultralytics 系）")
    ap.add_argument("--sahi", action="store_true", help="额外跑 SAHI 切片变体")
    args = ap.parse_args()

    imgs = load_frames()
    print(f"frames: {len(imgs)}")
    todo = YOLO_MODELS
    if args.models:
        names = [s.strip() for s in args.models.split(",") if s.strip()]
        todo = {k: v for k, v in YOLO_MODELS.items() if k in names}

    for name, (mtype, w, res) in todo.items():
        if not Path(w).exists():
            print(f"SKIP {name}: weights missing {w}")
            continue
        outp = OUT / f"{name}.json"
        if outp.exists():
            print(f"SKIP {name}: result exists")
            continue
        print(f"run {name} @{res} ...", flush=True)
        r = run_ultra(name, w, res, imgs)
        outp.write_text(json.dumps(r))
        print(f"  -> {len(r['frames'])} frames, fps={r['fps']}, dets={sum(len(v) for v in r['frames'].values())}")

    if args.sahi:
        w = ROOT / "runs/detect/bili_prod_v8sp2/weights/best.pt"
        name = "prod_v8s_p2_sahi640"
        outp = OUT / f"{name}.json"
        if not outp.exists() and w.exists():
            print(f"run {name} ...", flush=True)
            r = run_sahi(name, w, 1280, imgs)
            outp.write_text(json.dumps(r))
            print(f"  -> fps={r['fps']}, dets={sum(len(v) for v in r['frames'].values())}")


if __name__ == "__main__":
    main()
