#!/usr/bin/env python3
"""F1 格④⑤ GPU 补跑：D-FINE(COCO)/race_v8sp2_v3 + 融合 的 pipeline 级对照。

D-FINE 不走 ultralytics pipeline——用 golden_inference_detr 同款原生推理产
逐帧检出序列，再离线过 disc_fusion（与 pipeline 内联同一模块），对齐 c2/c3 的分析口径。
前置：连续帧素材（55-56min 片段 300 帧）。

用法: python tools/f1_cells45.py [--max-frames 300]
产物: results/f1_matrix/cells45.json
"""
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "D-FINE"))

from frisbee_analyzer.disc_fusion import fuse_disc_detections  # noqa: E402


def detect_dfine(model, imgs, res=640, low_conf=0.05):
    import torchvision.transforms.functional as TF
    out = []
    for img in imgs:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = TF.to_tensor(TF.resize(TF.to_pil_image(rgb), [res, res])).unsqueeze(0).cuda()
        with torch.no_grad():
            r = model(t)
        s = r["pred_logits"].sigmoid().squeeze(-1)
        boxes = r["pred_boxes"][0]
        h, w = img.shape[:2]
        dets = []
        for b, sc in zip(boxes.cpu().numpy(), s[0].cpu().numpy()):
            if float(sc) < low_conf:
                continue
            cx, cy, bw, bh = [float(v) for v in b]
            dets.append({"bbox": [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h],
                         "conf": round(float(sc), 4)})
        out.append(dets)
    return out


def detect_ultra(weights, imgs, conf=0.05, imgsz=1280):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    out = []
    for img in imgs:
        r = m.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                dets.append({"bbox": [float(v) for v in b], "conf": round(float(c), 4)})
        out.append(dets)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--video", default=str(ROOT / "movie/25866279684-1-192_55-56min.mp4"))
    ap.add_argument("--calib", default=str(ROOT / "results/f1_matrix/c0_raw_for_calib/auto_calib.json"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    imgs = []
    for _ in range(args.max_frames):
        ok, f = cap.read()
        if not ok:
            break
        imgs.append(f)
    cap.release()
    print(f"frames: {len(imgs)}", flush=True)

    results = {}
    # 格④: D-FINE (COCO 底座) + 融合
    os_d = str(ROOT / "D-FINE")
    sys.path.insert(0, os_d)
    cwd = Path.cwd()
    import os
    os.chdir(os_d)
    import torch
    from src.core import YAMLConfig
    cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_s_frisbee.yml")
    model = cfg.model
    ck = torch.load(str(ROOT / "data/bili_final_test/dfine_out/best_stg1.pth"), map_location="cpu")
    model.load_state_dict(ck.get("model", ck))
    model.eval().cuda()
    os.chdir(cwd)
    t0 = time.time()
    dets = detect_dfine(model, imgs)
    fused, st = fuse_disc_detections(dets, fps=25.0)
    results["c4_dfine_fusion"] = {
        "raw": sum(len(d) for d in dets), "stats": st.__dict__,
        "longest": st.longest_track_frames, "runtime_s": round(time.time()-t0, 1),
    }
    print(f"c4_dfine_fusion: raw={results['c4_dfine_fusion']['raw']} longest={st.longest_track_frames}f", flush=True)
    del model
    torch.cuda.empty_cache()

    # 格⑤: race_v8sp2_v3 + 融合
    t0 = time.time()
    dets5 = detect_ultra(ROOT / "runs/detect/race_v8sp2_v3/weights/best.pt", imgs)
    fused5, st5 = fuse_disc_detections(dets5, fps=25.0)
    results["c5_racev8_fusion"] = {
        "raw": sum(len(d) for d in dets5), "stats": st5.__dict__,
        "longest": st5.longest_track_frames, "runtime_s": round(time.time()-t0, 1),
    }
    print(f"c5_racev8_fusion: raw={results['c5_racev8_fusion']['raw']} longest={st5.longest_track_frames}f", flush=True)

    out = ROOT / "results/f1_matrix/cells45.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
