"""Step 4b: D-FINE-S 与 RF-DETR-small 在金标准 100 帧上推理（原生 API）。
运行(WSL): python3 tools/golden_inference_detr.py
输出: results/golden_inference/{dfine_s,rfdetr_small}.json
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import os as _os
ROOT = Path(_os.environ.get("FRISBEE_ROOT", str(Path(__file__).resolve().parents[1])))
FRAMES = ROOT / "data/golden_set_100/frames"
OUT = ROOT / "results/golden_inference"
OUT.mkdir(parents=True, exist_ok=True)
LOW_CONF = 0.01


def load_frames():
    imgs = {}
    for f in sorted(FRAMES.glob("*.jpg")):
        im = cv2.imread(str(f))
        if im is not None:
            imgs[f.name] = im
    return imgs


def run_dfine(imgs, res=640, ckpt_path=None, model_name="dfine_s", cfg_name="configs/dfine/dfine_hgnetv2_s_frisbee.yml"):
    sys.path.insert(0, str(ROOT / "D-FINE"))
    os.chdir(str(ROOT / "D-FINE"))
    import torchvision.transforms.functional as TF
    from src.core import YAMLConfig

    cfg = YAMLConfig(cfg_name)
    model = cfg.model
    ckpt = torch.load(str(ckpt_path or (ROOT / "data/bili_final_test/dfine_out/best_stg1.pth")), map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval().cuda()

    out = {}
    n = 0
    t0 = None
    for fname, img in imgs.items():
        if n == 3:
            t0 = time.time()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(TF.resize(TF.to_pil_image(rgb), [res, res])).unsqueeze(0).cuda()
        with torch.no_grad():
            r = model(tensor)
        logits = r["pred_logits"][0]
        boxes = r["pred_boxes"][0]
        scores = logits.sigmoid().squeeze(-1) if logits.dim() == 2 else logits.sigmoid()
        h, w = img.shape[:2]
        dets = []
        for b, s in zip(boxes.cpu().numpy(), scores.cpu().numpy()):
            sv = float(s)
            if sv < LOW_CONF:
                continue
            cx, cy, bw, bh = [float(v) for v in b]
            dets.append({"bbox": [round((cx - bw / 2) * w, 1), round((cy - bh / 2) * h, 1),
                                  round((cx + bw / 2) * w, 1), round((cy + bh / 2) * h, 1)],
                         "conf": round(sv, 4)})
        out[fname] = dets
        n += 1
    fps = round((n - 3) / max(time.time() - t0, 1e-6), 1) if n > 3 else None
    return {"model": model_name, "type": "dfine", "resolution": res, "fps": fps, "frames": out}


def run_rfdetr(imgs):
    from rfdetr import RFDETRSmall
    ck = ROOT / "data/bili_final_test/rfdetr_out/checkpoint_best_ema.pth"
    model = RFDETRSmall(pretrain_weights=str(ck))
    out = {}
    n = 0
    t0 = None
    for fname, img in imgs.items():
        if n == 3:
            t0 = time.time()
        d = model.predict(img, threshold=LOW_CONF)
        dets = []
        for xyxy, c in zip(d.xyxy, d.confidence):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            dets.append({"bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                         "conf": round(float(c), 4)})
        out[fname] = dets
        n += 1
    fps = round((n - 3) / max(time.time() - t0, 1e-6), 1) if n > 3 else None
    return {"model": "rfdetr_small", "type": "rfdetr", "resolution": 672, "fps": fps, "frames": out}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dfine-ckpt", type=str, default=None,
                    help="D-FINE 权重路径（默认 dfine_out/best_stg1.pth；赛马用 dfine_v3_out）")
    ap.add_argument("--dfine-name", type=str, default="dfine_s",
                    help="输出 json 的模型名（如 dfine_s_v3）")
    ap.add_argument("--skip-rfdetr", action="store_true")
    ap.add_argument("--dfine-cfg", type=str, default="configs/dfine/dfine_hgnetv2_s_frisbee.yml")
    args = ap.parse_args()

    imgs = load_frames()
    print(f"frames: {len(imgs)}", flush=True)

    p = OUT / f"{args.dfine_name}.json"
    if not p.exists():
        print(f"run {args.dfine_name} ...", flush=True)
        r = run_dfine(imgs, ckpt_path=args.dfine_ckpt, model_name=args.dfine_name, cfg_name=args.dfine_cfg)
        p.write_text(json.dumps(r))
        print(f"  -> fps={r['fps']} dets={sum(len(v) for v in r['frames'].values())}", flush=True)
    else:
        print(f"{args.dfine_name} exists, skip")

    p2 = OUT / "rfdetr_small.json"
    if not args.skip_rfdetr and not p2.exists():
        print("run rfdetr_small ...", flush=True)
        try:
            r2 = run_rfdetr(imgs)
        except ImportError as e:
            print(f"  rfdetr 不可用，跳过: {e}", flush=True)
            return
        p2.write_text(json.dumps(r2))
        print(f"  -> fps={r2['fps']} dets={sum(len(v) for v in r2['frames'].values())}", flush=True)
    elif args.skip_rfdetr:
        print("rfdetr skipped by flag")


if __name__ == "__main__":
    main()
