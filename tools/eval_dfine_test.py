"""D-FINE-S 在 COCO test 上的评测（@640，原生 API）。
输出: results/dfine_test_eval.json
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
DFINE_DIR = ROOT / "D-FINE"
CKPT = ROOT / "data/bili_final_test/dfine_out/best_stg1.pth"
TEST_ANN = ROOT / "data/datasets/frisbee_coco/test/_annotations.coco.json"
TEST_DIR = ROOT / "data/datasets/frisbee_coco/test"


def main():
    sys.path.insert(0, str(DFINE_DIR))
    os.chdir(str(DFINE_DIR))
    from src.core import YAMLConfig

    cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_s_frisbee.yml")
    model = cfg.model
    state = torch.load(str(CKPT), map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval().cuda()

    cocoGt = COCO(str(TEST_ANN))
    dets = []
    img_ids = cocoGt.getImgIds()
    t0 = time.time()
    n = 0
    for iid in img_ids:
        info = cocoGt.loadImgs(iid)[0]
        img = cv2.imread(str(TEST_DIR / info["file_name"]))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(TF.resize(TF.to_pil_image(rgb), [640, 640])).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(tensor)
        logits = out["pred_logits"][0]
        boxes = out["pred_boxes"][0]
        scores = logits.sigmoid().squeeze(-1) if logits.dim() == 2 else logits.sigmoid()
        keep = scores > 0.25
        for b, s in zip(boxes[keep], scores[keep]):
            cx, cy, w, h = [float(v) for v in b]
            x1 = (cx - w / 2) * info["width"]
            y1 = (cy - h / 2) * info["height"]
            dets.append({"image_id": iid, "category_id": 0, "score": float(s),
                         "bbox": [x1, y1, w * info["width"], h * info["height"]]})
        if n == 10:
            t0 = time.time()
            n = 0
        n += 1
    fps = round(n / max(time.time() - t0, 1e-6), 1)
    E = COCOeval(cocoGt, None, "bbox")
    E.cocoDt = cocoGt.loadRes(dets)
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    result = {"mAP50": round(float(E.stats[1]), 4), "mAP50-95": round(float(E.stats[0]), 4),
              "params_M": round(params, 2), "fps_640": fps}
    out_path = ROOT / "results/dfine_test_eval.json"
    out_path.write_text(json.dumps(result, indent=1))
    print("RESULT:", json.dumps(result))


if __name__ == "__main__":
    main()
