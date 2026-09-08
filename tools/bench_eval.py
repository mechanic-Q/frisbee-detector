"""检测器基准统一评测：同标尺 test mAP + 1080P 推理 FPS + 参数量。
在 WSL 运行: python3 tools/bench_eval.py
输出: results/bench_summary.json
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/mnt/e/frisbee-detector")
COMFY = Path("/home/lmr/comfy/ComfyUI/runs/detect/runs")
CLIP = ROOT / "movie/2024城市飞盘俱乐部锦标赛『决赛』【北京大哥 VS 上海沪蛙】高清版3-1.mp4"
OUT = ROOT / "results/bench_summary.json"
DUR = 2832.7

ULTRALYTICS_BENCH = {
    "yolov8s": COMFY / "bench_v8s/weights/best.pt",
    "yolo11s": COMFY / "bench_11s/weights/best.pt",
    "yolo26s": COMFY / "bench_26s/weights/best.pt",
}
RFDETR_DIR = ROOT / "data/bili_final_test/rfdetr_out"
N_FPS_FRAMES = 200


def fps_probe(predict_fn):
    """1080P 片段均匀取 200 帧，暖机 10 帧后计时"""
    cap = cv2.VideoCapture(str(CLIP))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, N_FPS_FRAMES).astype(int)
    t0 = time.time()
    n = 0
    for k, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        predict_fn(frame)
        n += 1
        if k == 10:  # 暖机后重置计时
            t0 = time.time()
            n = 0
    cap.release()
    return round(n / (time.time() - t0), 1)


def eval_ultralytics(name, weights):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data="configs/frisbee_merged.yaml", split="test", imgsz=1280, batch=2, verbose=False)
    fps = fps_probe(lambda f: m.predict(f, imgsz=1280, conf=0.35, verbose=False))
    return {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
            "P": round(float(r.box.mp), 4), "R": round(float(r.box.mr), 4),
            "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
            "fps_1080p": fps, "weights": str(weights)}


def eval_rfdetr():
    from rfdetr import RFDETRSmall
    import glob
    ck = None
    for pat in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        cands = sorted(glob.glob(str(RFDETR_DIR / "**" / pat), recursive=True))
        if cands:
            ck = cands[-1]
            break
    if ck is None:
        return {"error": "no rfdetr checkpoint found"}
    model = RFDETRSmall(pretrain_weights=ck)
    # test 集 COCO 评测
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(ROOT / "data/datasets/frisbee_coco/test/_annotations.coco.json"))
    dets = []
    img_ids = gt.getImgIds()
    t0 = time.time()
    n = 0
    for i, iid in enumerate(img_ids):
        info = gt.loadImgs(iid)[0]
        p = ROOT / "data/datasets/frisbee_coco/test" / info["file_name"]
        img = cv2.imread(str(p))
        det = model.predict(img, threshold=0.25)
        for xyxy, c, cid in zip(det.xyxy, det.confidence, det.class_id):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            dets.append({"image_id": iid, "category_id": 0, "score": float(c),
                         "bbox": [x1, y1, x2 - x1, y2 - y1]})
        if i == 10:
            t0 = time.time()  # 暖机
            n = 0
        n += 1
    infer_fps = round(n / max(time.time() - t0, 1e-6), 1)
    if dets:
        E = COCOeval(gt, None, "bbox")
        E.cocoDt = gt.loadRes(dets)
        E.params.imgIds = img_ids
        E.evaluate()
        E.accumulate()
        E.summarize()
        map5095, map50 = float(E.stats[0]), float(E.stats[1])
    else:
        map5095 = map50 = 0.0
    params = None
    for holder in (model.model, model.model.model if hasattr(model.model, "model") else None, model):
        if holder is None:
            continue
        try:
            params = sum(p.numel() for p in holder.parameters()) / 1e6
            break
        except Exception:
            continue
    return {"mAP50": round(map50, 4), "mAP50-95": round(map5095, 4),
            "params_M": round(params, 2) if params else None, "fps_1080p_equivalent": infer_fps,
            "resolution": 672, "weights": ck,
            "note": "fps 为 672 分辨率直推，与 ultralytics@1280 不同分辨率，仅供量级参考"}


def main():
    summary = {}
    for name, w in ULTRALYTICS_BENCH.items():
        if not w.exists():
            summary[name] = {"error": f"missing {w}"}
            print(name, "missing, skip")
            continue
        print(f"eval {name} ...", flush=True)
        summary[name] = eval_ultralytics(name, w)
        print(name, summary[name], flush=True)
    print("eval rfdetr ...", flush=True)
    summary["rfdetr-small"] = eval_rfdetr()
    print("rfdetr-small", summary["rfdetr-small"], flush=True)
    summary["_meta"] = {
        "dataset": "frisbee_merged_v2 (train 4629 / test 476, 同标尺 test split)",
        "recipe": "box=5 epochs=30 patience=8 imgsz=1280(batch2) | rfdetr: res672 bs4 ep30",
        "weights_policy": "全部官方 COCO 预训练默认权重起跑（同一水平）",
        "fps_note": "ultralytics@1280 直推 vs rfdetr@672 直推，分辨率不同仅作量级参考",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
