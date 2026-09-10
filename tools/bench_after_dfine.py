"""D-FINE 训练完成后的自动接力：test 评测 → prod 恢复。
运行: python tools/bench_after_dfine.py
等待 D-FINE 训练进程退出 → 自动执行剩余评测与 prod 恢复。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/bili_final_test/bench_after_dfine.log"
DFINE_CKPT = ROOT / "data/bili_final_test/dfine_out/checkpoint_best_ema.pth"
TEST_ANN = ROOT / "data/datasets/frisbee_coco/test/_annotations.coco.json"
TEST_DIR = ROOT / "data/datasets/frisbee_coco/test"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_gpu_free(timeout_min=600):
    t0 = time.time()
    while True:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            if not r.stdout.strip():
                return
        except Exception:
            return
        if time.time() - t0 > timeout_min * 60:
            log("等待超时")
            return
        log("GPU 占用中，等 60s...")
        time.sleep(60)


def eval_dfine_test():
    sys.path.insert(0, str(ROOT / "D-FINE"))
    os.chdir(str(ROOT / "D-FINE"))
    import torch
    import torchvision.transforms.functional as TF
    from src.core import YAMLConfig
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_s_frisbee.yml")
    model = cfg.model
    state = torch.load(str(DFINE_CKPT), map_location="cpu")
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
    log(f"dfine test eval: {json.dumps(result)}")


def eval_11s_test():
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "runs/detect/bench_11s/weights/best.pt"))
    r = m.val(data=str(ROOT / "configs/frisbee_merged_v2_win.yaml"),
              split="test", imgsz=1280, batch=4, verbose=False)
    result = {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
              "P": round(float(r.box.mp), 4), "R": round(float(r.box.mr), 4)}
    out_path = ROOT / "results/11s_test_eval.json"
    out_path.write_text(json.dumps(result, indent=1))
    log(f"11s test eval: {json.dumps(result)}")


def resume_prod():
    prod_last = ROOT / "runs/detect/bili_prod_v8sp2/weights/last.pt"
    if not prod_last.exists():
        log("prod last.pt not found, skip")
        return
    code = f"from ultralytics import YOLO; YOLO(r'{prod_last}').train(resume=True)"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=14400)
    log(f"prod resume exit={r.returncode}")


import os  # noqa: E402

if __name__ == "__main__":
    log("=== bench_after_dfine: 等 GPU 空闲 ===")
    wait_gpu_free(600)
    log("GPU 空闲，开始评测")
    eval_dfine_test()
    log("=== eval_11s ===")
    eval_11s_test()
    log("=== prod resume ===")
    resume_prod()
    log("=== ALL DONE ===")
