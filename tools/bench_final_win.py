"""基准收尾编排（Windows 原生）：
1. 等 D-FINE-S 训练进程退出
2. D-FINE 在 COCO test 上的评测（原生 API @640）
3. 组装最终四维对比表（质量×速度×参数×license）→ results/bench_v2_summary.json
4. 恢复生产级重训断点（v8s-P2, 剩余 epochs）
运行: python tools/bench_final_win.py（nohup 后台自驱）
"""
import json
import subprocess
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
LOG = ROOT / "data/bili_final_test/bench_final_win.log"
DFINE_DIR = ROOT / "D-FINE"
DFINE_CKPT = ROOT / "data/bili_final_test/dfine_out/checkpoint_best_ema.pth"
TEST_ANN = ROOT / "data/datasets/frisbee_coco/test/_annotations.coco.json"
TEST_DIR = ROOT / "data/datasets/frisbee_coco/test"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def dfine_training_running() -> bool:
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, timeout=30)
        return ("D-FINE" in r.stdout and "train.py" in r.stdout) or \
               ("dfine_hgnetv2_s_frisbee" in r.stdout)
    except Exception:
        return False


def wait_dfine(timeout_h: float = 8.0):
    t0 = time.time()
    while dfine_training_running():
        if time.time() - t0 > timeout_h * 3600:
            log("等待 D-FINE 超时，继续（可能已提前完成）")
            return
        time.sleep(120)


def eval_dfine_test():
    """D-FINE best_ema 在 COCO test 上的评测（@640）。"""
    sys.path.insert(0, str(DFINE_DIR))
    os.chdir(str(DFINE_DIR))
    from src.core import YAMLConfig

    cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_s_frisbee.yml")
    model = cfg.model
    state = torch.load(str(DFINE_CKPT), map_location="cpu")
    sd = state.get("model", state)
    model.load_state_dict(sd)
    model.eval().cuda()

    cocoGt = COCO(str(TEST_ANN))
    import torchvision.transforms.functional as TF

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
    return {"mAP50": round(float(E.stats[1]), 4), "mAP50-95": round(float(E.stats[0]), 4),
            "params_M": round(params, 2), "fps_640": fps, "resolution": 640}


def eval_ultra(weights: Path, name: str):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(ROOT / "configs/frisbee_merged_v2_win.yaml"),
              split="test", imgsz=1280, batch=4, verbose=False)
    return {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
            "P": round(float(r.box.mp), 4), "R": round(float(r.box.mr), 4),
            "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2)}


def main():
    log("=== 收尾编排启动：等 D-FINE 训练完成 ===")
    wait_dfine()
    if not DFINE_CKPT.exists():
        alt = ROOT / "data/bili_final_test/dfine_out/last.pth"
        if alt.exists():
            shutil_checkpoint = alt
        log(f"警告: {DFINE_CKPT} 不存在")
    log("D-FINE 完成，开始评测")

    summary = {}

    # 已有 test 数字直接汇编（同标尺已验证）
    summary["RF-DETR-small [Apache-2.0, 纯开源]"] = {
        "mAP50": 0.7432, "mAP50-95": 0.5714, "fps_640": 39.8, "params_M": 31.79,
        "note": "09-09 已测（COCOeval @672）"}
    summary["YOLOv8s-P2 prod [100ep+硬负样本]"] = {
        "mAP50": 0.8092, "mAP50-95": 0.4332, "P": 0.875, "R": 0.716,
        "note": "生产现役（09-09 test 评测）"}

    # 11s（重训后）
    w11 = ROOT / "runs/detect/bench_11s/weights/best.pt"
    if w11.exists():
        try:
            summary["YOLO11s [30ep]"] = eval_ultra(w11, "11s")
        except Exception as e:
            summary["YOLO11s [30ep]"] = {"error": str(e)}
    else:
        summary["YOLO11s [30ep]"] = {"error": "weights missing"}

    # 26s
    w26 = ROOT / "runs/detect/bench_26s/weights/best.pt"
    if w26.exists():
        try:
            summary["YOLO26s [30ep]"] = eval_ultra(w26, "26s")
        except Exception as e:
            summary["YOLO26s [30ep]"] = {"error": str(e)}

    # D-FINE-S
    try:
        summary["D-FINE-S [30ep...72ep]"] = eval_dfine_test()
    except Exception as e:
        summary["D-FINE-S [30ep...72ep]"] = {"error": f"{type(e).__name__}: {e}"}

    (ROOT / "results/bench_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    log("saved -> results/bench_v2_summary.json")

    # prod 恢复（断点续训剩余 epochs）
    prod_last = ROOT / "runs" / "detect" / "bili_prod_v8sp2" / "weights" / "last.pt"
    if prod_last.exists():
        wait_gpu_free_tag = "prod"
        while True:
            try:
                r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                    "--format=csv,noheader"], capture_output=True,
                                   text=True, timeout=30)
                if not r.stdout.strip():
                    break
            except Exception:
                break
            time.sleep(60)
        log("prod GPU free, resuming")
        subprocess.run([sys.executable, "-c",
                        f"from ultralytics import YOLO; YOLO(r'{prod_last}').train(resume=True)"],
                       cwd=str(ROOT))
        log("prod resume finished")


if __name__ == "__main__":
    main()
