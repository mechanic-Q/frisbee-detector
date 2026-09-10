"""GPU 队列 Phase 2：等 win_gpu_queue（D-FINE）退出 → 11s 重训 → D-FINE 评测 →
统一评测 → prod 断点恢复。全部串行，杜绝多进程并发崩溃。
运行: python tools/win_queue_phase2.py（nohup 后台自驱）
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/bili_final_test/win_queue_phase2.log"
DFINE_OUT = ROOT / "data/bili_final_test/dfine_out"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def dfine_train_alive() -> bool:
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, timeout=30)
        return ("D-FINE" in r.stdout and "train.py" in r.stdout)
    except Exception:
        return False


def wait_dfine_exit(timeout_h: float = 8.0):
    t0 = time.time()
    while dfine_train_alive():
        if time.time() - t0 > timeout_h * 3600:
            log("等待 D-FINE 超时，继续")
            return
        log(f"等 D-FINE 训练完成... ({round((time.time()-t0)/60)}min)")
        time.sleep(120)
    log("D-FINE 训练进程已退出")


def run(cmd, tag, cwd=None):
    log(f"{tag} START")
    r = subprocess.run(cmd, cwd=cwd)
    log(f"{tag} EXIT={r.returncode}")
    return r.returncode


def main():
    log("=== Phase 2 队列启动：等 D-FINE 完成 ===")
    wait_dfine_exit()

    # 1) 11s 重训（Windows 原生，无 cache）
    run([sys.executable, "models/train.py",
         "--data", "configs/frisbee_merged_v2_win.yaml",
         "--model", "yolo11s.pt",
         "--box", "5", "--epochs", "30", "--patience", "8", "--close-mosaic", "5",
         "--name", "bench_11s", "--workers", "4"], "11s-retrain")

    # 2) D-FINE test 评测（COCOeval @640）
    eval_dfine = [
        sys.executable, "-c", f'''
import sys, json, time
sys.path.insert(0, r"{DFINE_OUT.parents[1] / 'D-FINE'}")
import os; os.chdir(r"{DFINE_OUT.parents[1] / 'D-FINE'}")
import torch, cv2
import torchvision.transforms.functional as TF
from pathlib import Path
from src.core import YAMLConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
cfg = YAMLConfig(r"{ROOT / 'D-FINE/configs/dfine/dfine_hgnetv2_s_frisbee.yml'}")
model = cfg.model
state = torch.load(r"{DFINE_OUT / 'checkpoint_best_ema.pth'}", map_location="cpu")
model.load_state_dict(state.get("model", state))
model.eval().cuda()
cocoGt = COCO(r"{ROOT / 'data/datasets/frisbee_coco/test/_annotations.coco.json'}")
img_ids = cocoGt.getImgIds()
dets = []
t0 = time.time(); n = 0
import torchvision.transforms.functional as TF
for iid in img_ids:
    info = cocoGt.loadImgs(iid)[0]
    img = cv2.imread(str(Path(r"{ROOT / 'data/datasets/frisbee_coco/test'}") / info["file_name"]))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(TF.resize(TF.to_pil_image(rgb), [640, 640])).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(tensor)
    logits = out["pred_logits"][0]; boxes = out["pred_boxes"][0]
    scores = logits.sigmoid().squeeze(-1) if logits.dim() == 2 else logits.sigmoid()
    keep = scores > 0.25
    for b, s in zip(boxes[keep], scores[keep]):
        cx, cy, w, h = [float(v) for v in b]
        dets.append({{"image_id": iid, "category_id": 0, "score": float(s),
                     "bbox": [(cx-w/2)*info["width"], (cy-h/2)*info["height"],
                              w*info["width"], h*info["height"]]}})
    if n == 10: t0 = time.time(); n = 0
    n += 1
E = COCOeval(cocoGt, None, "bbox"); E.cocoDt = cocoGt.loadRes(dets)
E.params.imgIds = img_ids; E.evaluate(); E.accumulate(); E.summarize()
p = sum(p.numel() for p in model.parameters()) / 1e6
json.dump({{"mAP50": round(float(E.stats[1]),4), "mAP50-95": round(float(E.stats[0]),4),
           "params_M": round(p,2), "fps_640": round(n/max(time.time()-t0,1e-6),1)}},
          open(r"{DFINE_OUT / 'test_eval.json'}", "w"), indent=1)
print("DFINE EVAL DONE")
''']
    run(eval_dfine, "dfine-eval")

    # 3) 统一评测（v8s/11s/26s ultralytics test + FPS + 汇总）
    run([sys.executable, "tools/bench_eval.py"], "bench-eval")

    # 4) prod 断点恢复
    prod_last = ROOT / "runs/detect/bili_prod_v8sp2/weights/last.pt"
    if prod_last.exists():
        wait_gpu = "等 GPU 空闲..."
        log(f"prod-resume {wait_gpu}")
        run([sys.executable, "-c",
             f"from ultralytics import YOLO; YOLO(r'{prod_last}').train(resume=True)"],
            "prod-resume")


if __name__ == "__main__":
    main()
