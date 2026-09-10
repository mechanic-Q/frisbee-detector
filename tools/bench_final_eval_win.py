"""最终评测编排（Windows 原生）：等 D-FINE 训练完成 → 四模型统一评测 → 最终四维表。
运行: python tools/bench_final_eval_win.py（nohup 后台）
产出: results/bench_v2_summary.json
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/bili_final_test/bench_final_eval.log"
DFINE_CKPT_CANDIDATES = [
    ROOT / "data/bili_final_test/dfine_out/checkpoint_best_ema.pth",
    ROOT / "data/bili_final_test/dfine_out/last.pth",
]
UL_WEIGHTS = {
    "yolov8s(标准头)": ROOT / "runs/detect/bench_v8s/weights/best.pt",
    "yolo11s(标准头)": ROOT / "runs/detect/bench_11s/weights/best.pt",
    "yolo26s(标准头)": ROOT / "runs/detect/bench_26s/weights/best.pt",
    "yolov8s-p2(30ep参考)": ROOT / "runs/detect/bench_v8sp2_ref/weights/best.pt",
}


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def dfine_running() -> bool:
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
            capture_output=True, text=True, timeout=30)
        return "D-FINE" in r.stdout and "train.py" in r.stdout
    except Exception:
        return False


def wait_dfine(timeout_h: float = 8.0):
    t0 = time.time()
    while dfine_running():
        if time.time() - t0 > timeout_h * 3600:
            log("dfine wait timeout")
            return
        time.sleep(120)
    # 再等 60s 让文件落盘
    time.sleep(60)


def eval_ultra(name, weights: Path):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(ROOT / "configs/frisbee_merged_v2_win.yaml"),
              split="test", imgsz=1280, batch=4, verbose=False)
    t0 = time.time()
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(ROOT / "movie/2024城市飞盘俱乐部锦标赛『决赛』【北京大哥 VS 上海沪蛙】高清版3-1.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, 200).astype(int)
    n = 0
    for k, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        m.predict(f, imgsz=1280, conf=0.35, verbose=False)
        if k == 10:
            t0 = time.time()
            n = 0
        n += 1
    cap.release()
    fps = round(n / max(time.time() - t0, 1e-6), 1)
    return {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
            "P": round(float(r.box.mp), 4), "R": round(float(r.box.mr), 4),
            "fps_1080p": fps,
            "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
            "weights": str(weights)}


def eval_dfine(ckpt: Path):
    from rfdetr import RFDETRSmall
    import torch
    # D-FINE 权重用 rfdetr 的加载器不兼容——用 D-FINE 自带推理接口
    sys_path = str(ROOT / "D-FINE")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import yaml as yaml_lib

    from src.core import YAMLConfig
    cfg = YAMLConfig(str(ROOT / "D-FINE/configs/dfine/dfine_hgnetv2_s_frisbee.yml"))
    # 用 checkpoint 恢复模型权重做推理
    import torch as th
    state = th.load(str(ckpt), map_location="cpu")
    model = cfg.model
    model.load_state_dict(state["model"] if "model" in state else state)

    import cv2
    gt_ann = ROOT / "data/datasets/frisbee_coco/test/_annotations.coco.json"
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cocoGt = COCO(str(gt_ann))
    dets = []
    img_ids = cocoGt.getImgIds()
    model.eval()
    t0 = time.time()
    n = 0
    import torchvision
    T = torchvision.transforms.functional
    for iid in img_ids:
        info = cocoGt.loadImgs(iid)[0]
        p = ROOT / "data/datasets/frisbee_coco/test" / info["file_name"]
        img = cv2.imread(str(p))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = T.to_tensor(T.resize(T.to_pil_image(rgb), [640, 640])).unsqueeze(0)
        with th.no_grad():
            out = model(tensor)
        # D-FINE 输出: {'pred_logits': (1,300,1), 'pred_boxes': (1,300,4) cxcywh 归一化}
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
    if not dets:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "note": "no detections"}
    E = COCOeval(cocoGt, None, "bbox")
    E.cocoDt = cocoGt.loadRes(dets)
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    fps = 0.0  # fps 单测另测
    return {"mAP50": round(float(E.stats[1]), 4), "mAP50-95": round(float(E.stats[0]), 4),
            "fps_640": fps}


def main():
    log("=== 最终评测编排启动：等待 D-FINE 训练完成 ===")
    wait_dfine()
    log("D-FINE 训练结束，开始统一评测")

    summary = {}

    # 1) Ultralytics 家族（Windows 权重路径）
    for name, w in UL_WEIGHTS.items():
        if not w.exists():
            summary[name] = {"error": f"missing {w}"}
            continue
        try:
            summary[name] = eval_ultra(name, w)
        except Exception as e:
            summary[name] = {"error": f"{type(e).__name__}: {e}"}
        log(f"{name}: {summary[name].get('mAP50')}")

    # 2) RF-DETR-small（沿用已测数字，刷新存在性）
    try:
        from bench_eval_shared import rfdetr_numbers  # noqa
    except Exception:
        pass
    summary["rfdetr-small(纯开源)"] = {
        "mAP50": 0.7432, "mAP50-95": 0.5714, "fps_1080p_equivalent": 39.8,
        "params_M": 31.79, "resolution": 672,
        "note": "沿用 09-09 已测数字（同标尺）",
    }

    # 3) D-FINE-S（pycocotools COCOeval on test）
    ck = next((c for c in DFINE_CKPT_CANDIDATES if c.exists()), None)
    if ck:
        try:
            summary["dfine-s(纯开源)"] = eval_dfine_ck(ck)
        except Exception as e:
            summary["dfine-s(纯开源)"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        summary["dfine-s(纯开源)"] = {"error": "checkpoint not found"}

    (ROOT / "results/bench_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    log("saved -> results/bench_v2_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def eval_dfine_ck(ck: Path):
    """D-FINE checkpoint 在 COCO test 上的评测（原生 API）。"""
    import torch as th
    sys_path = str(ROOT / "D-FINE")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from src.core import YAMLConfig
    import torchvision
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cfg = YAMLConfig(str(ROOT / "D-FINE/configs/dfine/dfine_hgnetv2_s_frisbee.yml"))
    model = cfg.model
    state = th.load(str(ck), map_location="cpu")
    sd = state.get("model", state)
    model.load_state_dict(sd)
    model.eval().cuda()

    cocoGt = COCO(str(ROOT / "data/datasets/frisbee_coco/test/_annotations.coco.json"))
    import torchvision.transforms.functional as TF
    dets = []
    img_ids = cocoGt.getImgIds()
    t0 = time.time()
    n = 0
    for iid in img_ids:
        info = cocoGt.loadImgs(iid)[0]
        p = ROOT / "data/datasets/frisbee_coco/test" / info["file_name"]
        img = cv2.imread(str(p))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(TF.resize(TF.to_pil_image(rgb), [640, 640])).unsqueeze(0).cuda()
        with th.no_grad():
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
            "params_M": round(params, 2), "fps_640": fps}


if __name__ == "__main__":
    main()
