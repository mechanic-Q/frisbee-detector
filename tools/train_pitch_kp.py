"""E5 phase-3：场线/关键点检测模型训练脚手架（合成数据驱动，零人工标注）。

设计（依据调研：SCCvSD/SoccerSynth 合成→真实先例 + 飞盘场仅 6 线 + 8 关键点）：
- 输入：tools/render_synth_field.py 生成的合成帧 + json 真值（H、8 关键点、可见性）
- 任务：YOLO-pose 关键点回归（8 点）——比场线分割更直接、标注即真值
- 输出：runs/detect/pitch_kp/weights/best.pt，供 auto_calibrate 的"合成数据档"使用
- 评测：在合成 held-out 上算关键点 PCK@0.05，再用真实帧目检反投

用法：
  python tools/train_pitch_kp.py --gen 5000      # 生成 5000 合成帧 + 训练
  python tools/train_pitch_kp.py --eval-only PATH
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

KPT_NAMES = ["c_00", "c_100_00", "c_100_37", "c_0_37",
             "ezA_00", "ezA_37", "ezB_00", "ezB_37"]
DATASET = ROOT / "data/datasets/pitch_kp"


def gen_dataset(n_train: int, n_val: int, seed: int = 42):
    """生成 YOLO-pose 格式数据集：images/ + labels/(class cx cy w h px1 py1 v1 ...)"""
    from synth_field import render

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split, n in (("train", n_train), ("val", n_val)):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    total = n_train + n_val
    stats = {"frames": 0, "visible_kpts": 0}
    for i in range(total):
        img, meta = render.render_one(rng)
        split = "train" if i < n_train else "val"
        stem = f"kpt_{i:06d}"
        cv2.imwrite(str(DATASET / "images" / split / f"{stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        h, w = img.shape[:2]
        kps = meta["keypoints"]
        vis = [k for k in kps if k["visible"]]
        stats["frames"] += 1
        stats["visible_kpts"] += len(vis)
        if not vis:
            # 无可见关键点的帧不产生 pose 标签（纯背景帧，保留图片可作负样本）
            (DATASET / "labels" / split / f"{stem}.txt").write_text("")
            continue
        xs = [k["x"] for k in vis]
        ys = [k["y"] for k in vis]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        # 外扩 10% 保证所有点在内
        pad = 0.1 * max(x2 - x1, y2 - y1)
        x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for k in kps:
            if k["visible"]:
                parts += [f"{k['x'] / w:.6f}", f"{k['y'] / h:.6f}", "2"]
            else:
                parts += ["0", "0", "0"]
        (DATASET / "labels" / split / f"{stem}.txt").write_text(" ".join(parts))

    yaml_text = f"""path: {DATASET.as_posix()}
train: images/train
val: images/val
kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]
nc: 1
names: [pitch]
"""
    (ROOT / "configs/pitch_kp.yaml").write_text(yaml_text)
    print(f"dataset: {stats['frames']} frames, {stats['visible_kpts']} visible kpts -> {DATASET}")
    return stats


def train(epochs: int = 60, batch: int = 4, imgsz: int = 1280, name: str = "pitch_kp"):
    from ultralytics import YOLO

    model = YOLO("yolov8s-pose.yaml")
    r = model.train(
        data=str(ROOT / "configs/pitch_kp.yaml"),
        epochs=epochs, batch=batch, imgsz=imgsz, patience=15,
        name=name, project=str(ROOT / "runs/detect"),
        exist_ok=True, plots=False, seed=42,
    )
    return r


def eval_kpt(weights: str, n_eval: int = 200):
    """合成 held-out 上的关键点误差（PCK@0.05，阈值按画面宽度 5%）。"""
    from ultralytics import YOLO

    m = YOLO(weights)
    val_imgs = sorted((DATASET / "images/val").glob("*.jpg"))[:n_eval]
    errors = []
    for p in val_imgs:
        gt_txt = DATASET / "labels/val" / (p.stem + ".txt")
        if not gt_txt.exists() or not gt_txt.read_text().strip():
            continue
        parts = gt_txt.read_text().split()
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = np.array([float(v) for v in parts[5:]]).reshape(-1, 3)
        res = m.predict(img, imgsz=1280, conf=0.3, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            continue
        kp = res.keypoints.xy.cpu().numpy()[0]  # (8,2)
        for i in range(min(len(gt), len(kp))):
            if gt[i, 2] > 0:
                err = float(np.hypot(kp[i, 0] - gt[i, 0] * w, kp[i, 1] - gt[i, 1] * h))
                errors.append(err / w)
    if not errors:
        return {"pck@0.05": None, "n": 0}
    arr = np.array(errors)
    return {"pck@0.05": round(float((arr <= 0.05).mean()), 4),
            "median_err_frac": round(float(np.median(arr)), 4),
            "n": len(arr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=0, help="生成 N 张训练合成帧（0=跳过）")
    ap.add_argument("--gen-val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--eval-only", default=None, help="只评测指定权重")
    ap.add_argument("--name", default="pitch_kp")
    args = ap.parse_args()

    if args.eval_only:
        print(json.dumps(eval_kpt(args.eval_only), indent=1))
        return
    if args.gen:
        gen_dataset(args.gen, args.gen_val)
    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, name=args.name)
    best = ROOT / f"runs/detect/{args.name}/weights/best.pt"
    if best.exists():
        print("KPT EVAL:", json.dumps(eval_kpt(str(best)), indent=1))


if __name__ == "__main__":
    main()
