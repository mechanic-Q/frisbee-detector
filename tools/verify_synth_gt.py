"""合成帧 GT 定量验证：真值 H 投影出的场线必须落在画出的白色像素上。
对每个样本：沿 6 条场地线段世界坐标采样点 → H 投影到像素 →
统计线上一带 (±1px) 与线外横向偏移 (±12px) 的亮度差。差值应显著为正。
运行: python3 tools/verify_synth_gt.py --dir results/synth_preview
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELD_W, FIELD_H = 100.0, 37.0
EZ_A, EZ_B = 18.0, 82.0
SEGMENTS = [
    ((0, 0), (100, 0)), ((100, 0), (100, 37)), ((100, 37), (0, 37)), ((0, 37), (0, 0)),
    ((EZ_A, 0), (EZ_A, 37)), ((EZ_B, 0), (EZ_B, 37)),
]


def apply_h(H, pts_xy):
    p = np.hstack([pts_xy, np.ones((len(pts_xy), 1))])
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


def sample_brightness(img, px, py, k=1):
    h, w = img.shape[:2]
    x, y = int(round(px)), int(round(py))
    if not (k <= x < w - k and k <= y < h - k):
        return None
    patch = img[y - k:y + k + 1, x - k:x + k + 1].astype(np.float32)
    return float(patch.max())


def verify_sample(img_path: Path, meta_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    meta = json.loads(meta_path.read_text())
    H = np.array(meta["H"], dtype=np.float64)
    on_vals, off_vals = [], []
    for (a, b) in SEGMENTS:
        ts = np.linspace(0.02, 0.98, 40)
        world = np.array([[a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t] for t in ts])
        px = apply_h(H, world)
        for (x, y), t in zip(px, ts):
            if not (0 <= x < img.shape[1] - 12 and 0 <= y < img.shape[0] - 12):
                continue
            on = sample_brightness(gray, x, y)
            # 横向偏移采样背景（垂直于线段方向的图像偏移近似为 y 偏移）
            off1 = sample_brightness(gray, x, y - 12)
            off2 = sample_brightness(gray, x, y + 12)
            if on is not None and off1 is not None and off2 is not None:
                on_vals.append(on)
                off_vals.append((off1 + off2) / 2)
    if not on_vals:
        return {"in_frame_samples": 0}
    on_m, off_m = float(np.mean(on_vals)), float(np.mean(off_vals))
    return {"in_frame_samples": len(on_vals), "online_brightness": round(on_m, 1),
            "offline_brightness": round(off_m, 1), "contrast": round(on_m - off_m, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="results/synth_preview")
    args = ap.parse_args()
    d = ROOT / args.dir
    rows = []
    for meta_path in sorted(d.glob("synth_*.json")):
        img_path = d / (meta_path.stem + ".jpg")
        if not img_path.exists():
            continue
        r = verify_sample(img_path, meta_path)
        if r:
            rows.append({"sample": meta_path.stem, **r})
    good = [r for r in rows if r.get("in_frame_samples", 0) >= 50]
    contrasts = [r["contrast"] for r in good]
    summary = {
        "samples": len(rows),
        "samples_with_lines_in_frame": len(good),
        "mean_contrast_online_vs_offline": round(float(np.mean(contrasts)), 1) if contrasts else None,
        "min_contrast": round(float(np.min(contrasts)), 1) if contrasts else None,
        "verdict": "PASS" if contrasts and float(np.mean(contrasts)) > 30 else "FAIL/INSUFFICIENT",
    }
    print(json.dumps({"summary": summary, "per_sample": rows}, indent=1))
    (d / "gt_verification.json").write_text(json.dumps({"summary": summary, "per_sample": rows}, indent=1))


if __name__ == "__main__":
    main()
