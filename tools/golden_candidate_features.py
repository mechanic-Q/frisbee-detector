"""金标准判据修正：加入独立外观特征（白度+圆度），破除"同源共识误检"。
对每个候选框裁剪 → 计算：
  - white_ratio: 高亮度低饱和像素占比（飞盘是白色）
  - circularity: 轮廓圆度（飞盘是圆形，草纹/帽子/杂物不规则）
  - fill_ratio: 白色区域占框面积比
输出: data/golden_set_100/candidate_features.json + 阈值分析
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("FRISBEE_ROOT", str(Path(__file__).resolve().parents[1])))
GOLD = ROOT / "data/golden_set_100"
FRAMES = GOLD / "frames"


def features(img, bbox, pad=6):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    xa, ya = max(x1 - pad, 0), max(y1 - pad, 0)
    xb, yb = min(x2 + pad, w), min(y2 + pad, h)
    crop = img[ya:yb, xa:xb]
    if crop.size < 20:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # 白色像素：亮度高 + 饱和度低
    white = (V > 140) & (S < 80)
    white_ratio = float(white.mean())
    # 在框内中心区域算（排除边缘草）
    ch, cw = crop.shape[:2]
    cy0, cy1 = int(ch * 0.2), int(ch * 0.8)
    cx0, cx1 = int(cw * 0.2), int(cw * 0.8)
    center_white = float(white[cy0:cy1, cx0:cx1].mean()) if (cy1 > cy0 and cx1 > cx0) else 0.0
    # 圆度：对白色掩码找最大轮廓
    mask = (white * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    fill = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        if perim > 0:
            circularity = float(4 * np.pi * area / (perim * perim))
        bx, by, bw, bh = cv2.boundingRect(c)
        fill = float(area / (bw * bh)) if bw * bh > 0 else 0.0
    # 长宽比（飞盘近似正方形框，但透视下可扁）
    bw_, bh_ = x2 - x1, y2 - y1
    aspect = max(bw_, bh_) / max(min(bw_, bh_), 1)
    return {"white_ratio": round(white_ratio, 4), "center_white": round(center_white, 4),
            "circularity": round(circularity, 4), "fill": round(fill, 4),
            "aspect": round(aspect, 3), "size_px": round(max(bw_, bh_), 1),
            "w": round(bw_, 1), "h": round(bh_, 1)}


def main():
    cands = json.loads((GOLD / "candidates.json").read_text())
    out = {}
    n = 0
    for fname in sorted(cands.keys()):
        img = cv2.imread(str(FRAMES / fname))
        if img is None:
            continue
        items = []
        for b in cands[fname]:
            f = features(img, b["bbox"])
            if f is None:
                continue
            items.append({"bbox": b["bbox"], "conf": b["conf"], "n_models": b["n_models"], **f})
        out[fname] = items
        n += len(items)
    (GOLD / "candidate_features.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"features computed for {n} candidates in {len(out)} frames")

    # 阈值分析：看 >=3 模型共识框的外观特征分布
    strong = [it for v in out.values() for it in v if it["n_models"] >= 3]
    print(f"\n>=3 共识框: {len(strong)}")
    for key in ("white_ratio", "center_white", "circularity", "fill", "size_px"):
        vals = sorted(it[key] for it in strong)
        if vals:
            print(f"  {key:14s} min={vals[0]:.3f} p25={vals[len(vals)//4]:.3f} "
                  f"median={vals[len(vals)//2]:.3f} p75={vals[3*len(vals)//4]:.3f} max={vals[-1]:.3f}")

    # 用严格外观判据过滤后的数量
    for thr in (0.10, 0.15, 0.20, 0.25):
        keep = [it for it in strong if it["center_white"] >= thr and it["circularity"] >= 0.35
                and 10 <= it["size_px"] <= 100]
        print(f"  严格判据(center_white>={thr}, circ>=0.35, 10-100px): {len(keep)} 框")


if __name__ == "__main__":
    main()
