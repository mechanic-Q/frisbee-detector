"""金标准帧并排对比可视化：D-FINE-S（第1） vs prod v8s-P2（第2） vs 金标准GT。
输出: data/golden_set_100/review/compare_*.jpg（每帧 3 列并排）
       data/golden_set_100/review/compare_index.json（每帧统计）
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("FRISBEE_ROOT", str(Path(__file__).resolve().parents[1])))
GOLD = ROOT / "data/golden_set_100"
FRAMES = GOLD / "frames"
REVIEW = GOLD / "review"
INFER = ROOT / "results/golden_inference"

# 各模型最佳工作点（来自 golden_metrics 的 best_f1）
CONF = {"dfine_s": 0.30, "prod_v8s_p2": 0.15}
PANEL_W = 1280          # 每列宽度
N_PANELS = 8            # 展示帧数


def load(name):
    return json.loads((INFER / f"{name}.json").read_text())


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def draw(img, dets, gt, conf_th, title, color):
    """画检测框（绿=命中GT, 红=误检）与GT框（黄色虚线）。"""
    vis = img.copy()
    matched = [False] * len(gt)
    n_tp = n_fp = 0
    for d in sorted(dets, key=lambda x: -x["conf"]):
        if d["conf"] < conf_th:
            continue
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        bi, bv = -1, 0.0
        for i, g in enumerate(gt):
            if matched[i]:
                continue
            v = iou(d["bbox"], g)
            if v > bv:
                bi, bv = i, v
        if bv >= 0.5:
            matched[bi] = True
            n_tp += 1
            c = (0, 220, 0)
        else:
            n_fp += 1
            c = (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 3)
        cv2.putText(vis, f'{d["conf"]:.2f}', (x1, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    n_fn = 0
    for i, g in enumerate(gt):
        x1, y1, x2, y2 = [int(v) for v in g]
        if not matched[i]:
            n_fn += 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    # 标题条
    bar = np.zeros((46, vis.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{title}  TP{n_tp} FP{n_fp} FN{n_fn} (conf>={conf_th})",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return np.vstack([bar, vis])


def main():
    cands = json.loads((GOLD / "candidates.json").read_text())
    dfine = load("dfine_s")
    prod = load("prod_v8s_p2")

    # 金标准（与 golden_metrics 同判据）
    def gt_of(fname):
        out = []
        for b in cands.get(fname, []):
            x1, y1, x2, y2 = b["bbox"]
            if b["n_models"] >= 3 and b["conf"] >= 0.30 and 6 <= max(x2 - x1, y2 - y1) <= 80:
                out.append(b["bbox"])
        return out

    # 选帧：优先"有GT且两模型表现不同"的帧（信息量大）
    scored = []
    for fname in sorted(cands.keys()):
        gt = gt_of(fname)
        if not gt:
            continue
        d_dets = [d for d in dfine["frames"].get(fname, []) if d["conf"] >= CONF["dfine_s"]]
        p_dets = [d for d in prod["frames"].get(fname, []) if d["conf"] >= CONF["prod_v8s_p2"]]
        diff = abs(len(d_dets) - len(p_dets))
        scored.append((diff, len(gt), fname, gt, d_dets, p_dets))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    picks = scored[:N_PANELS]

    index = []
    for k, (diff, ngt, fname, gt, d_dets, p_dets) in enumerate(picks):
        img = cv2.imread(str(FRAMES / fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        # GT 面板
        gt_vis = img.copy()
        for g in gt:
            x1, y1, x2, y2 = [int(v) for v in g]
            cv2.rectangle(gt_vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
        bar = np.zeros((46, w, 3), np.uint8)
        cv2.putText(bar, f"GOLDEN GT: {len(gt)} frisbee(s)  [{fname}]",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        gt_panel = np.vstack([bar, gt_vis])

        panel_d = draw(img, dfine["frames"].get(fname, []), gt, CONF["dfine_s"],
                       "D-FINE-S (rank#1, mAP50 .659)", (0, 220, 0))
        panel_p = draw(img, prod["frames"].get(fname, []), gt, CONF["prod_v8s_p2"],
                       "prod v8s-P2 (rank#2, mAP50 .638)", (255, 160, 0))
        row = np.vstack([gt_panel, panel_d, panel_p])
        scale = PANEL_W / row.shape[1]
        row = cv2.resize(row, (PANEL_W, int(row.shape[0] * scale)))
        cv2.imwrite(str(REVIEW / f"cmp_{k:02d}_{fname}"), row, [cv2.IMWRITE_JPEG_QUALITY, 90])
        index.append({"panel": k, "frame": fname, "gt": len(gt),
                      "dfine_dets": len(d_dets), "prod_dets": len(p_dets)})
        print(f"panel {k}: {fname} gt={len(gt)} dfine={len(d_dets)} prod={len(p_dets)}")

    (REVIEW / "compare_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(f"\n{len(index)} panels -> {REVIEW}/cmp_*.jpg")


if __name__ == "__main__":
    main()
