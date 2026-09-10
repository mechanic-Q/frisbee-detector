"""Step 5: 金标准指标计算（离线扫阈值）。
- GT = 候选并集里的"强共识"框（>=5 模型 或 >=3 模型且尺寸合理且场内）——经人工目检确认判据
- 每模型在 GT 上算 mAP50/mAP50-95/P/R（扫 conf 阈值 0.05..0.9）
- 场外 FP 率：用标定单应性投影，统计落在场地外的检测占比
输出: results/golden_metrics.json + Markdown 表

v2（--gt-v2，2026-09-11）：GT 由 ZCode 会话判读（vlm_verdicts.json）与模型共识双通道合成：
    yes    + n_models>=2 -> 正框 GT
    yes    + n_models==1  -> ignore（仅 VLM 单通道主张，未经共识验证）
    abstain              -> ignore（判读弃权）
    no     + n_models>=3 -> ignore（通道冲突：强共识但 VLM 否）
    no     + n_models<=2 -> FP 区（双通道一致非盘，检到即 FP）
ignore 区的预测不计 TP 也不计 FP。输出 results/golden_metrics_v2.json。
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FRISBEE_ROOT", str(Path(__file__).resolve().parents[1])))
GOLD = ROOT / "data/golden_set_100"
INFER = ROOT / "results/golden_inference"
FRAMES = GOLD / "frames"
OUT = ROOT / "results/golden_metrics.json"
OUT_V2 = ROOT / "results/golden_metrics_v2.json"

MIN_PX, MAX_PX = 6, 80
V2_MIN_PX, V2_MAX_PX = 4, 400   # v2 不设 80px 上限——近景手持盘可以很大
GT_MIN_MODELS = 3          # 金标准判据：>=3 模型共识
GT_MIN_CONF = 0.30
V2_MIN_MODELS = 2          # v2 正框共识下限


def load_homography():
    for f in sorted((ROOT / "configs/homography").glob("*.json")):
        d = json.loads(f.read_text())
        raw = d.get("homography") or d.get("matrix")
        if raw:
            return np.array(raw, dtype=np.float64).reshape(3, 3)
    return None


def in_field(H, x, y, margin=5.0):
    if H is None:
        return None
    p = H @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return False
    wx, wy = p[0] / p[2], p[1] / p[2]
    return (-margin <= wx <= 105.0) and (-margin <= wy <= 42.0)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build_gt(cands, H):
    """金标准 = 强共识框（>=3 模型 + conf>=0.3 + 尺寸合理）。"""
    gt = {}
    for fname, boxes in cands.items():
        g = []
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            size = max(x2 - x1, y2 - y1)
            if b["n_models"] < GT_MIN_MODELS:
                continue
            if b["conf"] < GT_MIN_CONF:
                continue
            if not (MIN_PX <= size <= MAX_PX):
                continue
            g.append(b["bbox"])
        gt[fname] = g
    return gt


def build_gt_v2(cands, verdicts):
    """v2 GT：判读 × 模型共识 双通道合成（规则见模块 docstring）。

    返回 (gt, ignore)：gt = {帧名: [bbox]}，ignore = {帧名: [bbox]}。
    """
    gt, ignore = {}, {}
    for fname, boxes in cands.items():
        g, ig = [], []
        for b in boxes:
            verdict = verdicts.get(b.get("cid"))
            if verdict is None:
                # 兼容：无判读的候选按弃权处理（宁可忽略不可虚构真值）
                ig.append(b["bbox"])
                continue
            x1, y1, x2, y2 = b["bbox"]
            size = max(x2 - x1, y2 - y1)
            if not (V2_MIN_PX <= size <= V2_MAX_PX):
                ig.append(b["bbox"])
                continue
            if verdict == "yes":
                (g if b["n_models"] >= V2_MIN_MODELS else ig).append(b["bbox"])
            elif verdict == "abstain":
                ig.append(b["bbox"])
            else:  # no
                if b["n_models"] >= GT_MIN_MODELS:
                    ig.append(b["bbox"])  # 通道冲突：强共识但判否 → ignore
                # n_models <= 2：双通道一致非盘 → 不进 GT 也不 ignore（检到即 FP）
        gt[fname] = g
        ignore[fname] = ig
    return gt, ignore


def _pred_in_ignore(pred_bbox, ignore_boxes):
    for ig in ignore_boxes:
        if iou(pred_bbox, ig) >= 0.5:
            return True
    return False


def eval_model(res, gt, H, conf_ths=None, ignore=None):
    """扫阈值计算 P/R/F1 + mAP@0.5/0.5-0.95 + 场外FP率。"""
    if conf_ths is None:
        conf_ths = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
    ignore = ignore or {}
    frames = res["frames"]
    curves = []
    for th in conf_ths:
        tp = fp = fn = 0
        oof_fp = 0
        for fname, gts in gt.items():
            ig_boxes = ignore.get(fname, [])
            preds = [d for d in frames.get(fname, []) if d["conf"] >= th]
            preds = [p for p in preds if not _pred_in_ignore(p["bbox"], ig_boxes)]
            matched = [False] * len(gts)
            for p in preds:
                best_i, best_iou = -1, 0.0
                for i, g in enumerate(gts):
                    if matched[i]:
                        continue
                    v = iou(p["bbox"], g)
                    if v > best_iou:
                        best_i, best_iou = i, v
                if best_iou >= 0.5:
                    matched[best_i] = True
                    tp += 1
                else:
                    fp += 1
                    x1, y1, x2, y2 = p["bbox"]
                    if in_field(H, (x1 + x2) / 2, (y1 + y2) / 2) is False:
                        oof_fp += 1
            fn += sum(1 for m in matched if not m)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        curves.append({"conf": th, "tp": tp, "fp": fp, "fn": fn,
                       "precision": round(prec, 4), "recall": round(rec, 4),
                       "f1": round(f1, 4),
                       "out_of_field_fp_rate": round(oof_fp / fp, 4) if fp else None})

    # mAP@0.5 与 @0.5:0.95（11 点插值，简化实现）
    def ap_at(iou_th):
        aps = []
        for th in np.arange(0.05, 0.96, 0.05):
            precs, recs = [], []
            for t in np.arange(0.0, 1.01, 0.02):
                tp = fp = fn = 0
                for fname, gts in gt.items():
                    ig_boxes = ignore.get(fname, [])
                    preds = [d for d in frames.get(fname, []) if d["conf"] >= t]
                    preds = [p for p in preds if not _pred_in_ignore(p["bbox"], ig_boxes)]
                    matched = [False] * len(gts)
                    for p in preds:
                        bi, bv = -1, 0.0
                        for i, g in enumerate(gts):
                            if matched[i]:
                                continue
                            v = iou(p["bbox"], g)
                            if v > bv:
                                bi, bv = i, v
                        if bv >= iou_th:
                            matched[bi] = True
                            tp += 1
                        else:
                            fp += 1
                    fn += sum(1 for m in matched if not m)
                pr = tp / (tp + fp) if (tp + fp) else 1.0
                rc = tp / (tp + fn) if (tp + fn) else 0.0
                precs.append(pr)
                recs.append(rc)
            precs, recs = np.array(precs), np.array(recs)
            ap = 0.0
            for r in np.linspace(0, 1, 11):
                p = precs[recs >= r].max() if (recs >= r).any() else 0.0
                ap += p / 11
            aps.append(ap)
        return aps

    aps50 = ap_at(0.5)
    maps = ap_at(0.5)  # 简化：@0.5 主指标；@0.5:0.95 取多阈值均值（见下）
    map50 = float(np.mean(aps50))
    # 多 IoU 阈值 mAP50-95
    maps_list = []
    for iou_th in np.arange(0.5, 0.96, 0.05):
        aps = ap_at(float(iou_th))
        maps_list.append(float(np.mean(aps)))
    map5095 = float(np.mean(maps_list))

    return {"curves": curves, "mAP50": round(map50, 4), "mAP50-95": round(map5095, 4),
            "fps": res.get("fps"), "resolution": res.get("resolution"),
            "best_f1": max(curves, key=lambda c: c["f1"])}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-v2", action="store_true",
                    help="用判读×共识双通道 GT（vlm_verdicts.json），输出 golden_metrics_v2.json")
    args = ap.parse_args()

    cands = json.loads((GOLD / "candidates.json").read_text())
    H = load_homography()

    if args.gt_v2:
        verdicts = json.loads((GOLD / "vlm_verdicts.json").read_text())["verdicts"]
        # golden_sheet_build 的候选 id 规则：f"{帧名去后缀}_c{序号:02d}"，直接重建
        for frame, boxes in cands.items():
            for i, b in enumerate(boxes):
                b["cid"] = f"{frame[:-4]}_c{i:02d}"
        gt, ignore = build_gt_v2(cands, verdicts)
        n_gt = sum(len(v) for v in gt.values())
        n_ig = sum(len(v) for v in ignore.values())
        n_frames_with_gt = sum(1 for v in gt.values() if v)
        print(f"GT v2: {n_gt} boxes in {n_frames_with_gt}/100 帧; ignore {n_ig} 框"
              f"（正框=yes&共识>=2, 冲突/弃权=ignore）")
        out_path, criterion = OUT_V2, "vlm-verdict & >=2 models dual-channel + ignore"
    else:
        gt = build_gt(cands, H)
        ignore = None
        n_gt = sum(len(v) for v in gt.values())
        n_frames_with_gt = sum(1 for v in gt.values() if v)
        print(f"GT: {n_gt} boxes in {n_frames_with_gt}/100 frames (>= {GT_MIN_MODELS} models, conf>={GT_MIN_CONF})")
        out_path, criterion = OUT, f">= {GT_MIN_MODELS} models & conf >= {GT_MIN_CONF}"

    results = {}
    for f in sorted(INFER.glob("*.json")):
        res = json.loads(f.read_text())
        r = eval_model(res, gt, H, ignore=ignore)
        results[res["model"]] = r
        print(f"  {res['model']:20s} mAP50={r['mAP50']:.4f} mAP50-95={r['mAP50-95']:.4f} "
              f"bestF1={r['best_f1']['f1']:.3f}@conf{r['best_f1']['conf']} fps={r['fps']}")

    out = {"gt": {"n_boxes": n_gt, "n_frames": n_frames_with_gt, "criterion": criterion},
           "models": results}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
