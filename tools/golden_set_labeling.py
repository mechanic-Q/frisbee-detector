"""Step 2+3: 从全模型推理结果构建候选并集 → GLM-4V 交叉校验 → 生成金标准标注任务。
运行: python tools/golden_set_labeling.py [--stage candidates|vlm|report]
输出:
  data/golden_set_100/candidates.json   候选并集（每帧的 cluster 后候选框）
  data/golden_set_100/vlm_verdicts.json GLM-4V 逐帧判断（含盘数/位置描述）
  data/golden_set_100/labels/*.txt      金标准 YOLO 标注（vlm 通过 or 保守保留）
  data/golden_set_100/label_report.json 一致率与需人工复核清单
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("FRISBEE_ROOT", str(Path(__file__).resolve().parents[1])))
GOLD = ROOT / "data/golden_set_100"
FRAMES = GOLD / "frames"
INFER = ROOT / "results/golden_inference"
LABELS = GOLD / "labels"

# 候选聚类参数
IOU_MERGE = 0.45      # 跨模型框合并阈值
MIN_MODELS = 1        # 至少 1 个模型命中（最大化召回）
# 分层入口阈值：DETR 系（dfine/rfdetr）输出全部 300 queries，低 conf 噪声极大，
# YOLO 系经 NMS 后噪声小。用不同阈值保证候选质量可比。
ENTRY_CONF = {"ultralytics": 0.10, "dfine": 0.30, "rfdetr": 0.30}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def build_candidates():
    """所有模型推理结果的框并集 → 聚类合并 → 每帧候选列表。"""
    files = sorted(INFER.glob("*.json"))
    if not files:
        print("no inference results found")
        return
    all_res = {f.stem: json.loads(f.read_text()) for f in files}
    print(f"models: {list(all_res)}")

    # 收集全部帧
    frame_names = set()
    for r in all_res.values():
        frame_names |= set(r["frames"].keys())

    out = {}
    for fname in sorted(frame_names):
        boxes = []   # (bbox, conf, model)
        for mname, r in all_res.items():
            thr = ENTRY_CONF.get(r.get("type", "ultralytics"), 0.10)
            for d in r["frames"].get(fname, []):
                if d["conf"] >= thr:
                    boxes.append((d["bbox"], d["conf"], mname))
        if not boxes:
            out[fname] = []
            continue
        # 贪心聚类：按 conf 降序，与已有簇 IoU>阈值则并入
        boxes.sort(key=lambda x: -x[1])
        clusters = []   # {"bbox": ..., "max_conf": ..., "models": set, "n": int}
        for bbox, conf, mname in boxes:
            placed = False
            for c in clusters:
                if iou(bbox, c["bbox"]) > IOU_MERGE:
                    c["models"].add(mname)
                    c["n"] += 1
                    if conf > c["max_conf"]:
                        c["max_conf"] = conf
                        c["bbox"] = bbox
                    placed = True
                    break
            if not placed:
                clusters.append({"bbox": bbox, "max_conf": conf,
                                 "models": {mname}, "n": 1})
        cands = [{"bbox": c["bbox"], "conf": round(c["max_conf"], 4),
                  "n_models": len(c["models"]), "models": sorted(c["models"])}
                 for c in clusters]
        cands.sort(key=lambda c: (-c["n_models"], -c["conf"]))
        out[fname] = cands

    (GOLD / "candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    n_frames = len(out)
    n_cands = sum(len(v) for v in out.values())
    multi = sum(1 for v in out.values() for c in v if c["n_models"] >= 3)
    print(f"candidates: {n_frames} frames, {n_cands} boxes, {multi} boxes hit by >=3 models")
    hist = defaultdict(int)
    for v in out.values():
        for c in v:
            hist[c["n_models"]] += 1
    print("n_models hist:", dict(sorted(hist.items())))


def vlm_judge_batch(batch_size=6):
    """GLM-4V 对每帧判断飞盘数量（独立通道，用于交叉校验）。"""
    api_key = None
    for cand in (ROOT / "tools/vlm_review_crops.py",):
        if cand.exists():
            m = re.search(r'GLM_API_KEY = "([^"]+)"', cand.read_text(encoding="utf-8"))
            if m:
                api_key = m.group(1)
    api_key = os.environ.get("GLM_API_KEY") or api_key
    if not api_key:
        print("no GLM API key found")
        return
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4")

    frames = sorted(FRAMES.glob("*.jpg"))
    out_path = GOLD / "vlm_verdicts.json"
    verdicts = json.loads(out_path.read_text()) if out_path.exists() else {}

    PROMPT = ("这是一帧极限飞盘比赛画面。请数出画面中可见的飞盘(白色小圆盘/碟)数量。"
              "只输出一个数字（0/1/2/3），不要解释。若不确定输出 -1。")

    todo = [f for f in frames if f.name not in verdicts]
    print(f"vlm judge: {len(todo)} frames to process (done: {len(verdicts)})")
    for i, fp in enumerate(todo):
        img = cv2.imread(str(fp))
        # 为提升小目标识别：中心区域裁剪放大
        h, w = img.shape[:2]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b64 = base64.b64encode(buf).decode()
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model="glm-4v-flash",
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": PROMPT}]}],
                    max_tokens=8, temperature=0.1)
                txt = (r.choices[0].message.content or "").strip()
                m = re.search(r"-?\d+", txt)
                verdicts[fp.name] = int(m.group(0)) if m else -1
                break
            except Exception as e:
                if attempt == 2:
                    verdicts[fp.name] = -1
                time.sleep(2 * (attempt + 1))
        if (i + 1) % 10 == 0:
            out_path.write_text(json.dumps(verdicts, ensure_ascii=False, indent=1))
            print(f"  {i+1}/{len(todo)}", flush=True)
    out_path.write_text(json.dumps(verdicts, ensure_ascii=False, indent=1))
    print(f"vlm verdicts saved: {len(verdicts)}")


def build_labels():
    """候选并集 + VLM 校验 → 金标准标注（保守策略 + 复核清单）。"""
    cands = json.loads((GOLD / "candidates.json").read_text())
    vlm_path = GOLD / "vlm_verdicts.json"
    verdicts = json.loads(vlm_path.read_text()) if vlm_path.exists() else {}

    LABELS.mkdir(parents=True, exist_ok=True)
    review = []
    stats = {"frames": 0, "kept": 0, "dropped_vlm0": 0, "review_flagged": 0}

    for fname, boxes in cands.items():
        img = cv2.imread(str(FRAMES / fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        vlm_n = verdicts.get(fname, -1)   # -1=未知
        strong = [b for b in boxes if b["n_models"] >= 2]     # 多模型共识
        weak = [b for b in boxes if b["n_models"] == 1]       # 单模型

        if vlm_n == 0:
            keep = []                      # VLM 明确说没有盘 → 全丢
            stats["dropped_vlm0"] += 1
        elif vlm_n > 0:
            # VLM 说有盘：保留强共识 + 最强的若干弱框（数量对齐 VLM 判断）
            keep = list(strong)
            if len(keep) < vlm_n:
                keep += weak[: vlm_n - len(keep)]
            if len(strong) != vlm_n:
                review.append({"frame": fname, "reason": f"vlm={vlm_n} but strong={len(strong)}"})
                stats["review_flagged"] += 1
        else:
            # VLM 不确定：保守保留强共识，弱框标记复核
            keep = list(strong)
            if weak:
                review.append({"frame": fname, "reason": f"vlm unknown, {len(weak)} weak boxes"})
                stats["review_flagged"] += 1

        lines = []
        for b in keep:
            x1, y1, x2, y2 = b["bbox"]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (LABELS / (Path(fname).stem + ".txt")).write_text("\n".join(lines))
        stats["frames"] += 1
        stats["kept"] += len(keep)

    report = {
        "stats": stats,
        "vlm_coverage": f"{len(verdicts)}/{len(cands)}",
        "vlm_counts": {str(k): sum(1 for v in verdicts.values() if v == k)
                       for k in sorted(set(verdicts.values()))},
        "review_needed": review,
    }
    (GOLD / "label_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "review_needed"},
                     ensure_ascii=False, indent=1))
    print(f"review list: {len(review)} frames -> {GOLD/'label_report.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "candidates", "vlm", "labels"])
    args = ap.parse_args()
    if args.stage in ("all", "candidates"):
        build_candidates()
    if args.stage in ("all", "vlm"):
        vlm_judge_batch()
    if args.stage in ("all", "labels"):
        build_labels()


if __name__ == "__main__":
    main()
