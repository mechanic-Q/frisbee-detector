#!/usr/bin/env python3
"""v3 硬负样本采样拼图：全片扫描检出(conf>=0.30, 排除金标准帧) → 裁剪拼图 →
ZCode 会话判读确认"非飞盘" → 含确认裁剪的帧作为 v3 空标签背景负样本。

与 golden_sheet_build.py 同一套拼图格式（36 格/张，编号左上角，manifest 映射）。

用法:
    python tools/hardneg_sheet_build.py [--max 600] [--video movie/xxx.mp4]
输出:
    data/hardneg_v3/sheets/sheet_NN.jpg + manifest.json（cell -> {frame_idx, bbox, conf}）
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from golden_sheet_build import CELL, COLS, LABEL_H, draw_label  # 同一套格式

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "data/golden_set_100/detect_scan.json"
GOLDEN_FRAMES = ROOT / "data/golden_set_100/frames"
OUT = ROOT / "data/hardneg_v3"
GRID = 128  # 中心点量化网格（去重保多样性）
ROWS = (36 + COLS - 1) // COLS


def sample_candidates(max_n: int, seed: int = 42):
    golden = {int(p.stem[1:]) for p in GOLDEN_FRAMES.glob("f*.jpg")}
    ds = json.loads(SCAN.read_text(encoding="utf-8"))
    cand = []
    for k, dets in ds.items():
        fn = int(k)
        if fn in golden:
            continue  # 防泄漏：金标准评测帧绝不进训练
        for d in sorted(dets, key=lambda x: -x["conf"])[:2]:  # 每帧最多 2 个
            if d["conf"] >= 0.30:
                cand.append((fn, [float(x) for x in d["bbox"]], float(d["conf"])))
    # 中心点网格去重：每格最多 2 个
    rng = random.Random(seed)
    rng.shuffle(cand)
    cells: dict[tuple, int] = {}
    picked = []
    for fn, bbox, conf in cand:
        cx = int((bbox[0] + bbox[2]) / 2) // GRID
        cy = int((bbox[1] + bbox[3]) / 2) // GRID
        key = (cx, cy)
        if cells.get(key, 0) >= 2:
            continue
        cells[key] = cells.get(key, 0) + 1
        picked.append((fn, bbox, conf))
        if len(picked) >= max_n:
            break
    return sorted(picked)  # 按帧序返回，便于顺序抽帧


def extract_crops(video_path: Path, picked):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"cannot open {video_path}"
    crops, wanted, ui = {}, sorted({fn for fn, _, _ in picked}), 0
    while ui < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        while ui < len(wanted) and wanted[ui] < fi:
            ui += 1  # 视频丢帧保护
        if ui < len(wanted) and wanted[ui] == fi:
            img = frame
            for (fn, bbox, _conf) in [p for p in picked if p[0] == fi]:
                h, w = img.shape[:2]
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                side = max(x2 - x1, y2 - y1, 8.0) * 2.0
                half = side / 2
                ax1, ay1 = int(max(0, cx - half)), int(max(0, cy - half))
                ax2, ay2 = int(min(w, cx + half)), int(min(h, cy + half))
                crop = img[ay1:ay2, ax1:ax2] if (ax2 - ax1 > 4 and ay2 - ay1 > 4) \
                    else np.full((32, 32, 3), 40, np.uint8)
                ch, cw = crop.shape[:2]
                s = max(ch, cw)
                padded = np.full((s, s, 3), 40, np.uint8)
                padded[(s - ch) // 2:(s - ch) // 2 + ch, (s - cw) // 2:(s - cw) // 2 + cw] = crop
                crops[(fn, tuple(bbox))] = cv2.resize(padded, (CELL, CELL), interpolation=cv2.INTER_CUBIC)
            ui += 1
    cap.release()
    return crops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--video", type=str,
                    default="movie/2024城市飞盘俱乐部锦标赛『决赛』【北京大哥 VS 上海沪蛙】高清版3-1.mp4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    video = (ROOT / args.video) if not Path(args.video).is_absolute() else Path(args.video)
    picked = sample_candidates(args.max, args.seed)
    print(f"采样 {len(picked)} 候选 / {len({p[0] for p in picked})} 帧，顺序抽帧中...")
    crops = extract_crops(video, picked)
    print(f"裁剪 {len(crops)} 个")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(exist_ok=True)
    manifest = {"video": str(video), "seed": args.seed, "sheets": []}
    sheet_h = ROWS * (CELL + LABEL_H)
    sheet_w = COLS * CELL
    items = list(crops.items())
    for s in range(0, len(items), 36):
        chunk = items[s:s + 36]
        idx = s // 36
        board = np.full((sheet_h, sheet_w, 3), 12, np.uint8)
        cells_meta = []
        for j, ((fn, bbox), crop) in enumerate(chunk):
            cell = draw_label(crop, j)
            r, c = divmod(j, COLS)
            board[r * (CELL + LABEL_H):r * (CELL + LABEL_H) + CELL + LABEL_H, c * CELL:c * CELL + CELL] = cell
            cells_meta.append({"cell": j, "frame_idx": fn, "bbox": list(bbox)})
        out = OUT / "sheets" / f"sheet_{idx:02d}.jpg"
        cv2.imwrite(str(out), board, [cv2.IMWRITE_JPEG_QUALITY, 90])
        manifest["sheets"].append({"sheet": out.name, "cells": cells_meta})
        print(f"{out.name}: {len(chunk)} cells")
    (OUT / "sheets" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"manifest -> {OUT / 'sheets' / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
