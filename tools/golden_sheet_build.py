#!/usr/bin/env python3
"""金标准候选拼图生成器（纯本地图像处理，不做任何识别）。

把 data/golden_set_100/candidates.json 的全部候选框裁剪（2 倍上下文）拼成
编号 contact sheet，供 ZCode 会话本体逐张读图判读（见 glm-batch-recognition-ask-first
记忆的授权约束：识图只能在 ZCode 内进行，不写外部 API 脚本）。

用法:
    python tools/golden_sheet_build.py [--cells-per-sheet 36] [--seed 42]

输出:
    data/golden_set_100/sheets/sheet_NN.jpg      编号拼图
    data/golden_set_100/sheets/manifest.json     格号 -> 候选元数据映射（判读后回填用）
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/golden_set_100/candidates.json"
FRAMES_DIR = ROOT / "data/golden_set_100/frames"
OUT_DIR = ROOT / "data/golden_set_100/sheets"

CONTEXT_FACTOR = 2.0  # 裁剪边长 = 框长边 × 2（保留上下文）
CELL = 256            # 单元格边长 px
COLS = 6              # 每行格数 → 6×6=36 格/张
LABEL_H = 34          # 格号条高度


def load_candidates() -> list[dict]:
    cand = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    items = []
    for frame, boxes in sorted(cand.items()):
        for i, b in enumerate(boxes):
            items.append({
                "id": f"{frame[:-4]}_c{i:02d}",
                "frame": frame,
                "bbox": [float(x) for x in b["bbox"]],
                "conf": float(b["conf"]),
                "n_models": int(b["n_models"]),
            })
    return items


def crop_cell(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 8.0) * CONTEXT_FACTOR
    # 正方形裁剪，越界钳位
    half = side / 2
    ax1 = int(max(0, cx - half)); ay1 = int(max(0, cy - half))
    ax2 = int(min(w, cx + half)); ay2 = int(min(h, cy + half))
    if ax2 - ax1 < 4 or ay2 - ay1 < 4:
        return np.full((CELL, CELL, 3), 40, np.uint8)
    crop = img[ay1:ay2, ax1:ax2]
    # 短边不足时补边到正方形（灰边），避免拉伸变形
    ch, cw = crop.shape[:2]
    s = max(ch, cw)
    padded = np.full((s, s, 3), 40, np.uint8)
    padded[(s - ch) // 2:(s - ch) // 2 + ch, (s - cw) // 2:(s - cw) // 2 + cw] = crop
    return cv2.resize(padded, (CELL, CELL), interpolation=cv2.INTER_CUBIC)


def draw_label(cell: np.ndarray, num: int) -> np.ndarray:
    bar = np.full((LABEL_H, CELL, 3), 25, np.uint8)
    text = f"{num:03d}"
    cv2.putText(bar, text, (6, LABEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 3, cv2.LINE_AA)
    return np.vstack([bar, cell])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-per-sheet", type=int, default=36)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = load_candidates()
    rng = random.Random(args.seed)
    rng.shuffle(items)  # 帧序均匀混合，避免判读顺序偏差

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "cell": CELL, "sheets": []}
    rows = (args.cells_per_sheet + COLS - 1) // COLS
    sheet_w = COLS * CELL
    sheet_h = rows * (CELL + LABEL_H)

    for s in range(0, len(items), args.cells_per_sheet):
        chunk = items[s:s + args.cells_per_sheet]
        sheet_idx = s // args.cells_per_sheet
        board = np.full((sheet_h, sheet_w, 3), 12, np.uint8)
        cells_meta = []
        for j, item in enumerate(chunk):
            img = cv2.imread(str(FRAMES_DIR / item["frame"]))
            cell = crop_cell(img, item["bbox"])
            cell = draw_label(cell, j)
            r, c = divmod(j, COLS)
            y0 = r * (CELL + LABEL_H)
            x0 = c * CELL
            board[y0:y0 + CELL + LABEL_H, x0:x0 + CELL] = cell
            cells_meta.append({"cell": j, **item})
        out = OUT_DIR / f"sheet_{sheet_idx:02d}.jpg"
        cv2.imwrite(str(out), board, [cv2.IMWRITE_JPEG_QUALITY, 90])
        manifest["sheets"].append({"sheet": out.name, "cells": cells_meta})
        print(f"{out.name}: {len(chunk)} cells")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"total {len(items)} candidates -> {len(manifest['sheets'])} sheets")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
