"""Step 3c: 高共识框裁剪目检网格（人工作为独立验证通道，替代不可用的 VLM）。
输出: data/golden_set_100/review/crop_grid_*.jpg  每格 = 一个候选框的放大裁剪
       data/golden_set_100/review/crops_manifest.json  每格对应的 (frame, bbox, n_models)
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
CROP_SIZE = 200
PAD = 40
COLS = 6


def crop_around(img, x1, y1, x2, y2, pad=PAD, size=CROP_SIZE):
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    h, w = img.shape[:2]
    half = pad + int(max(x2 - x1, y2 - y1) / 2)
    xa, ya = max(cx - half, 0), max(cy - half, 0)
    xb, yb = min(cx + half, w), min(cy + half, h)
    c = img[ya:yb, xa:xb]
    if c.size == 0:
        return np.zeros((size, size, 3), np.uint8), None
    # 放大到固定尺寸
    ch, cw = c.shape[:2]
    s = size / max(ch, cw)
    c = cv2.resize(c, (max(1, int(cw * s)), max(1, int(ch * s))))
    canvas = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - c.shape[0]) // 2, (size - c.shape[1]) // 2
    canvas[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    # 画中心十字标记原框中心
    cv2.drawMarker(canvas, (size // 2, size // 2), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
    return canvas, (xa, ya, xb, yb)


def main():
    cands = json.loads((GOLD / "candidates.json").read_text())
    # 收集全部候选，按共识度降序
    items = []
    for fname, boxes in cands.items():
        for i, b in enumerate(boxes):
            items.append({"frame": fname, "idx": i, "bbox": b["bbox"],
                          "n_models": b["n_models"], "conf": b["conf"]})
    items.sort(key=lambda x: (-x["n_models"], -x["conf"]))

    # 取前 72 个（12 行 × 6 列）做目检
    picks = items[:72]
    manifest = []
    tiles = []
    for k, it in enumerate(picks):
        img = cv2.imread(str(FRAMES / it["frame"]))
        if img is None:
            continue
        c, region = crop_around(img, *it["bbox"])
        cv2.putText(c, f"{k+1}: {it['n_models']}m", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(c)
        manifest.append({"cell": k + 1, **it})

    rows = []
    for i in range(0, len(tiles), COLS):
        row = tiles[i:i + COLS]
        while len(row) < COLS:
            row.append(np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(str(REVIEW / "crop_grid_top72.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (REVIEW / "crops_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"crop grid: {len(tiles)} tiles -> {REVIEW/'crop_grid_top72.jpg'}")
    print("cells 1-72, sorted by consensus (n_models desc, conf desc)")


if __name__ == "__main__":
    main()
