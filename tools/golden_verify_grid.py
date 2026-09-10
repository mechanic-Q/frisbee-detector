"""生成全部共识框的裁剪网格供人工判定（每格标注外观特征，便于快速判读）。
输出: data/golden_set_100/review/verify_grid_*.jpg + verify_manifest.json
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
SIZE = 180
COLS = 8
PER_BOARD = 64


def crop_zoom(img, bbox, size=SIZE):
    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    h, w = img.shape[:2]
    half = max(30, int(max(x2 - x1, y2 - y1) * 0.9))
    xa, ya = max(cx - half, 0), max(cy - half, 0)
    xb, yb = min(cx + half, w), min(cy + half, h)
    c = img[ya:yb, xa:xb]
    if c.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    ch, cw = c.shape[:2]
    s = size / max(ch, cw)
    c = cv2.resize(c, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - c.shape[0]) // 2, (size - c.shape[1]) // 2
    canvas[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    cv2.drawMarker(canvas, (size // 2, size // 2), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
    return canvas


def main():
    feats = json.loads((GOLD / "candidate_features.json").read_text())
    items = []
    for fname, arr in feats.items():
        for it in arr:
            if it["n_models"] >= 3:
                items.append({"frame": fname, **it})
    # 按"外观像真盘"排序：白度高+圆度高+尺寸合理
    def score(it):
        return (it["center_white"] * 0.5 + it["circularity"] * 0.5)
    items.sort(key=lambda x: -score(x))
    print(f"consensus boxes to verify: {len(items)}")

    manifest = []
    for b in range(0, len(items), PER_BOARD):
        group = items[b:b + PER_BOARD]
        tiles = []
        for k, it in enumerate(group):
            img = cv2.imread(str(FRAMES / it["frame"]))
            if img is None:
                continue
            c = crop_zoom(img, it["bbox"])
            idx = b + k + 1
            cv2.putText(c, f"{idx}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(c, f"w{it['center_white']:.1f} c{it['circularity']:.1f}",
                        (4, SIZE - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            tiles.append(c)
            manifest.append({"idx": idx, "frame": it["frame"], "bbox": it["bbox"],
                             "n_models": it["n_models"], "conf": it["conf"],
                             "center_white": it["center_white"], "circularity": it["circularity"],
                             "size_px": it["size_px"]})
        rows = []
        for i in range(0, len(tiles), COLS):
            row = tiles[i:i + COLS]
            while len(row) < COLS:
                row.append(np.zeros((SIZE, SIZE, 3), np.uint8))
            rows.append(np.hstack(row))
        board = np.vstack(rows)
        cv2.imwrite(str(REVIEW / f"verify_grid_{b//PER_BOARD:02d}.jpg"), board,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  board {b//PER_BOARD}: cells {b+1}-{min(b+PER_BOARD, len(items))}")

    (REVIEW / "verify_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"manifest -> {REVIEW/'verify_manifest.json'}")


if __name__ == "__main__":
    main()
