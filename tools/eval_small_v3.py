#!/usr/bin/env python3
"""§13.20 路线 1 复测三门（训成后跑）：
① val mAP 不回退 ② chunk4 P(在管|可见) 对照 ③ 误检率重抽维持 <10%

用法: python tools/eval_small_v3.py
前置: runs/detect/frisbee_small_v3/weights/best.pt 已训成
"""
import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / "runs/detect/frisbee_small_v3/weights/best.pt"
VIDEO = ROOT / "data/bili_final_test/accept_chunk_4.mp4"
OUT = ROOT / "results/f1_matrix/smallv3_chunk4"
CALIB = ROOT / "results/f1_matrix/phaseA_chunk4/segment_calib.json"
BASELINE_TRACKS = ROOT / "results/f1_matrix/reacq2_chunk4/tracks.json"
VISIBLE = ROOT / "results/f1_matrix/visible_candidates_v2.json"


def main() -> int:
    if not WEIGHTS.exists():
        print(f"weights not found: {WEIGHTS}")
        return 1

    # ① val mAP（train.py --validate-only）
    print("== Gate 1: val mAP")
    r = subprocess.run([sys.executable, "models/train.py", "--validate-only",
                        "--model-path", str(WEIGHTS)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    val_line = [l for l in (r.stdout or "").splitlines() if "mAP50" in l or "all" in l]
    print("\n".join(val_line[-3:]) or r.stdout[-500:])

    # ② chunk4 全管线（新权重）
    print("== Gate 2: chunk4 P(在管|可见)")
    r = subprocess.run([sys.executable, "-m", "frisbee_analyzer.pipeline",
                        "--video", str(VIDEO), "--tile-grid", "2",
                        "--disc-weights", str(WEIGHTS), "--disc-conf", "0.35",
                        "--disc-tile-grid", "2", "--disc-tile-trigger=conf<0.5",
                        "--disc-fusion", "--hand-roi", "--possession-impute",
                        "--auto-calibrate", "--output-dir", str(OUT)],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    for line in (r.stdout or "").splitlines():
        if any(k in line for k in ("disc-fusion", "possession-impute", "events")):
            print(line[:180])

    new = json.loads((OUT / "tracks.json").read_text(encoding="utf-8"))
    base = json.loads(BASELINE_TRACKS.read_text(encoding="utf-8"))
    cand = json.loads(VISIBLE.read_text(encoding="utf-8"))
    vis = {int(k) for k in cand}
    for name, doc in (("baseline", base), ("small_v3", new)):
        disc = doc.get("disc_frames", {})
        inpipe = {int(k) for k, v in disc.items() if v.get("status") in ("tracking", "predicting")}
        trk = sum(1 for v in disc.values() if v.get("status") == "tracking")
        inter = len(vis & inpipe)
        print(f"  {name}: P(在管|可见)={inter/len(vis):.0%} tracking={trk} 在管={len(inpipe)} "
              f"({len(inpipe)/len(doc.get('frames', {})):.0%})")

    # ③ 误检率重抽 40 crop
    print("== Gate 3: 误检率重抽（拼图供人工判读）")
    import cv2
    import numpy as np
    trk = {int(k): v for k, v in new["disc_frames"].items() if v.get("status") == "tracking"}
    random.seed(99)
    sample = random.sample(sorted(trk), min(40, len(trk)))
    cap = cv2.VideoCapture(str(VIDEO))
    cache = {}
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx in sample:
            cache[idx] = f
        idx += 1
        if idx > max(sample):
            break
    cap.release()
    crops = []
    for fi in sample:
        det = trk[fi]
        x1, y1, x2, y2 = det["bbox"]
        f = cache.get(fi)
        if f is None:
            continue
        h, w = f.shape[:2]
        crop = f[max(0, int(y1) - 20):min(h, int(y2) + 20),
                 max(0, int(x1) - 20):min(w, int(x2) + 20)]
        if crop.size == 0:
            continue
        crops.append(cv2.resize(crop, (80, 80)))
    rows = []
    for i in range(0, len(crops), 8):
        row = crops[i:i + 8]
        while len(row) < 8:
            row.append(np.zeros((80, 80, 3), dtype="uint8"))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    cv2.putText(sheet, f"small_v3 tracking crops n={len(crops)} (random seed 99)",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    out_png = ROOT / "results/f1_matrix/audit_sheet_small_v3.jpg"
    cv2.imwrite(str(out_png), sheet)
    print(f"  拼图 -> {out_png}（人工判读误检数）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
