#!/usr/bin/env python3
"""F1 Phase C: 5 格对照矩阵实验（管线级检测器裁决）。

素材: movie/25866279684-1-192_55-56min.mp4 (720P/25fps, 唯一有标定的真实片段)
标尺: 场外投影率 / 最长连续轨迹 / 状态分布 / 事件数 (Gate-C G1/G2/G4)

用法: python tools/f1_matrix.py [--max-frames 300]
产物: results/f1_matrix/matrix.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
VIDEO = ROOT / "movie/25866279684-1-192_55-56min.mp4"
CALIB = ROOT / "configs/homography/25866279684-1-192.json"
OUT = ROOT / "results/f1_matrix"
SHADOW = ROOT / "runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt"
DFINE_PTH = ROOT / "data/bili_final_test/dfine_out/best_stg1.pth"

CELLS = {
    # ① prod 基线（现状默认：yolo26x person 零样本，无盘）
    "c1_prod_baseline": {
        "args": ["--video", str(VIDEO), "--weights", "yolo26x.pt",
                 "--calibration", str(CALIB)],
    },
    # ② shadow_v1@0.15 裸盘检测（无融合）
    "c2_shadow_bare": {
        "args": ["--video", str(VIDEO), "--weights", "yolo26x.pt",
                 "--disc-weights", str(SHADOW), "--disc-conf", "0.15",
                 "--calibration", str(CALIB)],
    },
    # ③ shadow_v1 + 融合
    "c3_shadow_fusion": {
        "args": ["--video", str(VIDEO), "--weights", "yolo26x.pt",
                 "--disc-weights", str(SHADOW), "--disc-conf", "0.15",
                 "--disc-fusion", "--calibration", str(CALIB)],
    },
}


def run_cell(name: str, cell_args: list[str], max_frames: int) -> dict | None:
    out_dir = OUT / name
    cmd = [PY, "-m", "frisbee_analyzer.pipeline", *cell_args,
           "--max-frames", str(max_frames), "--output-dir", str(out_dir)]
    print(f"[matrix] {name}: start", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[matrix] {name}: FAIL exit={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}", flush=True)
        return None
    tracks = json.loads((out_dir / "tracks.json").read_text(encoding="utf-8"))
    print(f"[matrix] {name}: done in {dt:.0f}s", flush=True)
    return {"runtime_s": round(dt, 1), "tracks": tracks, "stdout": r.stdout[-4000:]}


def analyze(name: str, cell: dict, max_frames: int) -> dict:
    """管线级指标（Gate-C G1/G2 的素材侧）：盘状态分布、轨迹连续性、场外投影率。"""
    from utils.homography import load_calibration, pixel_to_world
    calib = load_calibration(str(CALIB))
    m = calib["matrix"]
    ch, cw = calib.get("image_size", [1280, 720])
    sx = 1280 / cw  # 视频即 1280x720，此标定原生匹配
    disc = cell["tracks"].get("disc_frames", {})
    statuses = {}
    longest = cur = 0
    prev_idx = None
    oof = in_field = 0
    speeds = []
    for idx_str in sorted(disc, key=int):
        d = disc[idx_str]
        st = d.get("status", "raw")
        statuses[st] = statuses.get(st, 0) + 1
        # 连续 tracking 长度（帧号连续才算连续轨迹）
        idx = int(idx_str)
        if st in ("tracking", "predicting"):
            cur = cur + 1 if (prev_idx is not None and idx - prev_idx == 1) else 1
            longest = max(longest, cur)
        else:
            cur = 0
        prev_idx = idx
        # 场外投影率（对有像素坐标的条目）
        cx, cy = d.get("cx"), d.get("cy")
        if cx is not None:
            try:
                wx, wy = pixel_to_world(m, cx, cy)
                if 0 <= wx <= 100 and 0 <= wy <= 37:
                    in_field += 1
                else:
                    oof += 1
                if d.get("speed_ms") is not None:
                    speeds.append(d["speed_ms"])
            except Exception:
                oof += 1
    total = sum(statuses.values()) or 1
    return {
        "cell": name,
        "n_disc_frames": len(disc),
        "n_player_frames": len(cell["tracks"].get("frames", {})),
        "status_dist": statuses,
        "longest_contiguous_frames": longest,
        "longest_contiguous_s": round(longest / 25.0, 2),
        "field_in": in_field,
        "field_out": oof,
        "field_out_rate": round(oof / max(in_field + oof, 1), 3),
        "speed_ms_median": sorted(speeds)[len(speeds)//2] if speeds else None,
        "runtime_s": cell["runtime_s"],
        "max_frames": max_frames,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--cells", type=str, default=None, help="逗号分隔，只跑指定格")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cells = CELLS
    if args.cells:
        want = set(args.cells.split(","))
        cells = {k: v for k, v in CELLS.items() if k in want}

    results = {}
    for name, spec in cells.items():
        mark = OUT / f"{name}.done"
        if mark.exists():
            print(f"[matrix] {name}: 已有标记，跳过", flush=True)
            results[name] = json.loads(mark.read_text(encoding="utf-8"))
            continue
        cell = run_cell(name, spec["args"], args.max_frames)
        if cell is None:
            results[name] = {"error": "pipeline failed"}
            continue
        summary = analyze(name, cell, args.max_frames)
        mark.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        results[name] = summary

    # 汇总表
    print("\n===== F1 Matrix Summary =====")
    for name, s in results.items():
        if "error" in s:
            print(f"{name}: ERROR")
            continue
        print(f"{name}: disc_frames={s['n_disc_frames']} longest={s['longest_contiguous_frames']}f"
              f"({s['longest_contiguous_s']}s) field_out_rate={s['field_out_rate']} "
              f"status={s['status_dist']}")
    (OUT / "matrix.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {OUT / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
