"""Step 1: 金标准 100 帧分层抽帧。
50 帧全局均匀 + 50 帧检测导向（用 prod 模型 1fps 扫全片找有检测的帧）。
运行: python tools/golden_set_sampling.py
输出: data/golden_set_100/frames/*.jpg + sampling_report.json
"""
import json
import os
import subprocess
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "movie/2024城市飞盘俱乐部锦标赛『决赛』【北京大哥 VS 上海沪蛙】高清版3-1.mp4"
OUT = ROOT / "data/golden_set_100"
FRAMES_DIR = OUT / "frames"
PROD_W = ROOT / "runs/detect/bili_prod_v8sp2/weights/best.pt"
DETECT_SCAN = OUT / "detect_scan.json"

N_UNIFORM = 50
N_DETECT = 50
SCAN_STRIDE = 30          # 1fps 扫描
SCAN_CONF = 0.15          # 低阈值：最大化候选召回


def scan_frames():
    """用 prod 模型 1fps 扫全片，返回 {frame_idx: [{bbox,conf}, ...]}"""
    if DETECT_SCAN.exists():
        print("scan cache exists, load")
        return json.loads(DETECT_SCAN.read_text())
    from ultralytics import YOLO
    m = YOLO(str(PROD_W))
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = {}
    idx = 0
    n_scanned = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        r = m.predict(frame, conf=SCAN_CONF, imgsz=1280, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            out[str(idx)] = [
                {"bbox": [round(float(v), 1) for v in b], "conf": round(float(c), 3)}
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().tolist())
            ]
        idx += SCAN_STRIDE
        n_scanned += 1
        if n_scanned % 300 == 0:
            print(f"  scanned {n_scanned} frames ({idx}/{total}), hits={len(out)}", flush=True)
    cap.release()
    DETECT_SCAN.write_text(json.dumps(out))
    print(f"scan done: {len(out)} hit frames / {n_scanned} scanned")
    return out


def main():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 1) 全局均匀 50 帧
    uniform = [int(total * (i + 0.5) / N_UNIFORM) for i in range(N_UNIFORM)]

    # 2) 检测导向 50 帧
    hits = scan_frames()
    hit_frames = sorted(int(k) for k in hits)
    print(f"hit frames: {len(hit_frames)}")
    if len(hit_frames) >= N_DETECT:
        # 在命中的帧里均匀取样，保证时间分散
        step = len(hit_frames) / N_DETECT
        detect_picks = [hit_frames[int(i * step)] for i in range(N_DETECT)]
    else:
        detect_picks = hit_frames

    # 3) 合并去重（保留时间顺序）
    all_picks = sorted(set(uniform) | set(detect_picks))
    print(f"total picked: {len(all_picks)} (uniform {len(uniform)} + detect {len(detect_picks)})")

    # 4) 抽帧存盘
    meta = []
    for i, fno in enumerate(all_picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            continue
        stem = f"f{fno:06d}"
        cv2.imwrite(str(FRAMES_DIR / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta.append({
            "frame_idx": fno,
            "t_sec": round(fno / fps, 2),
            "file": f"{stem}.jpg",
            "sampling": "uniform" if fno in set(uniform) else "detect",
            "scan_hits": len(hits.get(str(fno), [])),
        })
        if (i + 1) % 25 == 0:
            print(f"  extracted {i+1}/{len(all_picks)}", flush=True)
    cap.release()

    report = {
        "video": str(VIDEO), "fps": fps, "total_frames": total,
        "n_uniform": len(uniform), "n_detect_hits": len(hit_frames),
        "n_picked": len(meta), "scan_conf": SCAN_CONF, "scan_stride": SCAN_STRIDE,
        "frames": meta,
    }
    (OUT / "sampling_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"done: {len(meta)} frames -> {FRAMES_DIR}")


if __name__ == "__main__":
    main()
