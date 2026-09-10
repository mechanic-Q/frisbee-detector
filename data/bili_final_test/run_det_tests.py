"""Phase 0 零样本测试：roboflow DFL 球员检测器 + 现有飞盘检测器 在 B 站决赛视频上的对比。
在 WSL 中运行: python3 data/bili_final_test/run_det_tests.py
输出: data/bili_final_test/out/ 下的标注图与检测统计。
"""
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path("/mnt/e/frisbee-detector")
FRAMES_DIR = ROOT / "data/bili_final_test/frames"
OUT_DIR = ROOT / "data/bili_final_test/out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_MODEL = str(ROOT / "models/pretrained/football-player-detection.pt")
FRISBEE_MODEL = str(ROOT / "runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt")

# DFL 类别: 0=ball 1=goalkeeper 2=player 3=referee
PLAYER_CLASSES = [1, 2, 3]
PLAYER_COLORS = {1: (0, 255, 255), 2: (255, 160, 0), 3: (0, 0, 255)}  # BGR


def run_model(model, frame, conf, imgsz, classes=None):
    t0 = time.time()
    res = model.predict(frame, conf=conf, imgsz=imgsz, classes=classes, verbose=False)[0]
    dt = time.time() - t0
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
    clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
    return boxes, confs, clss, dt


def draw(frame, boxes, confs, clss, colors, tag):
    img = frame.copy()
    for (x1, y1, x2, y2), c, k in zip(boxes, confs, clss):
        color = colors.get(int(k), (255, 255, 255)) if colors else (255, 255, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"{int(k)} {c:.2f}", (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.putText(img, tag, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def main():
    frame_paths = sorted(FRAMES_DIR.glob("probe_*.jpg"))
    frames = {p.stem: cv2.imread(str(p)) for p in frame_paths}

    results = {"frames": {}}

    # --- 1) roboflow DFL player detector, zero-shot ---
    pm = YOLO(PLAYER_MODEL)
    for name, frame in frames.items():
        boxes, confs, clss, dt = run_model(pm, frame, conf=0.25, imgsz=1280, classes=PLAYER_CLASSES)
        counts = {}
        for k in clss:
            counts[int(k)] = counts.get(int(k), 0) + 1
        results["frames"][name] = {"player_dets": len(boxes), "player_counts_by_class": counts,
                                   "player_ms": round(dt * 1000)}
        img = draw(frame, boxes, confs, clss, PLAYER_COLORS, f"DFL player zero-shot {name}")
        cv2.imwrite(str(OUT_DIR / f"players_{name}.jpg"), img)
        print(f"[players] {name}: {len(boxes)} dets, by_class={counts}, {dt*1000:.0f} ms")

    # --- 2) existing frisbee detector (p2_shadow_v1, DEFAULT conf) ---
    fm = YOLO(FRISBEE_MODEL)
    for name, frame in frames.items():
        boxes, confs, clss, dt = run_model(fm, frame, conf=0.35, imgsz=1280)
        results["frames"].setdefault(name, {})["frisbee_dets"] = len(boxes)
        results["frames"][name]["frisbee_ms"] = round(dt * 1000)
        img = draw(frame, boxes, confs, clss, None, f"frisbee p2_shadow_v1 {name}")
        cv2.imwrite(str(OUT_DIR / f"frisbee_{name}.jpg"), img)
        print(f"[frisbee] {name}: {len(boxes)} dets, {dt*1000:.0f} ms")

    (OUT_DIR / "det_stats.json").write_text(json.dumps(results, indent=2))
    print("saved ->", OUT_DIR / "det_stats.json")


if __name__ == "__main__":
    main()
