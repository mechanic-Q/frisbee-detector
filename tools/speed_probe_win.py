"""Windows 原生 vs WSL 训练速度对照：frisbee_merged_v2 上 3 epochs（同配方缩小版）。
Windows 侧运行: python tools/speed_probe_win.py
输出: results/speed_probe_win.json
"""
import json
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

t0 = time.time()
m = YOLO("yolov8s.pt")
r = m.train(
    data=str(ROOT / "configs/frisbee_merged_v2.yaml"),
    epochs=3, patience=3, box=5, close_mosaic=0,
    imgsz=1280, batch=2, workers=4,
    cache="ram",
    name="speed_probe_win",
    project=str(OUT / "speed_probe_win_run"),
    exist_ok=True, plots=False,
)
dt = time.time() - t0
res = {
    "platform": "windows-native",
    "epochs": 3,
    "total_min": round(dt / 60, 1),
    "min_per_epoch": round(dt / 60 / 3, 2),
    "workers": 4, "cache": "ram", "batch": 2, "imgsz": 1280,
    "val_map50": round(float(r.box.map50), 4) if r and r.box else None,
}
(OUT / "speed_probe_win.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
