"""自动标定 vs 手动标定的可视化对比（截图 + 叠加框）。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.filters import FIELD_POLYGON_M, point_in_polygon  # noqa: E402
from utils.homography import load_calibration, pixel_to_world, world_to_pixel  # noqa: E402

RAW = Path("runs/gui_analysis/raw300/tracks.json")
AUTO = Path("runs/gui_analysis/raw300/auto_calib.json")
MANUAL = Path("runs/gui_analysis/calib_demo.json")
OUT = Path("runs/gui_analysis/screenshots")

doc = json.loads(RAW.read_text(encoding="utf-8"))
auto = json.loads(AUTO.read_text(encoding="utf-8"))
manual = json.loads(MANUAL.read_text(encoding="utf-8"))
m_auto = np.array(auto["matrix"])
m_man = np.array(manual["matrix"])

cap = cv2.VideoCapture(protocol_win := str(doc["video"]).replace("E:/", "/mnt/e/").replace("E:\\", "/mnt/e/"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 9000)
ok, frame = cap.read()
cap.release()
if not ok:
    # 可能是路径问题，退化用 doc 里 video 的 wsl 路径
    cap = cv2.VideoCapture(protocol.win_to_wsl(doc["video"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 9000)
    ok, frame = cap.read()
assert ok, "cannot read frame"

OUT.mkdir(parents=True, exist_ok=True)


def render(matrix, title, fname, use_polygon=True):
    img = frame.copy()
    # 场地多边形（自动/手动矩阵投影）
    poly_px = [world_to_pixel(matrix, wx, wy) for wx, wy in FIELD_POLYGON_M]
    pts = np.array(poly_px, dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (80, 148, 255))
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.polylines(img, [pts], True, (80, 148, 255), 3)
    # 框按多边形内外着色
    for d in doc["frames"].get("9000", []):
        x1, y1, x2, y2 = map(int, d["bbox"])
        wx, wy = pixel_to_world(matrix, (x1 + x2) / 2, y2)
        inside = point_in_polygon(wx, wy, FIELD_POLYGON_M) if use_polygon else (y2 >= 1080 * 0.30)
        color = (0, 200, 0) if inside else (0, 0, 220)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, title, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
    cv2.putText(img, title, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    cv2.imwrite(str(OUT / fname), img)
    print("saved", fname)


render(m_auto, "AUTO (players-extent)", "06_autocalib_overlay.jpg")
render(m_man, "MANUAL DEMO (eyeballed)", "07_manualcalib_overlay.jpg")
