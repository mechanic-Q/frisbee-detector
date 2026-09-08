"""合成渲染编排：相机 + 场地模型 → (图像, 真值)。"""
import cv2
import numpy as np

from . import camera as cam_mod
from . import raster
from . import field_model as fm


def render_one(rng):
    img = np.zeros((cam_mod.H_IMG, cam_mod.W_IMG, 3), np.uint8)
    raster.draw_grass(img, rng)
    cam = cam_mod.sample_camera(rng)
    K = cam_mod.build_k(rng, cam)
    rvec, tvec = cam_mod.build_rt(cam)

    for (a, b) in fm.line_segments():
        pa, pb = cam_mod.project([a, b], K, rvec, tvec, cam)
        mid = cam_mod.project([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], K, rvec, tvec, cam)[0]
        unit = cam_mod.project([a, (a[0] + 1.0, a[1])], K, rvec, tvec, cam)
        scale = float(np.linalg.norm(unit[1] - unit[0]))  # px / m
        if not (0 <= mid[0] < cam_mod.W_IMG and 0 <= mid[1] < cam_mod.H_IMG):
            continue
        wpx = float(np.clip(rng.uniform(*fm.LINE_WIDTH_M) * scale, 1.2, 14))
        raster.draw_segment(img, pa, pb, wpx, rng)

    for (bx, by) in fm.brick_marks():
        pp = cam_mod.project((bx, by), K, rvec, tvec, cam)[0]
        unit = cam_mod.project([(bx, by), (bx + 1.0, by)], K, rvec, tvec, cam)
        scale = float(np.linalg.norm(unit[1] - unit[0]))
        if 0 <= pp[0] < cam_mod.W_IMG and 0 <= pp[1] < cam_mod.H_IMG:
            cv2.circle(img, tuple(np.round(pp).astype(int)),
                       max(2, int(rng.uniform(0.15, 0.25) * scale)), (235, 235, 235), -1, cv2.LINE_AA)

    raster.add_occluders(img, rng)
    raster.grade(img, rng)

    corners_w = fm.corners()
    corners_px = cam_mod.project(corners_w, K, rvec, tvec, cam)
    Hmat, _ = cv2.findHomography(corners_w, corners_px, 0)
    kp_px = cam_mod.project(fm.keypoints(), K, rvec, tvec, cam)
    kps = [{"x": float(x), "y": float(y), "visible": bool(0 <= x < cam_mod.W_IMG and 0 <= y < cam_mod.H_IMG)}
           for x, y in kp_px]
    meta = {"H": np.asarray(Hmat).tolist(), "keypoints": kps,
            "camera": {"profile": cam["profile"], "h": float(cam["h"]), "cx": float(cam["cx"]),
                       "cy": float(cam["cy"]), "fov": float(cam["fov"]), "roll": float(cam["roll"]),
                       "k1": float(cam["k1"]), "k2": float(cam["k2"]),
                       "look": [float(v) for v in cam["look"]]}}
    return img, meta
