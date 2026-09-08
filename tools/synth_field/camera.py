"""相机采样与投影（几何层，无渲染依赖）。"""
import cv2
import numpy as np

W_IMG, H_IMG = 1920, 1080


def sample_camera(rng):
    """两种成像档案：三脚架近景(60%) / 高位全景(40%)。返回 dict。"""
    if rng.random() < 0.6:
        profile = "tripod"
        h = rng.uniform(4.0, 12.0)
        cx = rng.uniform(5, 95)
        cy = rng.choice([-1.0, 1.0]) * rng.uniform(8, 25)
        ly = np.clip(abs(cy) * rng.uniform(0.3, 0.9), 2.0, 35.0)
        if cy > 0:
            ly = 37.0 - ly
        look = np.array([np.clip(cx + rng.uniform(-25, 25), 5, 95), ly, 0.0])
        fov = rng.uniform(45, 95)
    else:
        profile = "drone"
        h = rng.uniform(18.0, 35.0)
        cx = rng.uniform(20, 80)
        cy = rng.uniform(-25, 62)
        look = np.array([rng.uniform(30, 70), rng.uniform(8, 29), 0.0])
        fov = rng.uniform(60, 100)
    return {"profile": profile, "h": h, "cx": cx, "cy": cy,
            "look": look, "fov": fov, "roll": rng.uniform(-6, 6),
            "k1": float(rng.uniform(-0.08, 0.08)), "k2": float(rng.uniform(-0.02, 0.02))}


def build_rt(cam):
    """相机外参。OpenCV 相机系: x右 y下 z朝场景内。"""
    C = np.array([cam["cx"], cam["cy"], cam["h"]])
    fwd = cam["look"] - C
    fwd = fwd / np.linalg.norm(fwd)
    up0 = np.array([0, 0, 1.0])
    right = np.cross(fwd, up0)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, -up, fwd], axis=0)
    cr, sr = np.cos(np.deg2rad(cam["roll"])), np.sin(np.deg2rad(cam["roll"]))
    R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]]) @ R
    rvec, _ = cv2.Rodrigues(R)
    return rvec, -R @ C


def build_k(rng, cam):
    f = (W_IMG / 2) / np.tan(np.deg2rad(cam["fov"]) / 2)
    return np.array([[f, 0, W_IMG / 2 + rng.uniform(-40, 40)],
                     [0, f, H_IMG / 2 + rng.uniform(-30, 30)],
                     [0, 0, 1.0]])


def project(pts, K, rvec, tvec, cam, use_distortion=False):
    """世界点(x,y[,z]) → 像素 (N,2)。1-D 单点也可。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.shape[-1] == 2:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])
    dist = np.array([cam["k1"], cam["k2"], 0, 0, 0], dtype=np.float64) if use_distortion \
        else np.zeros(5, dtype=np.float64)
    out, _ = cv2.projectPoints(pts.reshape(-1, 1, 3).astype(np.float32),
                               np.asarray(rvec, dtype=np.float64),
                               np.asarray(tvec, dtype=np.float64),
                               np.asarray(K, dtype=np.float64), dist)
    return out.reshape(-1, 2)


def homography(cam, K, rvec, tvec, corners_px, corners_world):
    H, _ = cv2.findHomography(corners_world, corners_px, 0)
    return H
