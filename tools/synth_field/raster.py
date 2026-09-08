"""光栅化层：把投影后的几何画到像素图上（草地/白线/遮挡物/后期），不依赖相机模型。"""
import cv2
import numpy as np

W_IMG, H_IMG = 1920, 1080


def draw_grass(img, rng):
    base = np.array([70 + rng.integers(-12, 12), 105 + rng.integers(-15, 15), 60 + rng.integers(-12, 12)])
    small = rng.normal(0, 14, (H_IMG // 4, W_IMG // 4, 3)).astype(np.float32)
    small = cv2.resize(small, (W_IMG, H_IMG), interpolation=cv2.INTER_CUBIC)
    large = rng.normal(0, 22, (H_IMG // 32, W_IMG // 32, 3)).astype(np.float32)
    large = cv2.resize(large, (W_IMG, H_IMG), interpolation=cv2.INTER_CUBIC)
    img[:] = np.clip(base + small + large, 0, 255).astype(np.uint8)
    stripes = (np.sin(np.linspace(0, rng.uniform(20, 60), H_IMG)) * 6)[:, None, None]
    img[:] = np.clip(img.astype(np.float32) + stripes, 0, 255).astype(np.uint8)


def draw_segment(img, p1, p2, width_px, rng):
    n = max(int(np.hypot(*(p2 - p1)) / 6), 2)
    ts = np.linspace(0, 1, n)
    for i in range(n - 1):
        a = p1 + (p2 - p1) * ts[i]
        b = p1 + (p2 - p1) * ts[i + 1]
        wpx = max(1, int(width_px * (0.8 + 0.4 * rng.random())))
        col = int(255 - rng.integers(0, 25))
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                 (col, col, col), wpx, cv2.LINE_AA)


def add_occluders(img, rng, n_lo=2, n_hi=14):
    for _ in range(rng.integers(n_lo, n_hi + 1)):
        cx, cy = rng.uniform(0, W_IMG), rng.uniform(H_IMG * 0.25, H_IMG)
        hpx = rng.uniform(30, 220)
        wpx = hpx * rng.uniform(0.25, 0.4)
        col = tuple(int(c) for c in rng.choice(
            [(40, 40, 150), (150, 60, 40), (30, 30, 30), (90, 90, 90)]))
        cv2.ellipse(img, (int(cx), int(cy)), (int(wpx / 2), int(hpx / 2)),
                    float(rng.uniform(-20, 20)), 0, 360, col, -1)


def grade(img, rng):
    if rng.random() < 0.7:
        img[:] = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.6))
    img[:] = np.clip(img.astype(np.float32) * rng.uniform(0.72, 1.25)
                     + rng.normal(0, rng.uniform(2, 9), img.shape), 0, 255).astype(np.uint8)
