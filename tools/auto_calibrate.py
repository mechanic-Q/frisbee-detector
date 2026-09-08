"""E5-v0：场地自动标定可行性验证（免训练档）。
在 WSL 运行: python3 tools/auto_calibrate.py
流程: PySceneDetect 镜头分段 → 每段白线可见度评分选关键帧 → 白线掩码 + HoughLinesP 提线
      → 直线聚类合并 → 矩形四线搜索（两两近似平行+四角凸四边形）→ 单应性 → 场地网格反投叠加
输出: results/calib_scenes.json + results/calib_overlay_*.jpg
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # worktree 根
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
VIDEO = Path("/mnt/e/frisbee-detector/movie/2024城市飞盘俱乐部锦标赛『决赛』【北京大哥 VS 上海沪蛙】高清版3-1.mp4")

FIELD_W, FIELD_H = 100.0, 37.0   # WFDF 米
MAX_LINES = 10


def scene_keyframes(video: Path, interval_s=240.0, topk=10):
    """单机位连续摇拍无硬切 → 按时间等间隔采样，返回白线可见度 topk 帧 [(rank, sec, frame)]"""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps
    samples = []
    sec = 10.0
    while sec < dur - 5:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap.read()
        if ok:
            mask = whitemask(frame)
            ratio, n, _ = line_score(mask)
            samples.append((n, ratio, sec, frame))
        sec += interval_s
    cap.release()
    samples.sort(key=lambda s: (-s[0], -s[1]))
    return [(i, s[2], s[3]) for i, s in enumerate(samples[:topk])]


def whitemask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    mask = ((S < 80) & (V > 165)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    return mask


def line_score(mask):
    """白线可见度评分：白像素占比 + Hough 线段数"""
    ratio = mask.mean() / 255
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=80,
                            minLineLength=mask.shape[1] // 14, maxLineGap=25)
    n = 0 if lines is None else len(lines)
    return ratio, n, lines


def merge_lines(lines, img_shape, angle_tol=4, dist_tol=25):
    """把相近角度+相近截距的线段合并为带权直线（rho, theta, 票数）"""
    if lines is None:
        return []
    segs = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        theta = np.arctan2(y2 - y1, x2 - x1) % np.pi
        segs.append((x1, y1, x2, y2, theta))
    groups = []
    for s in segs:
        placed = False
        for g in groups:
            dth = min(abs(s[4] - g["theta"]), np.pi - abs(s[4] - g["theta"]))
            if dth < np.deg2rad(angle_tol):
                g["segs"].append(s)
                g["theta"] = np.arctan2(np.sin(g["theta"]) + np.sin(s[4]),
                                        np.cos(g["theta"]) + np.cos(s[4]))
                placed = True
                break
        if not placed:
            groups.append({"theta": s[4], "segs": [s]})
    merged = []
    for g in groups:
        pts = []
        for x1, y1, x2, y2, _ in g["segs"]:
            pts += [(x1, y1), (x2, y2)]
        pts = np.array(pts, dtype=float)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean)
        d = vt[0]  # 方向
        n = np.array([-d[1], d[0]])  # 法向
        rho = float(n @ mean)
        merged.append({"rho": rho, "n": n.tolist(), "theta": float(np.arctan2(d[1], d[0])),
                       "votes": len(g["segs"]), "len": float(np.linalg.norm(pts.max(0) - pts.min(0)))})
    merged.sort(key=lambda m: -m["votes"])
    return merged[:MAX_LINES]


def intersections(merged):
    """候选四边形：从直线里挑 4 条（两对近似平行）求四角点"""
    res = []
    m = merged
    n = len(m)
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(n):
                if c in (a, b):
                    continue
                for d in range(c + 1, n):
                    if d in (a, b):
                        continue
                    la, lb, lc, ld = m[a], m[b], m[c], m[d]

                    def angdiff(p, q):
                        v = abs(p["theta"] - q["theta"]) % np.pi
                        return min(v, np.pi - v)
                    if angdiff(la, lb) < np.deg2rad(60):  # ab 需近似平行(长边)
                        continue
                    if angdiff(lc, ld) < np.deg2rad(60):
                        continue
                    def corner(p, q):
                        A = np.array([p["n"], q["n"]])
                        if abs(np.linalg.det(A)) < 1e-6:
                            return None
                        return np.linalg.solve(A, np.array([p["rho"], q["rho"]]))
                    pts = [corner(la, lc), corner(la, ld), corner(lb, lc), corner(lb, ld)]
                    if any(p is None for p in pts):
                        continue
                    P = np.array(pts)
                    h, w = 1080, 1920
                    # 硬约束：四角必须都在画面内（防杂物线假阳性）
                    if not ((P[:, 0] >= 0).all() and (P[:, 0] < w).all()
                            and (P[:, 1] >= 0).all() and (P[:, 1] < h).all()):
                        continue
                    # 面积（shifted shoelace），要够大像块场地
                    x, y = P[:, 0], P[:, 1]
                    area = abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
                    if area < 0.05 * w * h:
                        continue
                    res.append((area, P.tolist(), (a, b, c, d)))
    res.sort(key=lambda r: -r[0])
    return res[:1]


def overlay(img, H):
    """把 10m 网格反投到画面"""
    vis = img.copy()
    step = 10.0
    pts = []
    for gx in np.arange(0, FIELD_W + 1, step):
        for gy in np.arange(0, FIELD_H + 1, step):
            pts.append((gx, gy))
    P = np.array(pts, dtype=np.float64)
    Hp, _ = cv2.findHomography(P.reshape(-1, 1, 2), None) if False else (None, None)
    proj = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H).reshape(-1, 2)
    for (gx, gy), (px, py) in zip(pts, proj):
        if 0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]:
            cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)
    # 场地边框
    corners = np.array([[0, 0], [FIELD_W, 0], [FIELD_W, FIELD_H], [0, FIELD_H]], dtype=np.float64)
    cp = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2).astype(int)
    cv2.polylines(vis, [cp], True, (0, 255, 0), 2)
    return vis


def main():
    kfs = scene_keyframes(VIDEO)
    report = []
    for scene_idx, sec, img in kfs:
        mask = whitemask(img)
        ratio, n, lines = line_score(mask)
        merged = merge_lines(lines, img.shape)
        cand = intersections(merged) if len(merged) >= 4 else []
        entry = {"scene": scene_idx, "t_sec": round(sec, 1),
                 "white_ratio": round(ratio, 4), "hough_segments": n,
                 "merged_lines": len(merged), "solved": bool(cand)}
        if cand:
            area, P, line_ids = cand[0]
            src = np.array(P, dtype=np.float64)
            dst = np.array([[0, 0], [0, FIELD_H], [FIELD_W, FIELD_H], [FIELD_W, 0]], dtype=np.float64)
            H, _ = cv2.findHomography(src, dst, 0)
            if H is not None:
                vis = overlay(img, H)
                cv2.imwrite(str(OUT / f"calib_overlay_scene{scene_idx}.jpg"), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                entry["corner_area_frac"] = round(area / (1920 * 1080), 3)
        report.append(entry)
        print(f"scene {scene_idx:2d} t={sec:6.1f}s white={ratio:.3f} lines={len(merged):2d} solved={entry['solved']}", flush=True)

    solved = sum(1 for r in report if r["solved"])
    summary = {"scenes_checked": len(report), "solved": solved,
               "solve_rate": round(solved / max(len(report), 1), 3)}
    (OUT / "calib_scenes.json").write_text(json.dumps({"summary": summary, "scenes": report}, indent=2))
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
