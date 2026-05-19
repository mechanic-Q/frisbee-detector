"""Shared utilities for label review tools.

Pure functions for label I/O, IoU computation, P2 cache, and review state.
No Streamlit dependency — safe to unit test.
"""

import json
from pathlib import Path


def read_label(txt_path: Path) -> dict | None:
    """Read first YOLO bbox from a label file. Returns None if empty/missing."""
    if not txt_path.exists():
        return None
    content = txt_path.read_text().strip()
    if not content:
        return None
    for line in content.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0].isdigit():
            return {
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            }
    return None


def write_label(txt_path: Path, box: dict | None) -> None:
    """Write a YOLO label. box=None writes empty (negative sample)."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if box:
        txt_path.write_text(f"0 {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}\n")
    else:
        txt_path.write_text("")


def compute_iou(cx1, cy1, w1, h1, cx2, cy2, w2, h2) -> float:
    """Compute IoU between two YOLO-format bboxes (normalized cx,cy,w,h)."""
    x1 = max(cx1 - w1 / 2, cx2 - w2 / 2)
    y1 = max(cy1 - h1 / 2, cy2 - h2 / 2)
    x2 = min(cx1 + w1 / 2, cx2 + w2 / 2)
    y2 = min(cy1 + h1 / 2, cy2 + h2 / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def load_p2_cache(cache_path: Path) -> dict:
    """Load P2 inference cache. Returns {} if not found."""
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_p2_cache(cache_path: Path, data: dict) -> None:
    """Save P2 inference cache to JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2))


def run_p2_inference(model_path: str, frames: list[Path], cache_path: Path) -> dict:
    """Run P2 model on frames, cache results. Returns {stem: {cx,cy,w,h,conf}}."""
    cached = load_p2_cache(cache_path)
    if cached:
        return cached

    from ultralytics import YOLO

    model = YOLO(model_path)
    results = {}
    for i, f in enumerate(frames):
        r = model.predict(str(f), conf=0.25, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            best = r.boxes[0]
            results[f.stem] = {
                "cx": float(best.xywhn[0][0]),
                "cy": float(best.xywhn[0][1]),
                "w": float(best.xywhn[0][2]),
                "h": float(best.xywhn[0][3]),
                "conf": float(best.conf[0]),
            }
        if (i + 1) % 20 == 0:
            print(f"  P2 inference: {i + 1}/{len(frames)}")

    save_p2_cache(cache_path, results)
    return results


def load_review_state(state_path: Path) -> dict:
    """Load review result JSON. Returns empty structure if not found."""
    if not state_path.exists():
        return {"accept": [], "reject": [], "skip": []}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"accept": [], "reject": [], "skip": []}


def save_review_state(state_path: Path, state: dict) -> None:
    """Save review result JSON."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


def sort_frames(frames: list[Path], p2_data: dict, source_boxes: dict,
                sort_mode: str) -> list[Path]:
    """Sort frames by chosen mode.

    sort_mode: "filename", "iou_desc", or "bbox_size"
    """
    if sort_mode == "filename":
        return sorted(frames)

    def score(f):
        stem = f.stem
        src = source_boxes.get(stem)
        p2 = p2_data.get(stem)
        if sort_mode == "iou_desc":
            if src and p2:
                return -compute_iou(
                    src["cx"], src["cy"], src["w"], src["h"],
                    p2["cx"], p2["cy"], p2["w"], p2["h"],
                )
            if src:
                return 1
            if p2:
                return 2 - p2["conf"]
            return float('inf')
        if sort_mode == "bbox_size":
            if src:
                return -(src["w"] * src["h"])
            return float('inf')
        return float('inf')

    return sorted(frames, key=score)
