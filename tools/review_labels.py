"""Universal frame label review tool for YOLO format.

Accept/Reject auto-generated labels (GSAM, pseudo, etc.) with optional
P2 model overlay. Manual labeling uses yololabeler directly.

Usage:
    # Review auto-labels (backward compatible with old review_labels.py)
    streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames

    # Review with P2 model overlay
    streamlit run tools/review_labels.py -- \
        --frames-dir data/datasets/game1080/frames \
        --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt

    # Force rebuild P2 cache
    streamlit run tools/review_labels.py -- \
        --frames-dir data/datasets/game1080/frames \
        --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt \
        --rebuild-cache
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import cv2
import streamlit as st

from _label_utils import (
    read_label,
    write_label,
    compute_iou,
    run_p2_inference,
    load_review_state,
    save_review_state,
    sort_frames,
)

st.set_page_config(page_title="Label Review", layout="wide")


def get_args():
    parser = argparse.ArgumentParser(description="Universal frame label reviewer")
    parser.add_argument("--frames-dir", default="data/datasets/game1080/frames")
    parser.add_argument("--labels-dir", default=None,
                        help="Source labels dir (auto-detected: labels_gsam, then labels)")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir for reviewed labels (default: <parent>/labels)")
    parser.add_argument("--model", default=None,
                        help="YOLO model for overlay comparison (optional)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for model overlay")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild P2 inference cache")
    return parser.parse_known_args()[0]


def auto_detect_dirs(frames_dir: Path, args):
    """Resolve labels_dir and output_dir based on frames_dir and args."""
    parent = frames_dir.parent

    if args.labels_dir:
        labels_dir = Path(args.labels_dir)
    else:
        candidates = [parent / "labels_gsam", parent / "labels"]
        labels_dir = next((d for d in candidates if d.exists()), parent / "labels")

    output_dir = Path(args.output_dir) if args.output_dir else parent / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    return labels_dir, output_dir


def load_frames(frames_dir: Path, labels_dir: Path) -> list[Path]:
    """Load frames that have source labels. Falls back to all frames if none labeled."""
    all_frames = sorted(frames_dir.glob("*.jpg"))
    if not all_frames:
        return []

    frames = []
    for f in all_frames:
        txt = labels_dir / f"{f.stem}.txt"
        if txt.exists():
            content = txt.read_text().strip()
            if content and content[0].isdigit():
                frames.append(f)
    return frames if frames else all_frames


def load_p2_data(model_path: str | None, frames: list[Path],
                 frames_dir: Path, rebuild: bool, conf: float = 0.25) -> dict:
    """Load or compute P2 inference results."""
    if not model_path:
        return {}

    cache_path = frames_dir.parent / "p2_cache.json"
    if rebuild and cache_path.exists():
        cache_path.unlink()

    return run_p2_inference(model_path, frames, cache_path, conf)


def draw_bbox(img, box, color, label, h, w):
    """Draw a single YOLO bbox on image."""
    x1 = int((box["cx"] - box["w"] / 2) * w)
    y1 = int((box["cy"] - box["h"] / 2) * h)
    x2 = int((box["cx"] + box["w"] / 2) * w)
    y2 = int((box["cy"] + box["h"] / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def render_frame(frame_path: Path,
                 source_box: dict | None, p2_box: dict | None,
                 p2_conf: float | None, iou: float | None):
    """Load image, draw bboxes, return display image."""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None, 0, 0
    h, w = img.shape[:2]
    if max(w, h) > 1200:
        scale = 1200 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        h, w = img.shape[:2]

    display = img.copy()

    if source_box:
        draw_bbox(display, source_box, (0, 255, 0), "SRC", h, w)

    if p2_box:
        label = f"P2 {p2_conf:.2f}"
        if iou is not None:
            label += f" IoU={iou:.2f}"
        draw_bbox(display, p2_box, (255, 100, 0), label, h, w)

    return display, h, w


def get_frame_info(stem, source_box, p2_data):
    """Compute P2 overlay info for a frame."""
    p2_entry = p2_data.get(stem)
    p2_box = None
    p2_conf = None
    iou = None
    if p2_entry:
        p2_box = {k: p2_entry[k] for k in ("cx", "cy", "w", "h")}
        p2_conf = p2_entry["conf"]
        if source_box:
            iou = compute_iou(
                source_box["cx"], source_box["cy"], source_box["w"], source_box["h"],
                p2_box["cx"], p2_box["cy"], p2_box["w"], p2_box["h"],
            )
    return p2_box, p2_conf, iou


def main():
    args = get_args()
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        st.error(f"Frames directory not found: {frames_dir}")
        return

    labels_dir, output_dir = auto_detect_dirs(frames_dir, args)

    frames = load_frames(frames_dir, labels_dir)
    if not frames:
        st.error("No frames found")
        return

    source_boxes = {}
    for f in frames:
        txt = labels_dir / f"{f.stem}.txt"
        source_boxes[f.stem] = read_label(txt)

    p2_data = {}
    if args.model:
        with st.spinner("Running P2 inference (cached after first run)..."):
            p2_data = load_p2_data(args.model, frames, frames_dir,
                                   args.rebuild_cache, args.conf)

    # review_result.json is saved at <frames_parent>/review_result.json
    # alongside frames/ and labels/
    state_path = output_dir.parent / "review_result.json"
    review_state = load_review_state(state_path)

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        sort_mode = st.selectbox("Sort by", [
            ("File name", "filename"),
            ("P2 IoU (high first)", "iou_desc"),
            ("BBox size (large first)", "bbox_size"),
        ], format_func=lambda x: x[0], index=0)[1]

        n_accept = len(review_state.get("accept", []))
        n_reject = len(review_state.get("reject", []))
        n_skip = len(review_state.get("skip", []))
        st.metric("Accept", n_accept)
        st.metric("Reject", n_reject)
        st.metric("Skip", n_skip)
        st.metric("Remaining", len(frames) - n_accept - n_reject - n_skip)

        if st.button("Export Results"):
            save_review_state(state_path, review_state)
            st.success(f"Saved to {state_path}")

    frames = sort_frames(frames, p2_data, source_boxes, sort_mode)
    total = len(frames)

    # Skip already-reviewed frames
    reviewed = set(review_state.get("accept", [])
                   + review_state.get("reject", [])
                   + review_state.get("skip", []))
    unreviewed = [f for f in frames if f.stem not in reviewed]
    if unreviewed:
        current_idx = frames.index(unreviewed[0])
    else:
        current_idx = total - 1

    if st.session_state.idx >= total:
        st.session_state.idx = current_idx

    idx = min(st.session_state.idx, total - 1)
    frame_path = frames[idx]
    stem = frame_path.stem
    source_box = source_boxes.get(stem)

    p2_box, p2_conf, iou = get_frame_info(stem, source_box, p2_data)

    display, _, _ = render_frame(frame_path, source_box, p2_box, p2_conf, iou)
    if display is None:
        st.error(f"Cannot load image: {frame_path}")
        return

    st.image(display, channels="BGR", use_container_width=True)

    # --- Buttons ---
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    with col1:
        if st.button("Accept", type="primary"):
            out_txt = output_dir / f"{stem}.txt"
            if p2_box and iou is not None and iou > 0.3:
                write_label(out_txt, p2_box)
            elif source_box:
                write_label(out_txt, source_box)
            review_state.setdefault("accept", []).append(stem)
            save_review_state(state_path, review_state)
            st.session_state.idx = idx + 1
            st.rerun()
    with col2:
        if st.button("Reject"):
            out_txt = output_dir / f"{stem}.txt"
            write_label(out_txt, None)
            review_state.setdefault("reject", []).append(stem)
            save_review_state(state_path, review_state)
            st.session_state.idx = idx + 1
            st.rerun()
    with col3:
        if st.button("Skip"):
            review_state.setdefault("skip", []).append(stem)
            save_review_state(state_path, review_state)
            st.session_state.idx = idx + 1
            st.rerun()
    with col4:
        st.empty()
    with col5:
        if st.button("<< Prev"):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

    # --- Info bar ---
    src_status = "has label" if source_box else "no label"
    p2_status = f"P2 conf={p2_conf:.2f}" if p2_conf else "P2: no detection"
    iou_status = f"IoU={iou:.2f}" if iou is not None else ""
    st.caption(
        f"{idx + 1}/{total} | {stem}.jpg | SRC: {src_status} | {p2_status} {iou_status} | "
        f"src: {labels_dir.name} -> out: {output_dir.name}"
    )

    if idx >= total - 1:
        st.success(
            f"All frames reviewed! Accept: {n_accept}, Reject: {n_reject}, Skip: {n_skip}"
        )


if __name__ == "__main__":
    main()
