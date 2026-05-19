"""Quick frame review tool for checking auto-generated YOLO labels.

Usage:
    streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import cv2
import streamlit as st

st.set_page_config(page_title="Label Review", layout="wide")


def save_label(txt_path: Path, box: dict | None):
    if box:
        txt_path.write_text(f"0 {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}\n")
    else:
        txt_path.write_text("")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/datasets/game1080/frames")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    frames_dir = Path(args.frames_dir)
    labels_dir = frames_dir.parent / "labels_gsam"
    
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        st.error("No frames found")
        return

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    st.title("Label Review")
    total = len(frames)

    idx = st.session_state.idx
    frame_path = frames[idx]
    fname = frame_path.stem
    txt_path = labels_dir / f"{fname}.txt"

    # Read current label
    has_label = False
    box = None
    if txt_path.exists():
        content = txt_path.read_text().strip()
        if content and content[0] == "0":
            parts = content.split()
            if len(parts) == 5:
                has_label = True
                box = {"cx": float(parts[1]), "cy": float(parts[2]), "w": float(parts[3]), "h": float(parts[4])}

    img = cv2.imread(str(frame_path))
    if img is None:
        st.error("Cannot load image")
        return
    h, w = img.shape[:2]
    if max(w, h) > 1000:
        scale = 1000 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        h, w = img.shape[:2]

    # Draw boxes
    display = img.copy()
    if box:
        x1 = int((box["cx"] - box["w"] / 2) * w)
        y1 = int((box["cy"] - box["h"] / 2) * h)
        x2 = int((box["cx"] + box["w"] / 2) * w)
        y2 = int((box["cy"] + box["h"] / 2) * h)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"GSAM", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(display, channels="BGR", use_container_width=True)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 4])
    with col1:
        if st.button("✅ Accept", type="primary"):
            if has_label:
                st.session_state.idx += 1
                st.rerun()
    with col2:
        if st.button("❌ Reject"):
            save_label(txt_path, None)
            st.session_state.idx += 1
            st.rerun()
    with col3:
        if st.button("⏭ Skip"):
            st.session_state.idx += 1
            st.rerun()
    with col4:
        pass

    st.caption(f"{idx+1}/{total} | {fname}.jpg  |  GSAM label: {'✅' if has_label else '❌ none'}")

    if idx >= total - 1:
        st.success("All frames reviewed!")


if __name__ == "__main__":
    main()
