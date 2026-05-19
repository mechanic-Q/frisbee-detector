"""Quick frame labeling tool for YOLO format.

Usage:
    streamlit run tools/label_frames.py -- --frames-dir data/datasets/game1080/frames
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import base64
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/datasets/game1080/frames")
    args, _ = parser.parse_known_args()
    return args


def load_labels(txt_path: Path) -> list[dict]:
    if not txt_path.exists():
        return []
    labels = []
    for line in txt_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 5:
            labels.append({
                "cls": int(parts[0]),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
    return labels


def save_labels(txt_path: Path, labels: list[dict]) -> None:
    lines = []
    for lbl in labels:
        lines.append(f"{lbl['cls']} {lbl['cx']:.6f} {lbl['cy']:.6f} {lbl['w']:.6f} {lbl['h']:.6f}")
    txt_path.write_text("\n".join(lines))


def main():
    args = get_args()
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        st.error(f"Directory not found: {frames_dir}")
        return

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        st.error("No .jpg files found")
        return

    labels_dir = frames_dir.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    st.set_page_config(page_title="Frame Labeler", layout="wide")
    st.title("Frame Labeler")
    st.caption(f"Frames: {len(frames)}")

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "boxes" not in st.session_state:
        st.session_state.boxes = []

    idx = st.session_state.idx
    frame_path = frames[idx]

    img = cv2.imread(str(frame_path))
    if img is None:
        st.error(f"Cannot read {frame_path}")
        return
    h, w = img.shape[:2]
    if max(w, h) > 1000:
        scale = 1000 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        h, w = img.shape[:2]

    # Load existing labels
    txt_path = labels_dir / frame_path.with_suffix(".txt").name
    existing = load_labels(txt_path)
    if existing and not st.session_state.boxes:
        st.session_state.boxes = existing

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(frame_path.name)

        # Draw boxes on image
        display = img.copy()
        for b in st.session_state.boxes:
            x1 = int((b["cx"] - b["w"] / 2) * w)
            y1 = int((b["cy"] - b["h"] / 2) * h)
            x2 = int((b["cx"] + b["w"] / 2) * w)
            y2 = int((b["cy"] + b["h"] / 2) * h)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"#{b['cls']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode()

        html = f"""
        <div style="position:relative;width:100%">
          <img id="labelimg" src="data:image/jpeg;base64,{img_b64}"
               style="width:100%;cursor:crosshair"
               onclick="handleClick(event)">
          <div id="info" style="margin-top:4px;font-size:14px;color:#1976d2;"></div>
        </div>
        <script>
          let startX=0, startY=0, drawing=false;
          function handleClick(e) {{
            const img = document.getElementById('labelimg');
            const rect = img.getBoundingClientRect();
            const sx = img.naturalWidth / rect.width;
            const sy = img.naturalHeight / rect.height;
            const x = (e.clientX - rect.left) * sx;
            const y = (e.clientY - rect.top) * sy;
            document.getElementById('info').innerHTML =
              'Clicked: (' + Math.round(x) + ', ' + Math.round(y) + ') - ' +
              'cx=' + (x/{w}).toFixed(4) + ' cy=' + (y/{h}).toFixed(4);
            window.parent.postMessage({{
              type: 'streamlit:setComponentValue',
              value: JSON.stringify({{x: x/{w}, y: y/{h}, w: {w}, h: {h}}})
            }}, '*');
          }}
        </script>
        """
        click_data = components.html(html, height=600)

        if click_data:
            import json
            try:
                data = json.loads(click_data) if isinstance(click_data, str) else click_data
                cx = float(data["x"])
                cy = float(data["y"])
                fw = float(data["w"])
                fh = float(data["h"])
                bw = st.session_state.get("box_w", 0.05)
                bh = st.session_state.get("box_h", 0.05)
                st.session_state.boxes.append({
                    "cls": 0,  # frisbee
                    "cx": round(cx, 6),
                    "cy": round(cy, 6),
                    "w": round(bw, 6),
                    "h": round(bh, 6),
                })
                st.rerun()
            except Exception:
                pass

    with col2:
        st.subheader("Box Size")
        st.caption("W/H as fraction of image (0-1)")
        bw = st.number_input("Width", 0.001, 0.5, 0.05, 0.005, key="box_w")
        bh = st.number_input("Height", 0.001, 0.5, 0.05, 0.005, key="box_h")

        st.subheader("Current Labels")
        for i, b in enumerate(st.session_state.boxes):
            st.text(f"#{b['cls']}: cx={b['cx']:.3f} cy={b['cy']:.3f} w={b['w']:.3f} h={b['h']:.3f}")

        if st.button("Undo Last"):
            if st.session_state.boxes:
                st.session_state.boxes.pop()
                st.rerun()

        if st.button("Save & Next", type="primary"):
            save_labels(txt_path, st.session_state.boxes)
            st.session_state.boxes = []
            if idx < len(frames) - 1:
                st.session_state.idx = idx + 1
                st.rerun()

        if st.button("Skip (no frisbee)"):
            save_labels(txt_path, [])
            st.session_state.boxes = []
            if idx < len(frames) - 1:
                st.session_state.idx = idx + 1
                st.rerun()

        st.caption(f"Frame {idx+1}/{len(frames)}")
        st.caption(f"Labeled: {len(list(labels_dir.glob('*.txt')))}")
