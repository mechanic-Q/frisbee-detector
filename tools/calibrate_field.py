"""Field calibration tool using Streamlit.

Click field line intersections on a video frame, enter real-world coordinates,
compute homography, verify with overlay and bird's-eye view.

Usage:
    streamlit run tools/calibrate_field.py -- --video movie/25866279684-1-192.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from utils.homography import (
    compute_homography,
    draw_field_overlay,
    load_calibration,
    pixel_to_world,
    save_calibration,
    warp_to_birdseye,
    world_to_pixel,
)

st.set_page_config(page_title="Field Calibration", layout="wide")

FIELD_W, FIELD_H = 100, 37


def get_args():
    parser = argparse.ArgumentParser(description="Field calibration tool")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to use for calibration (default: 0)",
    )
    args, _ = parser.parse_known_args()
    return args


@st.cache_data
def extract_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    args = get_args()
    video_path = Path(args.video)
    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
        st.info("Usage: streamlit run tools/calibrate_field.py -- --video <path>")
        return

    st.title("Field Calibration Tool")
    st.caption(f"Video: {video_path.name}")

    frame = extract_frame(str(video_path), args.frame)
    if frame is None:
        st.error("Failed to extract frame from video")
        return

    h, w = frame.shape[:2]
    st.caption(f"Frame: {w}x{h} | Field: 100m×37m | Origin: bottom-left (0,0)")

    if "points" not in st.session_state:
        st.session_state.points = []

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("Calibration Frame")
        st.caption("Click on the image to place a point, then enter World X/Y and click 'Add Point'")

        annotated = frame.copy()
        for i, pt in enumerate(st.session_state.points):
            px, py = int(pt["pixel"][0]), int(pt["pixel"][1])
            cv2.circle(annotated, (px, py), 8, (0, 255, 0), 2)
            cv2.putText(annotated, str(i + 1), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode()

        clickable_html = f"""
        <div style="position:relative;display:inline-block;width:100%">
          <img id="calimg" src="data:image/jpeg;base64,{img_b64}"
               style="width:100%;cursor:crosshair;border:1px solid #ccc;border-radius:4px"
               onclick="handleClick(event)">
          <div id="coord-display"
               style="margin-top:6px;font-size:15px;color:#1976d2;font-weight:500;">
            ↔ Click on a field-line intersection
          </div>
        </div>
        <script>
          let lastX = -1, lastY = -1;
          function handleClick(e) {{
            const img = document.getElementById('calimg');
            const rect = img.getBoundingClientRect();
            const scaleX = img.naturalWidth / rect.width;
            const scaleY = img.naturalHeight / rect.height;
            lastX = Math.round((e.clientX - rect.left) * scaleX);
            lastY = Math.round((e.clientY - rect.top) * scaleY);
            document.getElementById('coord-display').innerHTML =
              '📍 Selected: (' + lastX + ', ' + lastY +
              ')  <button onclick="sendCoords()" style="padding:2px 12px;cursor:pointer">↳ Fill inputs</button>';
          }}
          function sendCoords() {{
            if (lastX < 0) return;
            Streamlit.setComponentValue(JSON.stringify({{x: lastX, y: lastY}}));
          }}
        </script>
        """
        click_data = components.html(clickable_html, height=520)

        if click_data is not None:
            import json
            try:
                data = json.loads(click_data) if isinstance(click_data, str) else click_data
                st.session_state["click_px"] = int(data["x"])
                st.session_state["click_py"] = int(data["y"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        st.subheader("Add Control Point")
        input_cols = st.columns([1, 1, 1, 1, 1])
        with input_cols[0]:
            px_val = st.session_state.get("click_px", w // 2)
            px_in = st.number_input("Pixel X", min_value=0, max_value=w, value=px_val, key="px")
        with input_cols[1]:
            py_val = st.session_state.get("click_py", h // 2)
            py_in = st.number_input("Pixel Y", min_value=0, max_value=h, value=py_val, key="py")
        with input_cols[2]:
            wx_in = st.number_input("World X (m) →", min_value=0.0, max_value=float(FIELD_W), value=0.0, step=1.0, key="wx")
        with input_cols[3]:
            wy_in = st.number_input("World Y (m) ↑", min_value=0.0, max_value=float(FIELD_H), value=0.0, step=1.0, key="wy")
        with input_cols[4]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add Point", type="primary"):
                st.session_state.points.append({
                    "pixel": [float(px_in), float(py_in)],
                    "world": [float(wx_in), float(wy_in)],
                })
                st.rerun()

        btn_cols = st.columns([1, 1, 1, 1])
        with btn_cols[0]:
            if st.button("Undo Last"):
                if st.session_state.points:
                    st.session_state.points.pop()
                    st.rerun()
        with btn_cols[1]:
            if st.button("Clear All"):
                st.session_state.points = []
                st.rerun()
        with btn_cols[2]:
            compute_clicked = st.button("Compute Homography", type="primary")
        with btn_cols[3]:
            save_clicked = st.button("Save Calibration")

    with col_right:
        st.subheader("Points")
        if not st.session_state.points:
            st.info("Click image → enter World X/Y → Add Point")
        else:
            for i, pt in enumerate(st.session_state.points):
                st.text(f"#{i+1}: px=({pt['pixel'][0]:.0f},{pt['pixel'][1]:.0f}) "
                        f"→ w=({pt['world'][0]:.1f},{pt['world'][1]:.1f})")

        st.subheader("Field Coordinate Guide")
        st.markdown("""
```
  (0,37)┌─────────────┬──────────────┐(100,37)
        │             │              │
        │   Endzone   │    Field     │
   ←Y↑  │  (0,37)     │   (50,18.5)  │
        │     to      │    center    │
  (0,0) │  (0,18.5)   │              │
        ├─────────────┼──────────────┤
        │             │              │
        │   Field     │   Endzone    │
        │  (50,18.5)  │   (100,37)  │
        │   center    │     to       │
  (0,0) └─────────────┴──────────────┘(100,0)
      X→ (right = 100m)
```
**Common points:**
- Bottom-left corner: (0, 0)
- Bottom-right corner: (100, 0)
- Top-left corner: (0, 37)
- Top-right corner: (100, 37)
- Center line midpoint: (50, 18.5)
""")
        st.caption("Field: 100m × 37m. Origin = bottom-left.")

    if compute_clicked:
        pts = st.session_state.points
        if len(pts) < 4:
            st.error("Need at least 4 points to compute homography")
            return

        raw_points = [
            (p["pixel"][0], p["pixel"][1], p["world"][0], p["world"][1])
            for p in pts
        ]
        matrix, error = compute_homography(raw_points)

        if matrix is None:
            st.error("Homography computation failed. Points may be nearly collinear.")
            return

        st.session_state.matrix = matrix
        st.session_state.reproj_error = error

        quality = "Excellent" if error < 3 else ("Acceptable" if error < 8 else "Poor — consider recalibrating")
        st.success(f"Homography computed — RMSE: {error:.2f}px ({quality})")

        st.subheader("Homography Matrix")
        matrix_display = np.array2string(matrix, precision=4, suppress_small=True)
        st.code(matrix_display, language="text")

        _, obuf = cv2.imencode(".jpg", cv2.cvtColor(draw_field_overlay(frame, matrix), cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
        overlay_b64 = base64.b64encode(obuf).decode()
        _, bbuf = cv2.imencode(".jpg", cv2.cvtColor(warp_to_birdseye(frame, matrix), cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
        birdseye_b64 = base64.b64encode(bbuf).decode()

        vis_cols = st.columns(2)
        with vis_cols[0]:
            st.subheader("Field Line Overlay")
            components.html(f'<img src="data:image/jpeg;base64,{overlay_b64}" style="width:100%">', height=400)
        with vis_cols[1]:
            st.subheader("Bird's-Eye View")
            components.html(f'<img src="data:image/jpeg;base64,{birdseye_b64}" style="width:100%">', height=300)

        st.subheader("Per-Point Errors")
        for i, pt in enumerate(pts):
            px, py = pt["pixel"][0], pt["pixel"][1]
            rwx, rwy = pixel_to_world(matrix, px, py)
            err = ((rwx - pt["world"][0]) ** 2 + (rwy - pt["world"][1]) ** 2) ** 0.5
            st.text(f"#{i+1}: projected=({rwx:.2f},{rwy:.2f}) expected=({pt['world'][0]:.1f},{pt['world'][1]:.1f}) err={err:.2f}m")

    if save_clicked:
        if "matrix" not in st.session_state:
            st.error("Compute homography first")
            return

        output_dir = Path("configs/homography")
        output_path = output_dir / f"{video_path.stem}.json"

        point_errors = []
        matrix = st.session_state.matrix
        for pt in st.session_state.points:
            px, py = pt["pixel"][0], pt["pixel"][1]
            rwx, rwy = pixel_to_world(matrix, px, py)
            err_px = ((rwx - pt["world"][0]) ** 2 + (rwy - pt["world"][1]) ** 2) ** 0.5
            point_errors.append(round(err_px, 4))

        points_with_errors = []
        for pt, err in zip(st.session_state.points, point_errors):
            p = dict(pt)
            p["error_px"] = err
            points_with_errors.append(p)

        save_calibration(
            path=output_path,
            video=video_path.name,
            image_size=[w, h],
            field_size_m=[FIELD_W, FIELD_H],
            calibration_frame=args.frame,
            points=points_with_errors,
            matrix=matrix,
            reprojection_error_px=round(st.session_state.reproj_error, 4),
        )
        st.success(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
