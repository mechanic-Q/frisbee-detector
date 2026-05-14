"""Web-based detection review tool using Streamlit.

Browse detection frames and mark each as TP (true positive) or FP (false positive).

Usage:
  streamlit run tools/review_web.py -- --img-dir data/fp_spotcheck_50/

Controls:
  - Press Y or click [TP]  → marks as True Positive
  - Press N or click [FP]  → marks as False Positive
  - Press ← → to navigate between images
  - Progress and export saved to results.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import os
import streamlit as st

st.set_page_config(page_title="FP Spot-Check", layout="wide")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="data/fp_spotcheck_50")
    args, _ = parser.parse_known_args()
    return args

def load_results(results_path):
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            for row in csv.DictReader(f):
                results[row["filename"]] = row["label"]
    return results

def save_results(results, results_path):
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label"])
        w.writeheader()
        for fname, label in sorted(results.items()):
            w.writerow({"filename": fname, "label": label})

def main():
    args = get_args()
    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        st.error(f"Directory not found: {img_dir}")
        st.info("Usage: streamlit run tools/review_web.py -- --img-dir path/to/images")
        return

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not images:
        st.error("No images found")
        return

    results_path = img_dir / "review_results.csv"
    results = load_results(results_path)

    st.title("Detection Review — FP Spot-Check")
    st.caption(f"Directory: {img_dir.resolve()} | {len(results)}/{len(images)} reviewed")

    # Stats header
    tp = sum(1 for v in results.values() if v == "TP")
    fp = sum(1 for v in results.values() if v == "FP")
    remaining = len(images) - len(results)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(images))
    col2.metric("TP", tp, delta_color="inverse")
    col3.metric("FP", fp, delta_color="inverse")
    col4.metric("Remaining", remaining)

    # Image selector
    idx = 0
    reviewed = [i for i, img in enumerate(images) if img.name in results]
    unreviewed = [i for i, img in enumerate(images) if img.name not in results]
    if unreviewed:
        idx = unreviewed[0]
    elif reviewed:
        idx = reviewed[-1]

    # Navigation
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
    with nav_col1:
        if st.button("◀ Prev", use_container_width=True):
            idx = max(0, idx - 1)
    with nav_col2:
        if st.button("Next ▶", use_container_width=True):
            idx = min(len(images) - 1, idx + 1)
    with nav_col3:
        st.caption(f"Image {idx + 1}/{len(images)}: {images[idx].name}")
    with nav_col4:
        if st.button("Save Results", type="primary", use_container_width=True):
            save_results(results, results_path)
            st.success(f"Saved to {results_path}")
    with nav_col5:
        if st.button("Unreviewed", use_container_width=True):
            if unreviewed:
                idx = unreviewed[0]

    # Show current image
    img = images[idx]
    current_label = results.get(img.name, "Unreviewed")
    st.image(str(img), use_container_width=True)

    st.caption(f"Current: {img.name} — {current_label}")

    # TP/FP buttons
    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 6])
    with btn_col1:
        if st.button("✅ TP (Y)", use_container_width=True, type="primary"):
            results[img.name] = "TP"
            save_results(results, results_path)
            st.rerun()
    with btn_col2:
        if st.button("❌ FP (N)", use_container_width=True):
            results[img.name] = "FP"
            save_results(results, results_path)
            st.rerun()
    with btn_col3:
        if st.button("Skip", use_container_width=True):
            idx = min(len(images) - 1, idx + 1)

    # Keyboard shortcuts
    st.markdown("""
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    </style>
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.key === 'y' || e.key === 'Y') {
            document.querySelector('button:contains("TP")').click();
        } else if (e.key === 'n' || e.key === 'N') {
            document.querySelector('button:contains("FP")').click();
        }
    });
    </script>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
