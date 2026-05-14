"""Per-box detection review tool using Streamlit.

Show one cropped detection at a time, mark as TP (true positive) or FP (false positive).

Usage:
  streamlit run tools/review_web.py -- --crop-dir /tmp/perbox_crops

Controls:
  - Y / click TP → mark as True Positive (this IS a frisbee), auto-advance
  - N / click FP → mark as False Positive (this is NOT a frisbee), auto-advance
  - ← → navigate manually
  - Results auto-saved to results.csv in the crop directory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import streamlit as st

st.set_page_config(page_title="Per-Box Detection Review", layout="centered")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-dir", default="data/perbox_crops")
    args, _ = parser.parse_known_args()
    return args

@st.cache_data
def load_images(crop_dir):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in Path(crop_dir).iterdir() if p.suffix.lower() in exts])

def load_results(results_path):
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            for row in csv.DictReader(f):
                results[row["filename"]] = row["result"]
                results[f"{row['filename']}_frame"] = row.get("frame", "")
                results[f"{row['filename']}_conf"] = row.get("conf", "")
    return results

def save_results(results, results_path):
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "result", "frame", "conf"])
        w.writeheader()
        reviewed = {k for k in results if not k.endswith("_frame") and not k.endswith("_conf")}
        for fname in sorted(reviewed):
            img_path = crop_dir / fname
            w.writerow({
                "filename": fname,
                "result": results.get(fname, ""),
                "frame": results.get(f"{fname}_frame", ""),
                "conf": results.get(f"{fname}_conf", ""),
            })

def main():
    args = get_args()
    global crop_dir
    crop_dir = Path(args.crop_dir)
    if not crop_dir.exists():
        st.error(f"Directory not found: {crop_dir}")
        st.info("Usage: streamlit run tools/review_web.py -- --crop-dir /tmp/perbox_crops")
        return

    images = load_images(str(crop_dir))
    if not images:
        st.error("No images found")
        return

    results_path = crop_dir / "review_results.csv"
    results = load_results(results_path)

    st.title("Per-Box Detection Review")

    tp = sum(1 for k in results if not k.endswith("_frame") and not k.endswith("_conf") and results[k] == "TP")
    fp = sum(1 for k in results if not k.endswith("_frame") and not k.endswith("_conf") and results[k] == "FP")
    st.caption(f"{tp + fp}/{len(images)} reviewed — press Y=TP  N=FP")
    remaining = len(images) - tp - fp
    fp_rate = fp / (tp + fp) * 100 if (tp + fp) > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", len(images))
    col2.metric("TP", tp)
    col3.metric("FP", fp)
    col4.metric("Remaining", remaining)
    col5.metric("FP Rate", f"{fp_rate:.0f}%" if (tp + fp) > 0 else "—")

    reviewed_set = {k for k in results if not k.endswith("_frame") and not k.endswith("_conf")}
    unreviewed = [i for i, img in enumerate(images) if img.name not in reviewed_set]
    reviewed = [i for i, img in enumerate(images) if img.name in reviewed_set]

    idx = unreviewed[0] if unreviewed else (reviewed[-1] if reviewed else 0)

    nav_col, prev_col, next_col, status_col = st.columns([2, 1, 1, 4])
    with nav_col:
        if st.button("◀ Prev", use_container_width=True):
            idx = max(0, idx - 1)
    with prev_col:
        if st.button("Next ▶", use_container_width=True):
            idx = min(len(images) - 1, idx + 1)
    with next_col:
        pass
    with status_col:
        st.caption(f"{idx + 1}/{len(images)}: {images[idx].name}")

    img = images[idx]
    st.image(str(img), use_container_width=False, width=400)

    frame_info = img.stem
    st.caption(f"Detection: {frame_info}")

    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 6])
    with btn_col1:
        if st.button("✅ TP (Y)", use_container_width=True, type="primary"):
            results[img.name] = "TP"
            results[f"{img.name}_frame"] = img.stem.split("_f")[1].split("_c")[0] if "_f" in img.stem else ""
            results[f"{img.name}_conf"] = img.stem.split("_c")[1] if "_c" in img.stem else ""
            save_results(results, results_path)
            st.rerun()
    with btn_col2:
        if st.button("❌ FP (N)", use_container_width=True):
            results[img.name] = "FP"
            results[f"{img.name}_frame"] = img.stem.split("_f")[1].split("_c")[0] if "_f" in img.stem else ""
            results[f"{img.name}_conf"] = img.stem.split("_c")[1] if "_c" in img.stem else ""
            save_results(results, results_path)
            st.rerun()
    with btn_col3:
        if st.button("Skip", use_container_width=True):
            idx = min(len(images) - 1, idx + 1)

    st.markdown("""
    <style>
    .stApp { max-width: 600px; margin: 0 auto; }
    .st-emotion-cache-1r4qj8v { gap: 0.5rem; }
    </style>
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.key === 'y' || e.key === 'Y') {
            let btns = window.parent.document.querySelectorAll('button');
            for (let b of btns) { if (b.innerText.includes('TP')) { b.click(); break; } }
        } else if (e.key === 'n' || e.key === 'N') {
            let btns = window.parent.document.querySelectorAll('button');
            for (let b of btns) { if (b.innerText.includes('FP')) { b.click(); break; } }
        }
    });
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
