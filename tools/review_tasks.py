"""Generic Streamlit reviewer for annotation tasks.

Keyboard: y/n/u/s/r to decide, Backspace to undo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.annotation_core import AnnotationTask, load_project_config, read_tasks, write_tasks


def draw_bbox_on_frame(frame_path: str, bbox_xyxy: list[float] | None) -> np.ndarray | None:
    frame = cv2.imread(frame_path)
    if frame is None:
        return None
    if bbox_xyxy and len(bbox_xyxy) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.circle(frame, (max(0, x1), max(0, y1)), 8, (0, 0, 255), -1)
    return frame


def bgr_to_png_bytes(frame: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".png", rgb)
    return buf.tobytes()


def next_pending_task(tasks: list[AnnotationTask]) -> AnnotationTask | None:
    for task in tasks:
        if task.review_status == "pending":
            return task
    return None


def review_question(task: AnnotationTask) -> str:
    if task.task_type == "bbox_review":
        return "Is the red model box a frisbee?"
    if task.task_type == "frame_label":
        return "Is this red box a valid shadow frisbee candidate?"
    return "Review this task."


def apply_review_decision(
    tasks: list[AnnotationTask],
    task_id: str,
    reviewer_decision: str,
) -> list[AnnotationTask]:
    valid = {"frisbee", "not_frisbee", "uncertain", "skipped", "rejected"}
    if reviewer_decision not in valid:
        raise ValueError(f"Invalid reviewer_decision: {reviewer_decision}")
    for task in tasks:
        if task.task_id != task_id:
            continue
        if reviewer_decision == "skipped":
            task.review_status = "skipped"
            task.reviewer_decision = ""
        elif reviewer_decision == "rejected":
            task.review_status = "rejected"
            task.reviewer_decision = ""
        else:
            task.review_status = "accepted"
            task.reviewer_decision = reviewer_decision
    return tasks


def reset_task_to_pending(tasks: list[AnnotationTask], task_id: str) -> list[AnnotationTask]:
    for task in tasks:
        if task.task_id == task_id:
            task.review_status = "pending"
            task.reviewer_decision = ""
            break
    return tasks


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review annotation tasks")
    parser.add_argument("--project", required=True, help="Annotation project YAML")
    return parser.parse_known_args()[0]


DECISIONS = ["frisbee", "not_frisbee", "uncertain", "skipped", "rejected"]
DECISION_LABELS = {
    "frisbee": "✅ Frisbee",
    "not_frisbee": "❌ Not Frisbee",
    "uncertain": "❓ Uncertain",
    "skipped": "⏭ Skip",
    "rejected": "🚫 Reject",
}


def count_tasks(tasks: list[AnnotationTask]) -> dict[str, int]:
    return {
        "pending": sum(1 for task in tasks if task.review_status == "pending"),
        "accepted": sum(1 for task in tasks if task.review_status == "accepted"),
        "skipped": sum(1 for task in tasks if task.review_status == "skipped"),
        "rejected": sum(1 for task in tasks if task.review_status == "rejected"),
        "total": len(tasks),
    }


def render_keyboard_shortcuts() -> None:
    import streamlit.components.v1 as components

    components.html("""
    <script>
    (function() {
        const parentDoc = window.parent.document;
        if (parentDoc.__reviewTasksKeyboardInstalled) return;
        parentDoc.__reviewTasksKeyboardInstalled = true;

        function clickButton(label) {
            const buttons = parentDoc.querySelectorAll('button');
            for (const button of buttons) {
                if (button.innerText.includes(label) && !button.disabled) {
                    button.click();
                    return true;
                }
            }
            return false;
        }

        parentDoc.addEventListener('keydown', function(e) {
            const tag = e.target && e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            if (e.ctrlKey || e.metaKey || e.altKey) return;

            if (e.key === 'y' || e.key === 'Y') { e.preventDefault(); clickButton('Frisbee'); }
            else if (e.key === 'n' || e.key === 'N') { e.preventDefault(); clickButton('Not Frisbee'); }
            else if (e.key === 'u' || e.key === 'U') { e.preventDefault(); clickButton('Uncertain'); }
            else if (e.key === 's' || e.key === 'S') { e.preventDefault(); clickButton('Skip'); }
            else if (e.key === 'r' || e.key === 'R') { e.preventDefault(); clickButton('Reject'); }
            else if (e.key === 'Backspace') { e.preventDefault(); clickButton('Undo'); }
        }, true);
    })();
    </script>
    """, height=0)


def main() -> None:
    import streamlit as st

    args = get_args()
    config = load_project_config(args.project)
    task_store = Path(config["outputs"]["task_store"])
    tasks = read_tasks(task_store)

    st.set_page_config(page_title="Annotation Review", layout="wide")
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100vw; }
    h1 { font-size: 1.25rem !important; margin-bottom: 0.25rem !important; }
    .stButton > button { height: 2.8rem; font-size: 0.95rem; font-weight: 650; }
    div[data-testid="stMetric"] { background: #11182708; border-radius: 0.45rem; padding: 0.25rem 0.45rem; }
    div[data-testid="stMetricValue"] { font-size: 1.05rem; }
    hr { margin: 0.4rem 0; }
    .task-line { font-size: 0.78rem; color: #777; line-height: 1.25; }
    .question { font-size: 1.05rem; font-weight: 750; margin: 0.15rem 0 0.45rem 0; }
    </style>
    """, unsafe_allow_html=True)

    if "undo_task_id" not in st.session_state:
        st.session_state.undo_task_id = None

    counts = count_tasks(tasks)
    task = next_pending_task(tasks)

    title_left, title_right = st.columns([3, 2])
    with title_left:
        st.title(f"Review — {config['project_id']}")
    with title_right:
        st.caption("⌨ y=Frisbee  n=Not  u=Uncertain  s=Skip  r=Reject  Backspace=Undo")

    if task is None:
        st.balloons()
        st.success("All tasks reviewed!")
        return

    main_col, side_col = st.columns([4.2, 1.35], gap="medium")

    with side_col:
        st.markdown(f'<div class="question">{review_question(task)}</div>', unsafe_allow_html=True)
        for decision in DECISIONS:
            if st.button(DECISION_LABELS[decision], key=f"btn_{decision}", use_container_width=True):
                apply_review_decision(tasks, task.task_id, decision)
                write_tasks(task_store, tasks)
                st.session_state.undo_task_id = task.task_id
                st.rerun()
        if st.session_state.undo_task_id:
            if st.button("↩ Undo", use_container_width=True):
                reset_task_to_pending(tasks, st.session_state.undo_task_id)
                write_tasks(task_store, tasks)
                st.session_state.undo_task_id = None
                st.rerun()
        else:
            st.button("↩ Undo", disabled=True, use_container_width=True)

        st.divider()
        metric_cols = st.columns(2)
        metric_cols[0].metric("Pending", counts["pending"])
        metric_cols[1].metric("Done", counts["total"] - counts["pending"])
        metric_cols[0].metric("Accepted", counts["accepted"])
        metric_cols[1].metric("Rejected", counts["rejected"])
        st.progress((counts["total"] - counts["pending"]) / max(counts["total"], 1))

        st.divider()
        if task.crop_path and Path(task.crop_path).exists():
            st.image(task.crop_path, caption="Crop / zoom", use_container_width=True)
        st.markdown(
            f'<div class="task-line"><b>{task.task_type}</b><br>'
            f'{task.task_id}<br>'
            f'frame={task.frame_index} t={task.timestamp_sec:.1f}s<br>'
            f'conf={task.model_conf if task.model_conf is not None else "—"}<br>'
            f'tags={task.tags}</div>',
            unsafe_allow_html=True,
        )

    with main_col:
        if task.frame_path and Path(task.frame_path).exists():
            annotated = draw_bbox_on_frame(task.frame_path, task.bbox_xyxy)
            if annotated is not None:
                st.image(
                    bgr_to_png_bytes(annotated),
                    caption="Main frame — red box is the candidate under review",
                    use_container_width=True,
                )
        else:
            st.error(f"Frame not found: {task.frame_path}")

    render_keyboard_shortcuts()


if __name__ == "__main__":
    main()
