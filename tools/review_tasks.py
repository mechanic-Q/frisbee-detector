"""Generic Streamlit reviewer for annotation tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.annotation_core import AnnotationTask, load_project_config, read_tasks, write_tasks


def next_pending_task(tasks: list[AnnotationTask]) -> AnnotationTask | None:
    for task in tasks:
        if task.review_status == "pending":
            return task
    return None


def apply_review_decision(
    tasks: list[AnnotationTask],
    task_id: str,
    reviewer_decision: str,
) -> list[AnnotationTask]:
    valid_decisions = {"frisbee", "not_frisbee", "uncertain", "skipped", "rejected"}
    if reviewer_decision not in valid_decisions:
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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review annotation tasks")
    parser.add_argument("--project", required=True, help="Annotation project YAML")
    return parser.parse_known_args()[0]


def main() -> None:
    import streamlit as st

    args = get_args()
    config = load_project_config(args.project)
    task_store = Path(config["outputs"]["task_store"])
    tasks = read_tasks(task_store)

    st.set_page_config(page_title="Annotation Task Review", layout="wide")
    st.title("Annotation Task Review")
    st.caption(str(task_store))

    pending = sum(1 for task in tasks if task.review_status == "pending")
    accepted = sum(1 for task in tasks if task.review_status == "accepted")
    skipped = sum(1 for task in tasks if task.review_status == "skipped")
    rejected = sum(1 for task in tasks if task.review_status == "rejected")
    st.write({"pending": pending, "accepted": accepted, "skipped": skipped, "rejected": rejected})

    task = next_pending_task(tasks)
    if task is None:
        st.success("All tasks reviewed")
        return

    st.subheader(f"{task.task_type}: {task.task_id}")
    st.write(
        {
            "source_video": task.source_video,
            "timestamp_sec": task.timestamp_sec,
            "frame_index": task.frame_index,
            "sample_role": task.sample_role,
            "model_name": task.model_name,
            "model_conf": task.model_conf,
            "tags": task.tags,
        }
    )

    if task.task_type == "bbox_review" and task.crop_path:
        st.image(task.crop_path, caption="candidate crop", width=420)
        if task.frame_path:
            st.image(task.frame_path, caption="source frame context", use_container_width=True)
    elif task.task_type == "frame_label" and task.frame_path:
        st.image(task.frame_path, caption="candidate frame", use_container_width=True)

    columns = st.columns(5)
    choices = [
        (columns[0], "frisbee", "Frisbee"),
        (columns[1], "not_frisbee", "Not Frisbee"),
        (columns[2], "uncertain", "Uncertain"),
        (columns[3], "skipped", "Skip"),
        (columns[4], "rejected", "Reject"),
    ]
    for column, decision, label in choices:
        with column:
            if st.button(label, use_container_width=True):
                updated = apply_review_decision(tasks, task.task_id, decision)
                write_tasks(task_store, updated)
                st.rerun()


if __name__ == "__main__":
    main()
