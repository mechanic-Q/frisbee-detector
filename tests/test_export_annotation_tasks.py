"""Tests for annotation task exporter."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.annotation_core import AnnotationTask, ExcludeRange, write_tasks
from tools.export_annotation_tasks import export_reviewed_tasks


def test_export_reviewed_tasks_writes_positive_and_hard_negative(tmp_path):
    frame = tmp_path / "assets" / "frame.jpg"
    crop = tmp_path / "assets" / "crop.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"frame")
    crop.write_bytes(b"crop")

    tasks = [
        AnnotationTask(
            task_id="positive",
            task_type="frame_label",
            source_video="movie/full.mp4",
            timestamp_sec=400.0,
            frame_index=10000,
            sample_role="positive_candidate",
            review_status="accepted",
            reviewer_decision="frisbee",
            frame_path=str(frame),
            bbox_xyxy=[100.0, 120.0, 140.0, 160.0],
        ),
        AnnotationTask(
            task_id="hardneg",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=401.0,
            frame_index=10025,
            sample_role="hard_negative_candidate",
            review_status="accepted",
            reviewer_decision="not_frisbee",
            crop_path=str(crop),
        ),
    ]
    task_store = tmp_path / "tasks.jsonl"
    export_dir = tmp_path / "export"
    write_tasks(task_store, tasks)

    report = export_reviewed_tasks(
        task_store=task_store,
        export_dir=export_dir,
        exclude_ranges=[],
        image_size=(1920, 1080),
    )

    assert report["positive_count"] == 1
    assert report["hard_negative_count"] == 1
    assert (export_dir / "images" / "positive" / "positive.jpg").read_bytes() == b"frame"
    assert (export_dir / "labels" / "positive" / "positive.txt").read_text().startswith("0 ")
    assert (export_dir / "images" / "hard_negative" / "hardneg.jpg").read_bytes() == b"crop"
    assert (export_dir / "labels" / "hard_negative" / "hardneg.txt").read_text() == ""
    saved_report = json.loads((export_dir / "export_report.json").read_text())
    assert saved_report["positive_count"] == 1


def test_export_reviewed_tasks_blocks_leaking_task(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    task = AnnotationTask(
        task_id="leak",
        task_type="frame_label",
        source_video="movie/full.mp4",
        timestamp_sec=30.0,
        frame_index=750,
        sample_role="positive_candidate",
        review_status="accepted",
        reviewer_decision="frisbee",
        frame_path=str(frame),
        bbox_xyxy=[10.0, 20.0, 30.0, 40.0],
    )
    task_store = tmp_path / "tasks.jsonl"
    write_tasks(task_store, [task])

    ranges = [
        ExcludeRange(
            source_video="movie/full.mp4",
            start_sec=0.0,
            end_sec=300.0,
            reason="eval_segment_buffer",
            eval_segment_id="first60s",
        )
    ]

    try:
        export_reviewed_tasks(task_store, tmp_path / "export", ranges, image_size=(100, 100))
    except ValueError as exc:
        assert "leak" in str(exc)
    else:
        raise AssertionError("Expected leaking export to fail")
