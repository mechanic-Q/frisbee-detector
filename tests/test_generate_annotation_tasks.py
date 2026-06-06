"""Tests for annotation task generation helpers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.annotation_core import ExcludeRange, write_tasks
from tools.generate_annotation_tasks import (
    append_unique_tasks,
    build_bbox_review_task,
    should_keep_candidate,
)


def test_should_keep_candidate_filters_excluded_time():
    ranges = [
        ExcludeRange(
            source_video="movie/full.mp4",
            start_sec=0.0,
            end_sec=300.0,
            reason="eval_segment_buffer",
            eval_segment_id="first60s",
        )
    ]

    assert not should_keep_candidate("movie/full.mp4", 10.0, ranges)
    assert should_keep_candidate("movie/full.mp4", 400.0, ranges)


def test_build_bbox_review_task_uses_standard_fields():
    task = build_bbox_review_task(
        source_video="movie/full.mp4",
        timestamp_sec=400.0,
        frame_index=10000,
        bbox_xyxy=[100.0, 120.0, 140.0, 160.0],
        crop_path="data/annotation/assets/crop.jpg",
        frame_path="data/annotation/assets/frame.jpg",
        model_name="frisbee_det_p2_game_v3",
        model_conf=0.77,
        tags=["sideline"],
    )

    assert task.task_type == "bbox_review"
    assert task.sample_role == "hard_negative_candidate"
    assert task.review_status == "pending"
    assert task.reviewer_decision == ""
    assert task.crop_path.endswith("crop.jpg")
    assert task.tags == ["sideline"]


def test_append_unique_tasks_preserves_existing_task(tmp_path):
    existing = build_bbox_review_task(
        source_video="movie/full.mp4",
        timestamp_sec=400.0,
        frame_index=10000,
        bbox_xyxy=[100.0, 120.0, 140.0, 160.0],
        crop_path="old_crop.jpg",
        frame_path="old_frame.jpg",
        model_name="frisbee_det_p2_game_v3",
        model_conf=0.77,
    )
    duplicate = build_bbox_review_task(
        source_video="movie/full.mp4",
        timestamp_sec=400.0,
        frame_index=10000,
        bbox_xyxy=[100.0, 120.0, 140.0, 160.0],
        crop_path="new_crop.jpg",
        frame_path="new_frame.jpg",
        model_name="frisbee_det_p2_game_v3",
        model_conf=0.80,
    )
    task_store = tmp_path / "tasks.jsonl"
    write_tasks(task_store, [existing])

    merged = append_unique_tasks(task_store, [duplicate])

    assert len(merged) == 1
    assert merged[0].crop_path == "old_crop.jpg"
