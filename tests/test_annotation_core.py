"""Tests for annotation task core models and leakage rules."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.annotation_core import (
    AnnotationTask,
    EvalSegment,
    ExcludeRange,
    build_exclude_ranges,
    is_excluded_timestamp,
    load_project_config,
    make_task_id,
    read_tasks,
    write_tasks,
)


def test_build_exclude_ranges_clamps_to_zero():
    segment = EvalSegment(
        eval_segment_id="first60s",
        eval_video="movie/eval.mp4",
        source_video="movie/full.mp4",
        source_start_sec=0.0,
        source_end_sec=60.0,
        buffer_sec=240.0,
    )

    ranges = build_exclude_ranges([segment])

    assert ranges == [
        ExcludeRange(
            source_video="movie/full.mp4",
            start_sec=0.0,
            end_sec=300.0,
            reason="eval_segment_buffer",
            eval_segment_id="first60s",
        )
    ]


def test_is_excluded_timestamp_matches_source_video_only():
    ranges = [
        ExcludeRange(
            source_video="movie/full.mp4",
            start_sec=10.0,
            end_sec=20.0,
            reason="eval_segment_buffer",
            eval_segment_id="clip",
        )
    ]

    assert is_excluded_timestamp("movie/full.mp4", 15.0, ranges)
    assert not is_excluded_timestamp("movie/full.mp4", 25.0, ranges)
    assert not is_excluded_timestamp("movie/other.mp4", 15.0, ranges)


def test_make_task_id_is_stable():
    first = make_task_id("bbox_review", "movie/full.mp4", 12.5, 313, [1, 2, 3, 4])
    second = make_task_id("bbox_review", "movie/full.mp4", 12.5, 313, [1, 2, 3, 4])
    relative = make_task_id("bbox_review", "./movie/full.mp4", 12.5, 313, [1, 2, 3, 4])
    absolute = make_task_id(
        "bbox_review",
        str(Path.cwd() / "movie/full.mp4"),
        12.5,
        313,
        [1, 2, 3, 4],
    )

    assert first == second
    assert first == relative
    assert first == absolute
    assert first.startswith("bbox_review_")


def test_task_from_dict_ignores_unknown_fields_and_uses_defaults():
    task = AnnotationTask.from_dict(
        {
            "task_id": "bbox_review_abc",
            "task_type": "bbox_review",
            "source_video": "movie/full.mp4",
            "timestamp_sec": 12.5,
            "frame_index": 313,
            "sample_role": "hard_negative_candidate",
            "future_field": "ignored",
        }
    )

    assert task.task_id == "bbox_review_abc"
    assert task.review_status == "pending"
    assert task.reviewer_decision == ""
    assert task.bbox_xyxy is None
    assert task.tags == []


def test_task_from_dict_reports_missing_required_fields():
    with pytest.raises(ValueError, match="source_video"):
        AnnotationTask.from_dict(
            {
                "task_id": "bbox_review_abc",
                "task_type": "bbox_review",
                "timestamp_sec": 12.5,
                "frame_index": 313,
                "sample_role": "hard_negative_candidate",
            }
        )


def test_task_jsonl_round_trip(tmp_path):
    task = AnnotationTask(
        task_id="bbox_review_abc",
        task_type="bbox_review",
        source_video="movie/full.mp4",
        timestamp_sec=12.5,
        frame_index=313,
        sample_role="hard_negative_candidate",
        review_status="pending",
        reviewer_decision="",
        bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
        crop_path="assets/crop.jpg",
        frame_path="assets/frame.jpg",
        model_name="frisbee_det_p2_game_v3",
        model_conf=0.42,
        tags=["sideline"],
    )
    path = tmp_path / "tasks.jsonl"

    write_tasks(path, [task])
    loaded = read_tasks(path)

    assert loaded == [task]


def test_load_project_config_builds_exclude_ranges(tmp_path):
    config_path = tmp_path / "annotation.yaml"
    config_path.write_text(
        """
project_id: p2_shadow_fp_round1
eval_segments:
  - eval_segment_id: first60s
    eval_video: movie/eval.mp4
    source_video: movie/full.mp4
    source_start_sec: 0.0
    source_end_sec: 60.0
    buffer_sec: 240.0
outputs:
  task_store: data/annotation/tasks.jsonl
  asset_dir: data/annotation/assets
  export_dir: data/annotation/export
"""
    )

    config = load_project_config(config_path)

    assert config["project_id"] == "p2_shadow_fp_round1"
    assert config["exclude_ranges"][0].end_sec == pytest.approx(300.0)


def test_is_excluded_timestamp_normalizes_equivalent_paths():
    segment = EvalSegment(
        eval_segment_id="clip",
        eval_video="movie/eval.mp4",
        source_video="./movie/full.mp4",
        source_start_sec=10.0,
        source_end_sec=20.0,
        buffer_sec=0.0,
    )

    ranges = build_exclude_ranges([segment])

    assert ranges[0].source_video == "movie/full.mp4"
    assert is_excluded_timestamp("movie/full.mp4", 15.0, ranges)
    assert is_excluded_timestamp("./movie/full.mp4", 15.0, ranges)
    assert is_excluded_timestamp(str(Path.cwd() / "movie/full.mp4"), 15.0, ranges)


def test_is_excluded_timestamp_includes_boundaries_and_excludes_outside():
    ranges = [
        ExcludeRange(
            source_video="movie/full.mp4",
            start_sec=10.0,
            end_sec=20.0,
            reason="eval_segment_buffer",
            eval_segment_id="clip",
        )
    ]

    assert is_excluded_timestamp("movie/full.mp4", 10.0, ranges)
    assert is_excluded_timestamp("movie/full.mp4", 20.0, ranges)
    assert not is_excluded_timestamp("movie/full.mp4", 9.999, ranges)
    assert not is_excluded_timestamp("movie/full.mp4", 20.001, ranges)


def test_read_tasks_missing_and_empty_files_return_empty(tmp_path):
    missing_path = tmp_path / "missing.jsonl"
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("")

    assert read_tasks(missing_path) == []
    assert read_tasks(empty_path) == []
