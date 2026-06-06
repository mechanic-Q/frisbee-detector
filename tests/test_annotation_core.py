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
    assert_tasks_not_leaking,
    build_exclude_ranges,
    exportable_hard_negative_tasks,
    exportable_positive_tasks,
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


def test_export_filters_exclude_uncertain_tasks():
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
            bbox_xyxy=[100.0, 120.0, 140.0, 160.0],
            frame_path="assets/frame.jpg",
        ),
        AnnotationTask(
            task_id="uncertain",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=401.0,
            frame_index=10025,
            sample_role="hard_negative_candidate",
            review_status="accepted",
            reviewer_decision="uncertain",
            crop_path="assets/crop.jpg",
        ),
    ]

    assert [task.task_id for task in exportable_positive_tasks(tasks)] == ["positive"]
    assert exportable_hard_negative_tasks(tasks) == []


def test_export_filters_include_not_frisbee_hard_negatives():
    task = AnnotationTask(
        task_id="hardneg",
        task_type="bbox_review",
        source_video="movie/full.mp4",
        timestamp_sec=450.0,
        frame_index=11250,
        sample_role="hard_negative_candidate",
        review_status="accepted",
        reviewer_decision="not_frisbee",
        crop_path="assets/crop.jpg",
    )

    assert exportable_hard_negative_tasks([task]) == [task]


def test_export_filters_exclude_positive_without_bbox():
    task = AnnotationTask(
        task_id="positive_without_bbox",
        task_type="frame_label",
        source_video="movie/full.mp4",
        timestamp_sec=460.0,
        frame_index=11500,
        sample_role="positive_candidate",
        review_status="accepted",
        reviewer_decision="frisbee",
        frame_path="assets/frame.jpg",
    )

    assert exportable_positive_tasks([task]) == []


def test_export_filters_exclude_malformed_positive_bbox():
    malformed_bboxes = [
        [1.0, 2.0, 3.0],
        "abcd",
        12,
        [1.0, 2.0, "x", 4.0],
        [1.0, 2.0, float("nan"), 4.0],
        [4.0, 2.0, 1.0, 6.0],
        [1.0, 6.0, 4.0, 2.0],
    ]

    for index, bbox_xyxy in enumerate(malformed_bboxes):
        task = AnnotationTask(
            task_id=f"malformed_bbox_{index}",
            task_type="frame_label",
            source_video="movie/full.mp4",
            timestamp_sec=465.0 + index,
            frame_index=11600 + index,
            sample_role="positive_candidate",
            review_status="accepted",
            reviewer_decision="frisbee",
            bbox_xyxy=bbox_xyxy,
            frame_path="assets/frame.jpg",
        )

        assert exportable_positive_tasks([task]) == []


def test_export_filters_exclude_wrong_sample_role():
    positive_task = AnnotationTask(
        task_id="wrong_positive_role",
        task_type="frame_label",
        source_video="movie/full.mp4",
        timestamp_sec=470.0,
        frame_index=11750,
        sample_role="quality_check",
        review_status="accepted",
        reviewer_decision="frisbee",
        bbox_xyxy=[100.0, 120.0, 140.0, 160.0],
        frame_path="assets/frame.jpg",
    )
    hard_negative_task = AnnotationTask(
        task_id="wrong_hard_negative_role",
        task_type="bbox_review",
        source_video="movie/full.mp4",
        timestamp_sec=480.0,
        frame_index=12000,
        sample_role="quality_check",
        review_status="accepted",
        reviewer_decision="not_frisbee",
        crop_path="assets/crop.jpg",
    )

    assert exportable_positive_tasks([positive_task]) == []
    assert exportable_hard_negative_tasks([hard_negative_task]) == []


def test_assert_tasks_not_leaking_raises_for_excluded_task():
    task = AnnotationTask(
        task_id="leak",
        task_type="frame_label",
        source_video="movie/full.mp4",
        timestamp_sec=30.0,
        frame_index=750,
        sample_role="positive_candidate",
    )
    ranges = [
        ExcludeRange(
            source_video="movie/full.mp4",
            start_sec=0.0,
            end_sec=300.0,
            reason="eval_segment_buffer",
            eval_segment_id="first60s",
        )
    ]

    with pytest.raises(ValueError, match="leak"):
        assert_tasks_not_leaking([task], ranges)
