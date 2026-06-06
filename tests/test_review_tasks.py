"""Tests for generic annotation task reviewer helpers."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.annotation_core import AnnotationTask
from tools.review_tasks import apply_review_decision, next_pending_task


def test_next_pending_task_returns_first_pending():
    tasks = [
        AnnotationTask(
            task_id="done",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=1.0,
            frame_index=25,
            sample_role="hard_negative_candidate",
            review_status="accepted",
            reviewer_decision="not_frisbee",
        ),
        AnnotationTask(
            task_id="pending",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=2.0,
            frame_index=50,
            sample_role="hard_negative_candidate",
        ),
    ]

    assert next_pending_task(tasks).task_id == "pending"


def test_next_pending_task_returns_none_when_no_pending_tasks():
    tasks = [
        AnnotationTask(
            task_id="done",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=1.0,
            frame_index=25,
            sample_role="hard_negative_candidate",
            review_status="accepted",
            reviewer_decision="not_frisbee",
        )
    ]

    assert next_pending_task(tasks) is None


def test_apply_review_decision_updates_matching_task_only():
    tasks = [
        AnnotationTask(
            task_id="a",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=1.0,
            frame_index=25,
            sample_role="hard_negative_candidate",
        ),
        AnnotationTask(
            task_id="b",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=2.0,
            frame_index=50,
            sample_role="hard_negative_candidate",
        ),
    ]

    updated = apply_review_decision(tasks, "b", "not_frisbee")

    assert updated[0].review_status == "pending"
    assert updated[1].review_status == "accepted"
    assert updated[1].reviewer_decision == "not_frisbee"


def test_apply_review_decision_handles_skip_and_reject_branches():
    tasks = [
        AnnotationTask(
            task_id="skip-me",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=1.0,
            frame_index=25,
            sample_role="hard_negative_candidate",
            reviewer_decision="not_frisbee",
        ),
        AnnotationTask(
            task_id="reject-me",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=2.0,
            frame_index=50,
            sample_role="hard_negative_candidate",
            reviewer_decision="not_frisbee",
        ),
    ]

    updated = apply_review_decision(tasks, "skip-me", "skipped")
    updated = apply_review_decision(updated, "reject-me", "rejected")

    assert updated[0].review_status == "skipped"
    assert updated[0].reviewer_decision == ""
    assert updated[1].review_status == "rejected"
    assert updated[1].reviewer_decision == ""


def test_apply_review_decision_rejects_invalid_decision():
    tasks = [
        AnnotationTask(
            task_id="a",
            task_type="bbox_review",
            source_video="movie/full.mp4",
            timestamp_sec=1.0,
            frame_index=25,
            sample_role="hard_negative_candidate",
        )
    ]

    with pytest.raises(ValueError, match="Invalid reviewer_decision"):
        apply_review_decision(tasks, "a", "maybe")
