"""Core models and pure functions for annotation task workflows."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvalSegment:
    eval_segment_id: str
    eval_video: str
    source_video: str
    source_start_sec: float
    source_end_sec: float
    buffer_sec: float


@dataclass(frozen=True)
class ExcludeRange:
    source_video: str
    start_sec: float
    end_sec: float
    reason: str
    eval_segment_id: str


@dataclass
class AnnotationTask:
    task_id: str
    task_type: str
    source_video: str
    timestamp_sec: float
    frame_index: int
    sample_role: str
    review_status: str = "pending"
    reviewer_decision: str = ""
    eval_segment_id: str = ""
    bbox_xyxy: list[float] | None = None
    crop_path: str = ""
    frame_path: str = ""
    model_name: str = ""
    model_conf: float | None = None
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    reviewed_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationTask":
        task_fields = {task_field.name: task_field for task_field in fields(cls)}
        missing_fields = [
            name
            for name, task_field in task_fields.items()
            if task_field.default is MISSING
            and task_field.default_factory is MISSING
            and name not in data
        ]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Missing required AnnotationTask fields: {missing}")

        filtered_data = {
            name: value
            for name, value in data.items()
            if name in task_fields
        }
        return cls(**filtered_data)


def normalize_source_video(source_video: str) -> str:
    path = Path(source_video).expanduser()
    cwd = Path.cwd().resolve(strict=False)

    if path.is_absolute():
        resolved = path.resolve(strict=False)
        try:
            return resolved.relative_to(cwd).as_posix()
        except ValueError:
            return resolved.as_posix()

    return Path(os.path.normpath(str(path))).as_posix()


def build_exclude_ranges(eval_segments: list[EvalSegment]) -> list[ExcludeRange]:
    ranges: list[ExcludeRange] = []
    for segment in eval_segments:
        start_sec = max(0.0, segment.source_start_sec - segment.buffer_sec)
        end_sec = segment.source_end_sec + segment.buffer_sec
        ranges.append(
            ExcludeRange(
                source_video=normalize_source_video(segment.source_video),
                start_sec=start_sec,
                end_sec=end_sec,
                reason="eval_segment_buffer",
                eval_segment_id=segment.eval_segment_id,
            )
        )
    return ranges


def is_excluded_timestamp(
    source_video: str,
    timestamp_sec: float,
    exclude_ranges: list[ExcludeRange],
) -> bool:
    normalized_source_video = normalize_source_video(source_video)
    for exclude_range in exclude_ranges:
        if normalize_source_video(exclude_range.source_video) != normalized_source_video:
            continue
        if exclude_range.start_sec <= timestamp_sec <= exclude_range.end_sec:
            return True
    return False


def exportable_positive_tasks(tasks: list[AnnotationTask]) -> list[AnnotationTask]:
    return [
        task for task in tasks
        if task.task_type == "frame_label"
        and task.review_status == "accepted"
        and task.reviewer_decision == "frisbee"
        and task.frame_path
    ]


def exportable_hard_negative_tasks(tasks: list[AnnotationTask]) -> list[AnnotationTask]:
    return [
        task for task in tasks
        if task.task_type == "bbox_review"
        and task.review_status == "accepted"
        and task.reviewer_decision == "not_frisbee"
        and task.crop_path
    ]


def assert_tasks_not_leaking(
    tasks: list[AnnotationTask],
    exclude_ranges: list[ExcludeRange],
) -> None:
    leaking = [
        task.task_id
        for task in tasks
        if is_excluded_timestamp(task.source_video, task.timestamp_sec, exclude_ranges)
    ]
    if leaking:
        joined = ", ".join(leaking[:10])
        raise ValueError(f"Tasks overlap excluded eval ranges: {joined}")


def make_task_id(
    task_type: str,
    source_video: str,
    timestamp_sec: float,
    frame_index: int,
    bbox_xyxy: list[float] | None = None,
) -> str:
    bbox_part = "" if bbox_xyxy is None else ",".join(f"{value:.2f}" for value in bbox_xyxy)
    normalized_source_video = normalize_source_video(source_video)
    raw = f"{task_type}|{normalized_source_video}|{timestamp_sec:.3f}|{frame_index}|{bbox_part}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{task_type}_{digest}"


def read_tasks(task_store: str | Path) -> list[AnnotationTask]:
    path = Path(task_store)
    if not path.exists():
        return []
    tasks: list[AnnotationTask] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tasks.append(AnnotationTask.from_dict(json.loads(line)))
    return tasks


def write_tasks(task_store: str | Path, tasks: list[AnnotationTask]) -> None:
    path = Path(task_store)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(task.to_json() for task in tasks)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    eval_segments = [
        EvalSegment(**segment)
        for segment in config.get("eval_segments", [])
    ]
    config["eval_segments"] = eval_segments
    config["exclude_ranges"] = build_exclude_ranges(eval_segments)
    return config
