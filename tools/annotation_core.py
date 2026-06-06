"""Core models and pure functions for annotation task workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
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
        return cls(**data)


def build_exclude_ranges(eval_segments: list[EvalSegment]) -> list[ExcludeRange]:
    ranges: list[ExcludeRange] = []
    for segment in eval_segments:
        start_sec = max(0.0, segment.source_start_sec - segment.buffer_sec)
        end_sec = segment.source_end_sec + segment.buffer_sec
        ranges.append(
            ExcludeRange(
                source_video=segment.source_video,
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
    for exclude_range in exclude_ranges:
        if exclude_range.source_video != source_video:
            continue
        if exclude_range.start_sec <= timestamp_sec <= exclude_range.end_sec:
            return True
    return False


def make_task_id(
    task_type: str,
    source_video: str,
    timestamp_sec: float,
    frame_index: int,
    bbox_xyxy: list[float] | None = None,
) -> str:
    bbox_part = "" if bbox_xyxy is None else ",".join(f"{value:.2f}" for value in bbox_xyxy)
    raw = f"{task_type}|{source_video}|{timestamp_sec:.3f}|{frame_index}|{bbox_part}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{task_type}_{digest}"


def read_tasks(task_store: str | Path) -> list[AnnotationTask]:
    path = Path(task_store)
    if not path.exists():
        return []
    tasks: list[AnnotationTask] = []
    for line in path.read_text().splitlines():
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
    path.write_text(content)


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text()) or {}
    eval_segments = [
        EvalSegment(**segment)
        for segment in config.get("eval_segments", [])
    ]
    config["eval_segments"] = eval_segments
    config["exclude_ranges"] = build_exclude_ranges(eval_segments)
    return config
