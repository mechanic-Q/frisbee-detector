"""Generate annotation tasks from videos and model candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.annotation_core import (
    AnnotationTask,
    ExcludeRange,
    is_excluded_timestamp,
    load_project_config,
    make_task_id,
    read_tasks,
    write_tasks,
)


def should_keep_candidate(
    source_video: str,
    timestamp_sec: float,
    exclude_ranges: list[ExcludeRange],
) -> bool:
    return not is_excluded_timestamp(source_video, timestamp_sec, exclude_ranges)


def build_bbox_review_task(
    source_video: str,
    timestamp_sec: float,
    frame_index: int,
    bbox_xyxy: list[float],
    crop_path: str,
    frame_path: str,
    model_name: str,
    model_conf: float,
    tags: list[str] | None = None,
) -> AnnotationTask:
    task_id = make_task_id("bbox_review", source_video, timestamp_sec, frame_index, bbox_xyxy)
    return AnnotationTask(
        task_id=task_id,
        task_type="bbox_review",
        source_video=source_video,
        timestamp_sec=timestamp_sec,
        frame_index=frame_index,
        sample_role="hard_negative_candidate",
        review_status="pending",
        reviewer_decision="",
        bbox_xyxy=bbox_xyxy,
        crop_path=crop_path,
        frame_path=frame_path,
        model_name=model_name,
        model_conf=model_conf,
        tags=tags or [],
    )


def append_unique_tasks(task_store: str | Path, new_tasks: list[AnnotationTask]) -> list[AnnotationTask]:
    existing_tasks = read_tasks(task_store)
    tasks_by_id = {task.task_id: task for task in existing_tasks}
    for task in new_tasks:
        tasks_by_id.setdefault(task.task_id, task)
    merged_tasks = list(tasks_by_id.values())
    write_tasks(task_store, merged_tasks)
    return merged_tasks


def generate_bbox_review_tasks(
    project_config: dict,
    max_tasks: int,
    frame_stride: int,
) -> list[AnnotationTask]:
    import cv2
    from ultralytics import YOLO

    if max_tasks < 0:
        raise ValueError("max_tasks must be non-negative")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    model_config = project_config["models"]["candidate_model"]
    source_video = project_config["source_videos"][0]["source_video"]
    outputs = project_config["outputs"]
    asset_dir = Path(outputs["asset_dir"])
    crop_dir = asset_dir / "crops"
    frame_dir = asset_dir / "frames"
    crop_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_config["model_path"])
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source_video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_index = 0
    tasks: list[AnnotationTask] = []

    try:
        while len(tasks) < max_tasks:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % frame_stride != 0:
                continue

            timestamp_sec = frame_index / fps
            if not should_keep_candidate(source_video, timestamp_sec, project_config["exclude_ranges"]):
                continue

            results = model.predict(
                frame,
                conf=float(model_config.get("conf", 0.35)),
                verbose=False,
                imgsz=640,
            )
            if not results:
                continue

            frame_context_path = frame_dir / f"frame_f{frame_index:06d}.jpg"
            if not frame_context_path.exists():
                cv2.imwrite(str(frame_context_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            frame_height, frame_width = frame.shape[:2]
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    if len(tasks) >= max_tasks:
                        break
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    clipped_x1 = max(0, int(x1))
                    clipped_y1 = max(0, int(y1))
                    clipped_x2 = min(frame_width, int(x2))
                    clipped_y2 = min(frame_height, int(y2))
                    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
                        continue

                    crop = frame[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
                    crop_path = crop_dir / f"crop_f{frame_index:06d}_{len(tasks):04d}.jpg"
                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    tasks.append(
                        build_bbox_review_task(
                            source_video=source_video,
                            timestamp_sec=timestamp_sec,
                            frame_index=frame_index,
                            bbox_xyxy=[float(clipped_x1), float(clipped_y1), float(clipped_x2), float(clipped_y2)],
                            crop_path=str(crop_path),
                            frame_path=str(frame_context_path),
                            model_name=model_config["model_name"],
                            model_conf=float(box.conf[0]),
                            tags=["detection_crop"],
                        )
                    )
    finally:
        cap.release()

    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate annotation task candidates")
    parser.add_argument("--project", required=True, help="Annotation project YAML")
    parser.add_argument("--task-type", choices=["bbox_review"], required=True)
    parser.add_argument("--max-tasks", type=int, default=150)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_project_config(args.project)
    outputs = config["outputs"]
    if args.dry_run:
        print(f"Project: {config['project_id']}")
        print(f"Task store: {outputs['task_store']}")
        print(f"Exclude ranges: {len(config['exclude_ranges'])}")
        return 0

    generated_tasks = generate_bbox_review_tasks(
        project_config=config,
        max_tasks=args.max_tasks,
        frame_stride=args.frame_stride,
    )
    merged_tasks = append_unique_tasks(outputs["task_store"], generated_tasks)
    print(f"Generated tasks: {len(generated_tasks)}")
    print(f"Task store: {outputs['task_store']}")
    print(f"Total tasks: {len(merged_tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
