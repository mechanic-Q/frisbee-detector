"""Generate annotation tasks from videos and model candidates."""

from __future__ import annotations

import argparse
import base64
import os
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

# ── GLM-4V-Flash VLM client (reused across calls) ──────────────────────────
_GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
_GLM_CLIENT = None
_VLM_CACHE: dict[str, bool | None] = {}

def _get_glm_client():
    global _GLM_CLIENT
    if _GLM_CLIENT is None:
        from openai import OpenAI
        _GLM_CLIENT = OpenAI(api_key=_GLM_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")
    return _GLM_CLIENT

def vlm_has_frisbee(frame, frame_id: str) -> bool | None:
    """Ask GLM-4V-Flash whether a frisbee is visible in the frame.

    Returns True/False, or None on API error.
    """
    import cv2
    if frame_id in _VLM_CACHE:
        return _VLM_CACHE[frame_id]

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()
    try:
        client = _get_glm_client()
        resp = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": (
                        "Is there a frisbee (flying disc) visible anywhere in this image? "
                        "A frisbee is a round plastic disc thrown in ultimate frisbee. "
                        "Look carefully in shadow areas — frisbees in shadows may be dim. "
                        "Answer only YES or NO."
                    )}
                ]
            }]
        )
        answer = resp.choices[0].message.content.strip().upper()
        result = "YES" in answer
        _VLM_CACHE[frame_id] = result
        return result
    except Exception as e:
        print(f"  VLM error: {e}")
        _VLM_CACHE[frame_id] = None
        return None


def vlm_candidate_is_frisbee(frame, bbox_xyxy: list[float], frame_id: str) -> bool | None:
    """Ask GLM-4V-Flash whether the red boxed object is a frisbee."""
    import cv2
    cache_key = f"candidate:{frame_id}"
    if cache_key in _VLM_CACHE:
        return _VLM_CACHE[cache_key]

    annotated = frame.copy()
    x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()
    try:
        client = _get_glm_client()
        resp = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": (
                        "The image has one red rectangle. Is the object inside the red rectangle a frisbee/flying disc? "
                        "Ignore other objects outside the red rectangle. The object may be small or in shadow. "
                        "Answer only YES or NO."
                    )},
                ],
            }],
        )
        answer = resp.choices[0].message.content.strip().upper()
        result = "YES" in answer
        _VLM_CACHE[cache_key] = result
        return result
    except Exception as e:
        print(f"  VLM candidate error: {e}")
        _VLM_CACHE[cache_key] = None
        return None


def should_keep_candidate(
    source_video: str,
    timestamp_sec: float,
    exclude_ranges: list[ExcludeRange],
) -> bool:
    return not is_excluded_timestamp(source_video, timestamp_sec, exclude_ranges)


def should_process_frame_index(frame_index: int, start_frame: int = 0, end_frame: int = 0) -> bool:
    if start_frame > 0 and frame_index < start_frame:
        return False
    if end_frame > 0 and frame_index > end_frame:
        return False
    return True

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

def compute_shadow_score(pixel_rows) -> float:
    brightness_values: list[float] = []
    for row in pixel_rows:
        for pixel in row:
            if len(pixel) >= 3:
                brightness_values.append((float(pixel[0]) + float(pixel[1]) + float(pixel[2])) / 3.0)
    if not brightness_values:
        return 0.0

    mean_brightness = sum(brightness_values) / len(brightness_values)
    return max(0.0, min(1.0, 1.0 - mean_brightness / 255.0))

def build_frame_label_task(
    source_video: str,
    timestamp_sec: float,
    frame_index: int,
    frame_path: str,
    model_name: str,
    tags: list[str] | None = None,
    bbox_xyxy: list[float] | None = None,
    crop_path: str = "",
    model_conf: float | None = None,
) -> AnnotationTask:
    task_id = make_task_id("frame_label", source_video, timestamp_sec, frame_index, bbox_xyxy)
    return AnnotationTask(
        task_id=task_id,
        task_type="frame_label",
        source_video=source_video,
        timestamp_sec=timestamp_sec,
        frame_index=frame_index,
        sample_role="positive_candidate",
        review_status="pending",
        reviewer_decision="",
        bbox_xyxy=bbox_xyxy,
        crop_path=crop_path,
        frame_path=frame_path,
        model_name=model_name,
        model_conf=model_conf,
        tags=tags or [],
    )


def keep_temporally_spaced_tasks(
    tasks: list[AnnotationTask],
    min_frame_gap: int,
) -> list[AnnotationTask]:
    """Keep the best task in each fixed frame window.

    A dense stream of candidates every few frames should produce one candidate
    per min_frame_gap window, not one candidate for the entire connected chain.
    """
    if min_frame_gap <= 0:
        return tasks
    if not tasks:
        return []

    kept: list[AnnotationTask] = []
    ordered = sorted(tasks, key=lambda candidate: candidate.frame_index)
    window_start = ordered[0].frame_index
    window_tasks: list[AnnotationTask] = []

    def best_in_window(candidates: list[AnnotationTask]) -> AnnotationTask:
        return max(candidates, key=lambda task: task.model_conf or 0.0)

    for task in ordered:
        if task.frame_index < window_start + min_frame_gap:
            window_tasks.append(task)
            continue
        kept.append(best_in_window(window_tasks))
        window_start = task.frame_index
        window_tasks = [task]

    kept.append(best_in_window(window_tasks))
    return kept

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
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
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

def generate_frame_label_tasks(
    project_config: dict,
    max_tasks: int,
    frame_stride: int,
    shadow_threshold: float = 0.55,
    use_vlm: bool = False,
    candidate_conf: float = 0.03,
    temporal_dedupe_frames: int = 25,
    start_frame: int = 0,
    end_frame: int = 0,
) -> list[AnnotationTask]:
    """Generate frame_label tasks with low-conf model bboxes on shadow frames.

    Order matters: YOLO proposes precise low-conf boxes first, temporal dedupe reduces
    repeated frames, then VLM verifies only the remaining red-box candidates.
    """
    import cv2
    from ultralytics import YOLO

    if max_tasks < 0:
        raise ValueError("max_tasks must be non-negative")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if candidate_conf <= 0:
        raise ValueError("candidate_conf must be positive")

    model_config = project_config["models"]["candidate_model"]
    source_video = project_config["source_videos"][0]["source_video"]
    outputs = project_config["outputs"]
    asset_dir = Path(outputs["asset_dir"])
    frame_dir = asset_dir / "shadow_frames"
    crop_dir = asset_dir / "shadow_crops"
    frame_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_config["model_path"])
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source_video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_index = 0
    raw_tasks: list[AnnotationTask] = []
    raw_limit = max(max_tasks * 8, max_tasks)

    try:
        while len(raw_tasks) < raw_limit:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if not should_process_frame_index(frame_index, start_frame, end_frame):
                if end_frame > 0 and frame_index > end_frame:
                    break
                continue
            if frame_index % frame_stride != 0:
                continue

            timestamp_sec = frame_index / fps
            if not should_keep_candidate(source_video, timestamp_sec, project_config["exclude_ranges"]):
                continue

            shadow_score = compute_shadow_score(frame.tolist())
            if shadow_score < shadow_threshold:
                continue

            results = model.predict(frame, conf=candidate_conf, verbose=False, imgsz=640)
            if not results:
                continue

            frame_path = frame_dir / f"shadow_f{frame_index:06d}.jpg"
            if not frame_path.exists():
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            frame_height, frame_width = frame.shape[:2]
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    clipped_x1 = max(0, int(x1))
                    clipped_y1 = max(0, int(y1))
                    clipped_x2 = min(frame_width, int(x2))
                    clipped_y2 = min(frame_height, int(y2))
                    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
                        continue

                    crop = frame[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
                    crop_path = crop_dir / f"shadow_crop_f{frame_index:06d}_{len(raw_tasks):04d}.jpg"
                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    raw_tasks.append(
                        build_frame_label_task(
                            source_video=source_video,
                            timestamp_sec=timestamp_sec,
                            frame_index=frame_index,
                            frame_path=str(frame_path),
                            model_name=model_config["model_name"],
                            tags=["shadow", "low_conf_model"],
                            bbox_xyxy=[float(clipped_x1), float(clipped_y1), float(clipped_x2), float(clipped_y2)],
                            crop_path=str(crop_path),
                            model_conf=float(box.conf[0]),
                        )
                    )
    finally:
        cap.release()

    deduped_tasks = keep_temporally_spaced_tasks(raw_tasks, temporal_dedupe_frames)
    if not use_vlm:
        return deduped_tasks[:max_tasks]

    verified_tasks: list[AnnotationTask] = []
    vlm_yes = 0
    vlm_no = 0
    vlm_err = 0
    for task in deduped_tasks:
        if len(verified_tasks) >= max_tasks:
            break
        if not task.bbox_xyxy:
            continue
        frame = cv2.imread(task.frame_path)
        if frame is None:
            vlm_err += 1
            task.tags.append("vlm_error_kept")
            verified_tasks.append(task)
            continue
        result = vlm_candidate_is_frisbee(frame, task.bbox_xyxy, f"{task.source_video}:{task.frame_index}:{task.bbox_xyxy}")
        if result is True:
            vlm_yes += 1
            task.tags.append("vlm_verified")
            verified_tasks.append(task)
        elif result is False:
            vlm_no += 1
        else:
            vlm_err += 1
            task.tags.append("vlm_error_kept")
            verified_tasks.append(task)

    print(f"Low-conf candidates: raw={len(raw_tasks)} deduped={len(deduped_tasks)}")
    print(f"VLM candidates: YES={vlm_yes} NO={vlm_no} ERR={vlm_err}")
    return verified_tasks

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate annotation task candidates")
    parser.add_argument("--project", required=True, help="Annotation project YAML")
    parser.add_argument("--task-type", choices=["bbox_review", "frame_label"], required=True)
    parser.add_argument("--max-tasks", type=int, default=150)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--shadow-threshold", type=float, default=0.55)
    parser.add_argument("--candidate-conf", type=float, default=0.03)
    parser.add_argument("--temporal-dedupe-frames", type=int, default=25)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--use-vlm", action="store_true",
                        help="Use GLM-4V-Flash VLM to verify frisbee presence in shadow frames")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_project_config(args.project)
    outputs = config["outputs"]
    if args.dry_run:
        print(f"Project: {config['project_id']}")
        print(f"Task store: {outputs['task_store']}")
        print(f"Exclude ranges: {len(config['exclude_ranges'])}")
        return 0

    if args.task_type == "bbox_review":
        generated_tasks = generate_bbox_review_tasks(
            project_config=config,
            max_tasks=args.max_tasks,
            frame_stride=args.frame_stride,
        )
    else:
        generated_tasks = generate_frame_label_tasks(
            project_config=config,
            max_tasks=args.max_tasks,
            frame_stride=args.frame_stride,
            shadow_threshold=args.shadow_threshold,
            use_vlm=args.use_vlm,
            candidate_conf=args.candidate_conf,
            temporal_dedupe_frames=args.temporal_dedupe_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
    merged_tasks = append_unique_tasks(outputs["task_store"], generated_tasks)
    print(f"Generated tasks: {len(generated_tasks)}")
    print(f"Task store: {outputs['task_store']}")
    print(f"Total tasks: {len(merged_tasks)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
