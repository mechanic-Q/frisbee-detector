"""Export reviewed annotation tasks into trainable YOLO-style assets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.annotation_core import (
    ExcludeRange,
    assert_tasks_not_leaking,
    exportable_hard_negative_tasks,
    exportable_positive_tasks,
    load_project_config,
    read_tasks,
)


def xyxy_to_yolo_label(bbox_xyxy: list[float], image_size: tuple[int, int]) -> str:
    width, height = image_size
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}\n"


def export_reviewed_tasks(
    task_store: str | Path,
    export_dir: str | Path,
    exclude_ranges: list[ExcludeRange],
    image_size: tuple[int, int] = (1920, 1080),
) -> dict:
    tasks = read_tasks(task_store)
    export_tasks = exportable_positive_tasks(tasks) + exportable_hard_negative_tasks(tasks)
    assert_tasks_not_leaking(export_tasks, exclude_ranges)

    export_path = Path(export_dir)
    positive_images = export_path / "images" / "positive"
    positive_labels = export_path / "labels" / "positive"
    hardneg_images = export_path / "images" / "hard_negative"
    hardneg_labels = export_path / "labels" / "hard_negative"
    for directory in (positive_images, positive_labels, hardneg_images, hardneg_labels):
        directory.mkdir(parents=True, exist_ok=True)

    positives = exportable_positive_tasks(tasks)
    hardnegatives = exportable_hard_negative_tasks(tasks)

    for task in positives:
        if not task.bbox_xyxy:
            continue
        shutil.copy2(task.frame_path, positive_images / f"{task.task_id}.jpg")
        (positive_labels / f"{task.task_id}.txt").write_text(
            xyxy_to_yolo_label(task.bbox_xyxy, image_size)
        )

    for task in hardnegatives:
        shutil.copy2(task.crop_path, hardneg_images / f"{task.task_id}.jpg")
        (hardneg_labels / f"{task.task_id}.txt").write_text("")

    report = {
        "task_store": str(task_store),
        "positive_count": len(positives),
        "hard_negative_count": len(hardnegatives),
        "export_dir": str(export_path),
    }
    (export_path / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reviewed annotation tasks")
    parser.add_argument("--project", required=True, help="Annotation project YAML")
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    args = parser.parse_args()

    config = load_project_config(args.project)
    outputs = config["outputs"]
    report = export_reviewed_tasks(
        task_store=outputs["task_store"],
        export_dir=outputs["export_dir"],
        exclude_ranges=config["exclude_ranges"],
        image_size=(args.image_width, args.image_height),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
