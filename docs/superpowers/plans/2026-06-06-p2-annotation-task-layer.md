# P2 通用标注任务层实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现 P2 后阴影漏检和场边误检的数据闭环：统一任务配置、严格防泄露、候选任务生成、人工复核、导出训练样本和验证报告。

**架构：** 新增轻量通用 annotation 任务层，核心逻辑放在可单测的纯 Python 模块中；Streamlit reviewer 只负责读取任务、显示资产、写回人工决策。现有 `review_labels.py`、`review_web.py`、`extract_detection_crops.py` 保持兼容，第一阶段通过新工具承接通用流程。

**技术栈：** Python 标准库、PyYAML、OpenCV、Ultralytics YOLO、Streamlit、pytest。

---

## 参考规格

- 规格文档：`docs/superpowers/specs/2026-06-06-p2-annotation-task-layer-design.md`
- wiki 记录：`/mnt/e/Agent_memory/agent-memory/queries/frisbee-annotation-tool-design-2026-06-06.md`
- 相关旧工具：`tools/review_labels.py`、`tools/review_web.py`、`tools/extract_detection_crops.py`

## 文件结构

### 新增文件

| 文件 | 职责 |
|------|------|
| `docs/conventions/naming-glossary.md` | 项目公共命名词典，约束新增配置 key、CSV / JSON 字段和跨文件变量名 |
| `configs/annotation/p2_shadow_fp_round1.yaml` | 本轮 P2 标注项目配置，不包含数据和训练产物 |
| `tools/annotation_core.py` | 数据模型、防泄露区间、任务 JSONL、复核状态、导出过滤等纯函数 |
| `tools/generate_annotation_tasks.py` | 从视频和模型生成 `bbox_review` / `frame_label` 任务 |
| `tools/review_tasks.py` | 通用 Streamlit reviewer，按 `task_type` 渲染审核界面 |
| `tools/export_annotation_tasks.py` | 从任务池导出 YOLO 正样本、hard negative 和评估报告 |
| `tests/test_annotation_core.py` | 核心数据模型、防泄露、任务读写和导出过滤单测 |
| `tests/test_generate_annotation_tasks.py` | 候选生成纯函数单测，不依赖真实 YOLO 模型 |
| `tests/test_export_annotation_tasks.py` | 导出器单测，验证标签、空标签、泄露阻断和报告 |
| `tests/test_review_tasks.py` | reviewer 纯函数单测，验证 pending 选择和复核写回 |

### 修改文件

| 文件 | 职责 |
|------|------|
| `AGENTS.md` | 增加命名词典入口和 annotation 工具使用约束 |
| `tools/extract_detection_crops.py` | 增加可选任务池输出参数，保留旧 crop 输出模式 |
| `tools/review_web.py` | 增加提示，推荐新 reviewer；旧 TP / FP 审核入口保持可用 |

### 不改文件

- 不修改 `data/`、`movie/`、`runs/`、`*.pt`。
- 不修改 P2 模型结构和训练脚本。
- 不提交任何训练输出、抽帧图片或 review 结果。

## 提交规则

- 每个任务完成后 commit。
- 禁止 `git add .`。
- 每次只 `git add` 本任务涉及的具体文件。
- 禁止提交 `data/`、`movie/`、`runs/`、`*.pt`、`.env`、`node_modules/`。
- 不启动长时间训练；只运行测试、短命令和必要的只读校验。

---

### 任务 1：建立命名词典和项目配置

**文件：**
- 创建：`docs/conventions/naming-glossary.md`
- 创建：`configs/annotation/p2_shadow_fp_round1.yaml`
- 修改：`AGENTS.md`

- [ ] **步骤 1：创建命名词典**

创建 `docs/conventions/naming-glossary.md`，内容：

```markdown
# 命名词典

## 适用范围

本词典约束公共 CLI 参数、配置 key、JSONL / CSV 字段、跨文件函数参数和文档术语。局部临时变量可以保持简洁，但不能把同一公共概念写成多个名字。

## Annotation 数据闭环术语

| 标准名 | 说明 | 避免混用 |
|--------|------|----------|
| `source_video` | 可用于挖样本的完整源视频 | `input_video`、`target_video` |
| `eval_video` | 只用于评估的视频，不进入训练 | `test_video` |
| `eval_segment` | 评估片段在源视频中的映射记录 | `test_clip` |
| `exclude_range` | 禁止候选生成和训练导出的源视频时间区间 | `skip_range`、`blacklist_range` |
| `candidate_frame` | 待人工复核的整帧候选 | 公共接口中避免只写 `frame` |
| `candidate_bbox` | 待人工复核的检测框候选 | 公共接口中避免只写 `box` |
| `hard_negative_crop` | 明确非飞盘的误检裁剪图 | `fp_crop` 只允许作局部变量 |
| `reviewer_decision` | 人工复核结论 | `result`、`label` |
| `sample_role` | 样本在训练闭环中的角色 | `type`、`kind` |

## 复核状态

| 字段值 | 说明 |
|--------|------|
| `pending` | 未复核 |
| `accepted` | 已给出有效复核结论 |
| `rejected` | 明确不采用 |
| `skipped` | 暂时跳过 |
| `frisbee` | 人工确认是飞盘 |
| `not_frisbee` | 人工确认不是飞盘 |
| `uncertain` | 不确定，不进入训练 |
```

- [ ] **步骤 2：创建 annotation 项目配置**

创建 `configs/annotation/p2_shadow_fp_round1.yaml`，内容：

```yaml
project_id: p2_shadow_fp_round1
created_at: 2026-06-06

models:
  candidate_model:
    model_name: frisbee_det_p2_game_v3
    model_path: runs/detect/frisbee_det_p2_game_v3/weights/best.pt
    conf: 0.35

source_videos:
  - source_video: movie/videoplayback_trimmed.mp4
    role: mining_source

eval_segments:
  - eval_segment_id: videoplayback_first60s
    eval_video: movie/videoplayback_first60s.mp4
    source_video: movie/videoplayback_trimmed.mp4
    source_start_sec: 0.0
    source_end_sec: 60.044
    buffer_sec: 240.0

task_generators:
  - task_type: frame_label
    strategy: shadow_candidate
    max_tasks: 150
  - task_type: bbox_review
    strategy: detection_crop
    max_tasks: 150

outputs:
  task_store: data/annotation/p2_shadow_fp_round1/tasks.jsonl
  asset_dir: data/annotation/p2_shadow_fp_round1/assets
  export_dir: data/annotation/p2_shadow_fp_round1/export
```

- [ ] **步骤 3：在 AGENTS.md 增加规则入口**

在 `AGENTS.md` 的 `Critical gotchas` 后增加：

```markdown
### Annotation naming and leakage rules
- Before adding annotation tools or public fields, read `docs/conventions/naming-glossary.md`.
- Use `source_video`, `eval_video`, `eval_segment`, `exclude_range`, `candidate_frame`, `candidate_bbox`, `hard_negative_crop`, `reviewer_decision`, and `sample_role` consistently.
- `eval_video` is evaluation-only. Training candidates must pass annotation `exclude_ranges`.
- Do not commit generated annotation assets under `data/annotation/`.
```

- [ ] **步骤 4：检查文件**

运行：

```bash
sed -n '1,220p' docs/conventions/naming-glossary.md
sed -n '1,180p' configs/annotation/p2_shadow_fp_round1.yaml
rg -n "Annotation naming" AGENTS.md
```

预期：三个命令都能显示新增内容。

- [ ] **步骤 5：Commit**

```bash
git add docs/conventions/naming-glossary.md configs/annotation/p2_shadow_fp_round1.yaml AGENTS.md
git commit -m "docs(annotation): add naming glossary and P2 config"
```

---

### 任务 2：实现 annotation 核心模型和防泄露逻辑

**文件：**
- 创建：`tools/annotation_core.py`
- 创建：`tests/test_annotation_core.py`

- [ ] **步骤 1：编写失败测试**

创建 `tests/test_annotation_core.py`，先写入：

```python
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

    assert first == second
    assert first.startswith("bbox_review_")


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
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest tests/test_annotation_core.py -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'tools.annotation_core'`。

- [ ] **步骤 3：实现核心模块**

创建 `tools/annotation_core.py`：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest tests/test_annotation_core.py -v
```

预期：PASS，5 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add tools/annotation_core.py tests/test_annotation_core.py
git commit -m "feat(annotation): add task core and leakage ranges"
```

---

### 任务 3：实现导出过滤和 review 决策规则

**文件：**
- 修改：`tools/annotation_core.py`
- 修改：`tests/test_annotation_core.py`

- [ ] **步骤 1：编写失败测试**

追加到 `tests/test_annotation_core.py`：

```python
from tools.annotation_core import (
    assert_tasks_not_leaking,
    exportable_hard_negative_tasks,
    exportable_positive_tasks,
)


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
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest tests/test_annotation_core.py -v
```

预期：FAIL，报错包含 `ImportError` 或 `NameError`，缺少新增函数。

- [ ] **步骤 3：实现过滤函数**

追加到 `tools/annotation_core.py`：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest tests/test_annotation_core.py -v
```

预期：PASS，8 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add tools/annotation_core.py tests/test_annotation_core.py
git commit -m "feat(annotation): filter exportable reviewed tasks"
```

---

### 任务 4：实现导出器

**文件：**
- 创建：`tools/export_annotation_tasks.py`
- 创建：`tests/test_export_annotation_tasks.py`

- [ ] **步骤 1：编写失败测试**

创建 `tests/test_export_annotation_tasks.py`：

```python
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest tests/test_export_annotation_tasks.py -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'tools.export_annotation_tasks'`。

- [ ] **步骤 3：实现导出器**

创建 `tools/export_annotation_tasks.py`：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest tests/test_export_annotation_tasks.py tests/test_annotation_core.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add tools/export_annotation_tasks.py tests/test_export_annotation_tasks.py
git commit -m "feat(annotation): export reviewed tasks"
```

---

### 任务 5：实现检测 crop 任务生成器

**文件：**
- 创建：`tools/generate_annotation_tasks.py`
- 创建：`tests/test_generate_annotation_tasks.py`

- [ ] **步骤 1：编写失败测试**

创建 `tests/test_generate_annotation_tasks.py`：

```python
"""Tests for annotation task generation helpers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.annotation_core import ExcludeRange
from tools.generate_annotation_tasks import build_bbox_review_task, should_keep_candidate


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
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest tests/test_generate_annotation_tasks.py -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'tools.generate_annotation_tasks'`。

- [ ] **步骤 3：实现 bbox_review 生成器和 CLI**

创建 `tools/generate_annotation_tasks.py`：

```python
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
    existing = read_tasks(task_store)
    by_id = {task.task_id: task for task in existing}
    for task in new_tasks:
        by_id.setdefault(task.task_id, task)
    merged = list(by_id.values())
    write_tasks(task_store, merged)
    return merged


def generate_bbox_review_tasks(
    project_config: dict,
    max_tasks: int,
    frame_stride: int,
) -> list[AnnotationTask]:
    import cv2
    from ultralytics import YOLO

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

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if len(tasks) >= max_tasks:
                    break
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                h, w = frame.shape[:2]
                ix1 = max(0, int(x1))
                iy1 = max(0, int(y1))
                ix2 = min(w, int(x2))
                iy2 = min(h, int(y2))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                crop = frame[iy1:iy2, ix1:ix2]
                crop_path = crop_dir / f"crop_f{frame_index:06d}_{len(tasks):04d}.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                tasks.append(
                    build_bbox_review_task(
                        source_video=source_video,
                        timestamp_sec=timestamp_sec,
                        frame_index=frame_index,
                        bbox_xyxy=[x1, y1, x2, y2],
                        crop_path=str(crop_path),
                        frame_path=str(frame_context_path),
                        model_name=model_config["model_name"],
                        model_conf=float(box.conf[0]),
                        tags=["detection_crop"],
                    )
                )

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
    merged = append_unique_tasks(outputs["task_store"], generated_tasks)
    print(f"Generated tasks: {len(generated_tasks)}")
    print(f"Task store: {outputs['task_store']}")
    print(f"Total tasks: {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest tests/test_generate_annotation_tasks.py tests/test_annotation_core.py -v
```

预期：PASS。

- [ ] **步骤 5：运行 dry-run 验证配置读取**

运行：

```bash
python3 tools/generate_annotation_tasks.py --project configs/annotation/p2_shadow_fp_round1.yaml --task-type bbox_review --dry-run
```

预期输出包含：

```text
Project: p2_shadow_fp_round1
Task store: data/annotation/p2_shadow_fp_round1/tasks.jsonl
Exclude ranges: 1
```

- [ ] **步骤 6：Commit**

```bash
git add tools/generate_annotation_tasks.py tests/test_generate_annotation_tasks.py
git commit -m "feat(annotation): add task generation core"
```

---

### 任务 6：增加阴影 frame_label 任务生成函数

**文件：**
- 修改：`tools/generate_annotation_tasks.py`
- 修改：`tests/test_generate_annotation_tasks.py`

- [ ] **步骤 1：编写失败测试**

追加到 `tests/test_generate_annotation_tasks.py`：

```python
from tools.generate_annotation_tasks import build_frame_label_task, compute_shadow_score


def test_compute_shadow_score_is_higher_for_dark_patch():
    dark_patch = [[[20, 20, 20], [25, 25, 25]]]
    bright_patch = [[[220, 220, 220], [230, 230, 230]]]

    assert compute_shadow_score(dark_patch) > compute_shadow_score(bright_patch)


def test_build_frame_label_task_uses_positive_candidate_role():
    task = build_frame_label_task(
        source_video="movie/full.mp4",
        timestamp_sec=500.0,
        frame_index=12500,
        frame_path="data/annotation/assets/frame.jpg",
        model_name="frisbee_det_p2_game_v3",
        tags=["shadow"],
    )

    assert task.task_type == "frame_label"
    assert task.sample_role == "positive_candidate"
    assert task.review_status == "pending"
    assert task.tags == ["shadow"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest tests/test_generate_annotation_tasks.py -v
```

预期：FAIL，缺少 `compute_shadow_score` 和 `build_frame_label_task`。

- [ ] **步骤 3：实现 frame_label 辅助函数**

追加到 `tools/generate_annotation_tasks.py`：

```python
def compute_shadow_score(pixel_rows) -> float:
    values: list[float] = []
    for row in pixel_rows:
        for pixel in row:
            if len(pixel) >= 3:
                values.append((float(pixel[0]) + float(pixel[1]) + float(pixel[2])) / 3.0)
    if not values:
        return 0.0
    mean_brightness = sum(values) / len(values)
    return max(0.0, min(1.0, 1.0 - mean_brightness / 255.0))


def build_frame_label_task(
    source_video: str,
    timestamp_sec: float,
    frame_index: int,
    frame_path: str,
    model_name: str,
    tags: list[str] | None = None,
) -> AnnotationTask:
    task_id = make_task_id("frame_label", source_video, timestamp_sec, frame_index, None)
    return AnnotationTask(
        task_id=task_id,
        task_type="frame_label",
        source_video=source_video,
        timestamp_sec=timestamp_sec,
        frame_index=frame_index,
        sample_role="positive_candidate",
        review_status="pending",
        reviewer_decision="",
        frame_path=frame_path,
        model_name=model_name,
        tags=tags or [],
    )


def generate_frame_label_tasks(
    project_config: dict,
    max_tasks: int,
    frame_stride: int,
    shadow_threshold: float = 0.55,
) -> list[AnnotationTask]:
    import cv2

    model_config = project_config["models"]["candidate_model"]
    source_video = project_config["source_videos"][0]["source_video"]
    outputs = project_config["outputs"]
    frame_dir = Path(outputs["asset_dir"]) / "shadow_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source_video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_index = 0
    tasks: list[AnnotationTask] = []

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

        shadow_score = compute_shadow_score(frame.tolist())
        if shadow_score < shadow_threshold:
            continue

        frame_path = frame_dir / f"shadow_f{frame_index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        tasks.append(
            build_frame_label_task(
                source_video=source_video,
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                frame_path=str(frame_path),
                model_name=model_config["model_name"],
                tags=["shadow"],
            )
        )

    cap.release()
    return tasks
```

- [ ] **步骤 4：把 CLI 接入 frame_label**

把任务 5 中的 parser 修改为：

```python
parser.add_argument("--task-type", choices=["bbox_review", "frame_label"], required=True)
parser.add_argument("--shadow-threshold", type=float, default=0.55)
```

把生成分支修改为：

```python
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
    )
merged = append_unique_tasks(outputs["task_store"], generated_tasks)
```

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
python3 -m pytest tests/test_generate_annotation_tasks.py -v
```

预期：PASS。

- [ ] **步骤 6：运行 dry-run 验证 frame_label 配置读取**

运行：

```bash
python3 tools/generate_annotation_tasks.py --project configs/annotation/p2_shadow_fp_round1.yaml --task-type frame_label --dry-run
```

预期输出包含：

```text
Project: p2_shadow_fp_round1
Exclude ranges: 1
```

- [ ] **步骤 7：Commit**

```bash
git add tools/generate_annotation_tasks.py tests/test_generate_annotation_tasks.py
git commit -m "feat(annotation): add shadow frame task generation"
```

---

### 任务 7：实现通用 reviewer 的纯函数和 Streamlit 入口

**文件：**
- 创建：`tools/review_tasks.py`
- 创建：`tests/test_review_tasks.py`

- [ ] **步骤 1：编写失败测试**

创建 `tests/test_review_tasks.py`：

```python
"""Tests for generic annotation task reviewer helpers."""

import os
import sys

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
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest tests/test_review_tasks.py -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'tools.review_tasks'`。

- [ ] **步骤 3：实现 reviewer 纯函数和最小 UI**

创建 `tools/review_tasks.py`：

```python
"""Generic Streamlit reviewer for annotation tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.annotation_core import AnnotationTask, load_project_config, read_tasks, write_tasks


def next_pending_task(tasks: list[AnnotationTask]) -> AnnotationTask | None:
    for task in tasks:
        if task.review_status == "pending":
            return task
    return None


def apply_review_decision(
    tasks: list[AnnotationTask],
    task_id: str,
    reviewer_decision: str,
) -> list[AnnotationTask]:
    valid = {"frisbee", "not_frisbee", "uncertain", "skipped", "rejected"}
    if reviewer_decision not in valid:
        raise ValueError(f"Invalid reviewer_decision: {reviewer_decision}")
    for task in tasks:
        if task.task_id != task_id:
            continue
        if reviewer_decision == "skipped":
            task.review_status = "skipped"
            task.reviewer_decision = ""
        elif reviewer_decision == "rejected":
            task.review_status = "rejected"
            task.reviewer_decision = ""
        else:
            task.review_status = "accepted"
            task.reviewer_decision = reviewer_decision
    return tasks


def get_args():
    parser = argparse.ArgumentParser(description="Review annotation tasks")
    parser.add_argument("--project", required=True, help="Annotation project YAML")
    return parser.parse_known_args()[0]


def main() -> None:
    import streamlit as st

    args = get_args()
    config = load_project_config(args.project)
    task_store = Path(config["outputs"]["task_store"])
    tasks = read_tasks(task_store)

    st.set_page_config(page_title="Annotation Task Review", layout="wide")
    st.title("Annotation Task Review")
    st.caption(str(task_store))

    pending = sum(1 for task in tasks if task.review_status == "pending")
    accepted = sum(1 for task in tasks if task.review_status == "accepted")
    skipped = sum(1 for task in tasks if task.review_status == "skipped")
    st.write({"pending": pending, "accepted": accepted, "skipped": skipped})

    task = next_pending_task(tasks)
    if task is None:
        st.success("All tasks reviewed")
        return

    st.subheader(f"{task.task_type}: {task.task_id}")
    st.write(
        {
            "source_video": task.source_video,
            "timestamp_sec": task.timestamp_sec,
            "frame_index": task.frame_index,
            "model_name": task.model_name,
            "model_conf": task.model_conf,
            "tags": task.tags,
        }
    )

    if task.task_type == "bbox_review" and task.crop_path:
        st.image(task.crop_path, caption="candidate crop", width=420)
        if task.frame_path:
            st.image(task.frame_path, caption="source frame context", use_container_width=True)
    elif task.task_type == "frame_label" and task.frame_path:
        st.image(task.frame_path, caption="candidate frame", use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    choices = [
        (col1, "frisbee", "Frisbee"),
        (col2, "not_frisbee", "Not Frisbee"),
        (col3, "uncertain", "Uncertain"),
        (col4, "skipped", "Skip"),
        (col5, "rejected", "Reject"),
    ]
    for column, decision, label in choices:
        with column:
            if st.button(label, use_container_width=True):
                updated = apply_review_decision(tasks, task.task_id, decision)
                write_tasks(task_store, updated)
                st.rerun()


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest tests/test_review_tasks.py -v
```

预期：PASS。

- [ ] **步骤 5：验证 Streamlit 脚本可解析参数**

运行：

```bash
python3 -m py_compile tools/review_tasks.py
```

预期：无输出，退出码 0。

- [ ] **步骤 6：Commit**

```bash
git add tools/review_tasks.py tests/test_review_tasks.py
git commit -m "feat(annotation): add generic task reviewer"
```

---

### 任务 8：给旧 crop 工具增加任务池输出兼容入口

**文件：**
- 修改：`tools/extract_detection_crops.py`
- 修改：`tools/review_web.py`
- 修改：`tests/test_generate_annotation_tasks.py`

- [ ] **步骤 1：编写失败测试**

追加到 `tests/test_generate_annotation_tasks.py`：

```python
from tools.generate_annotation_tasks import append_unique_tasks


def test_append_unique_tasks_deduplicates_by_task_id(tmp_path):
    task = build_bbox_review_task(
        source_video="movie/full.mp4",
        timestamp_sec=400.0,
        frame_index=10000,
        bbox_xyxy=[100.0, 120.0, 140.0, 160.0],
        crop_path="assets/crop.jpg",
        frame_path="assets/frame.jpg",
        model_name="frisbee_det_p2_game_v3",
        model_conf=0.77,
    )
    task_store = tmp_path / "tasks.jsonl"

    merged = append_unique_tasks(task_store, [task, task])

    assert len(merged) == 1
```

- [ ] **步骤 2：运行测试确认当前通过或失败**

运行：

```bash
python3 -m pytest tests/test_generate_annotation_tasks.py -v
```

预期：PASS。如果 `append_unique_tasks` 已在任务 5 实现，测试通过；如果实现者遗漏，补回任务 5 中的函数。

- [ ] **步骤 3：修改 `tools/extract_detection_crops.py` 参数**

在 argparse 增加：

```python
parser.add_argument("--task-store", default=None, help="Optional JSONL task store for bbox_review tasks")
parser.add_argument("--frame-output", default=None, help="Optional directory for source frame context images")
```

在文件顶部 import：

```python
from tools.generate_annotation_tasks import append_unique_tasks, build_bbox_review_task
```

在主循环中保存 crop 后，如果 `args.task_store` 存在，创建任务：

```python
generated_tasks = []
frame_output_dir = Path(args.frame_output) if args.frame_output else output_dir / "frames"
frame_output_dir.mkdir(parents=True, exist_ok=True)
```

每个 detection 保存 crop 后追加：

```python
frame_context_path = frame_output_dir / f"frame_f{frame_idx:05d}_{video_name}.jpg"
if not frame_context_path.exists():
    cv2.imwrite(str(frame_context_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

if args.task_store:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    timestamp_sec = frame_idx / fps
    generated_tasks.append(
        build_bbox_review_task(
            source_video=str(video_path),
            timestamp_sec=timestamp_sec,
            frame_index=frame_idx,
            bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
            crop_path=str(output_dir / fname),
            frame_path=str(frame_context_path),
            model_name=model_path.stem,
            model_conf=conf,
            tags=["detection_crop"],
        )
    )
```

循环结束后写入：

```python
if args.task_store:
    merged = append_unique_tasks(args.task_store, generated_tasks)
    print(f"Task store: {args.task_store}")
    print(f"Generated tasks: {len(generated_tasks)}")
    print(f"Total tasks: {len(merged)}")
```

- [ ] **步骤 4：修改 `tools/review_web.py` 提示**

在标题下增加：

```python
st.info(
    "For new P2 annotation work, prefer tools/review_tasks.py with an annotation project YAML. "
    "This legacy reviewer remains available for older crop-only review_results.csv workflows."
)
```

- [ ] **步骤 5：运行测试和编译检查**

运行：

```bash
python3 -m pytest tests/test_generate_annotation_tasks.py -v
python3 -m py_compile tools/extract_detection_crops.py tools/review_web.py
```

预期：测试 PASS，编译无输出。

- [ ] **步骤 6：Commit**

```bash
git add tools/extract_detection_crops.py tools/review_web.py tests/test_generate_annotation_tasks.py
git commit -m "feat(annotation): bridge crop extractor to task store"
```

---

### 任务 9：补充验证命令和最终文档说明

**文件：**
- 修改：`docs/conventions/naming-glossary.md`
- 修改：`docs/superpowers/plans/2026-06-06-p2-annotation-task-layer.md`

- [ ] **步骤 1：更新命名词典的使用命令**

在 `docs/conventions/naming-glossary.md` 末尾追加：

````markdown
## 常用命令

生成任务配置 dry-run：

```bash
python3 tools/generate_annotation_tasks.py --project configs/annotation/p2_shadow_fp_round1.yaml --task-type bbox_review --dry-run
```

启动通用 reviewer：

```bash
streamlit run tools/review_tasks.py -- --project configs/annotation/p2_shadow_fp_round1.yaml
```

导出已复核样本：

```bash
python3 tools/export_annotation_tasks.py --project configs/annotation/p2_shadow_fp_round1.yaml
```
````

- [ ] **步骤 2：运行全量相关测试**

运行：

```bash
python3 -m pytest tests/test_annotation_core.py tests/test_generate_annotation_tasks.py tests/test_export_annotation_tasks.py tests/test_review_tasks.py -v
```

预期：PASS。

- [ ] **步骤 3：运行项目测试**

运行：

```bash
python3 -m pytest tests/ -v
```

预期：PASS。如果失败，先判断是否与本任务相关；只修复本任务引入的问题，不修改无关历史失败。

- [ ] **步骤 4：运行数据排除 dry-run**

运行：

```bash
python3 tools/generate_annotation_tasks.py --project configs/annotation/p2_shadow_fp_round1.yaml --task-type bbox_review --dry-run
```

预期输出包含：

```text
Project: p2_shadow_fp_round1
Exclude ranges: 1
```

- [ ] **步骤 5：确认没有误暂存数据**

运行：

```bash
git status --short
```

预期：本轮只包含代码、配置、文档、测试改动；不能出现 `data/`、`movie/`、`runs/`、`*.pt`、`.env`、`node_modules/` 进入暂存区。

- [ ] **步骤 6：Commit**

```bash
git add docs/conventions/naming-glossary.md docs/superpowers/plans/2026-06-06-p2-annotation-task-layer.md
git commit -m "docs(annotation): document task workflow commands"
```

---

## 最终验收

实现完成后运行：

```bash
python3 -m pytest tests/ -v
python3 tools/generate_annotation_tasks.py --project configs/annotation/p2_shadow_fp_round1.yaml --task-type bbox_review --dry-run
python3 -m py_compile tools/annotation_core.py tools/generate_annotation_tasks.py tools/review_tasks.py tools/export_annotation_tasks.py
```

验收标准：

- 所有新增单测通过。
- 现有测试没有因 annotation 改动产生新失败。
- dry-run 能正确读取 `configs/annotation/p2_shadow_fp_round1.yaml` 并显示 1 个排除区间。
- `review_tasks.py` 支持 `frisbee`、`not_frisbee`、`uncertain`、`skipped`、`rejected`。
- 导出器不会导出 `uncertain`，并会阻断落入 `exclude_ranges` 的任务。
- 未提交任何 `data/`、`movie/`、`runs/`、`*.pt`、`.env`、`node_modules/`。

## 实现后暂停点

完成上面代码和测试后停下，向用户报告：

- 生成了哪些工具。
- 测试结果。
- 是否发现旧工具命名或结构问题。
- 下一步应由用户决定是否开始生成真实候选任务。

不要自动启动长时间训练。
