# P2 后通用标注任务层设计

> **目标：** 为 P2 后的阴影漏检和场边误检问题建立一套可复用的数据闭环：候选生成、严格防泄露、人工复核、导出训练样本、重训复测。  
> **核心选择：** 在现有工具基础上新增轻量通用任务层，不直接部署 CVAT / Label Studio，也不继续为每个场景重写脚本。  
> **当前训练策略：** 采用 C 方案：阴影漏检正样本和误检 hard negative 双通道小批量闭环。

## 背景

P2 重训后，模型输出比上一阶段更干净，但仍有两个主要问题：

- 阴影区域中的飞盘容易漏检。
- 场边物品、人、白色小物体仍可能被误识别为飞盘。

已有工具能覆盖部分工作：

| 现有工具 | 当前能力 | 问题 |
|----------|----------|------|
| `tools/review_labels.py` | 审核帧级标签，支持模型 overlay | 缺少本轮统一来源元数据和防泄露入口 |
| `tools/extract_detection_crops.py` | 从视频和模型提取检测 crop | 字段偏一次性，不能表达评估禁区、任务类型和复核状态 |
| `tools/review_web.py` | 对 crop 做 TP / FP 复核 | 决策粒度太窄，缺少 `uncertain` 和通用任务模型 |

如果继续按场景增加脚本，阴影漏检、误检 crop、未来轨迹断点都会产生各自的 CSV、字段和导出逻辑。长期会导致变量名不统一、流程不可复用、数据泄露检查难以自动化。

## 设计选择

本轮选择新增一层轻量通用标注任务系统：

```text
annotation_project.yaml
        |
        v
候选生成器 -> 泄露过滤 -> tasks.jsonl -> 通用 reviewer
        |                                               |
        v                                               v
候选资产目录                                      人工复核结果
        |                                               |
        +--------------------> 导出器 ------------------+
                                |
                                v
                         YOLO 标签 / hard negative / 评估报告
```

这不是重写标注平台。它是把现有脚本收敛成三段式流程：

1. **生成任务：** 从视频、模型、阴影规则和采样策略生成候选。
2. **审核任务：** 人工确认飞盘、非飞盘或不确定。
3. **导出任务：** 只把通过复核且不泄露的样本导出到训练数据。

## 为什么不直接用 CVAT 或 Label Studio

CVAT 和 Label Studio 是成熟平台，但当前项目不需要完整平台能力。当前需求是小批量、高频的数据闭环：找问题样本、人工复核、重训、复测。直接接入完整平台会引入部署、权限、存储、导入导出、版本升级和 UI 适配成本，容易把模型数据修正任务变成平台集成工程。

本项目更适合采用中间路线：

- 保留现有 Streamlit 工具和 Python 脚本。
- 借鉴成熟平台的数据模型和流程概念。
- 用统一任务池和配置文件解决复用、来源追踪和防泄露问题。
- 保留未来导出到 CVAT、Label Studio 或 MOT 格式的可能。

## 外部项目借鉴

| 项目 | 借鉴点 | 本项目采用方式 |
|------|--------|----------------|
| [CVAT Track Mode](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/track-mode-basics/) | track、keyframe、interpolation、outside、merge / split | 为未来 `track_segment` 任务保留 `track_id` 和关键帧概念 |
| [CVAT AI Tools](https://docs.cvat.ai/docs/annotation/auto-annotation/ai-tools/) | 模型辅助标注和人工修正 | P2 模型先生成候选，人工复核后导出 |
| [Label Studio Video Object Detector](https://labelstud.io/templates/video_object_detector) | 视频检测任务配置化、结果结构化 | 用 `annotation_project.yaml` 描述任务输入和渲染模式 |
| [Label Studio Predictions](https://labelstud.io/guide/predictions) | 预标注与人工结果分离，记录模型版本和分数 | 保存 `model_name`、`model_conf`、候选 bbox 和人工决策 |
| [SoccerNet Tracking](https://github.com/SoccerNet/sn-tracking) | 体育视频跟踪关注 frame、bbox、identity 和评估协议 | 保留 `frame_index`、`timestamp_sec`、未来 `track_id` |
| [TeamTrack](https://github.com/AtomScott/TeamTrack) | 多运动场景、MOT 格式、轨迹数据分层组织 | 分离检测训练标签、hard negative、未来轨迹导出 |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | 复杂体育场景下按 split 组织，避免评估污染 | 强制登记评估片段和训练候选来源 |

## 范围

### 本轮包含

- 通用项目配置：`annotation_project.yaml`。
- 通用任务池：`tasks.jsonl`。第一阶段使用 JSONL，数据量显著变大后再考虑迁移到 SQLite。
- 两类首批任务：
  - `frame_label`：阴影漏检正样本审核。
  - `bbox_review`：误检 crop / bbox hard negative 审核。
- 严格防泄露：
  - `eval_segments` 显式记录评估片段在源视频中的区间。
  - `exclude_ranges` 自动生成，候选生成前必须过滤。
  - 无法定位评估片段来源时，禁止从该源视频挖训练样本。
- 人工复核状态：
  - `frisbee`
  - `not_frisbee`
  - `uncertain`
- 导出器：
  - 确认飞盘导出 YOLO 正样本标签。
  - 确认非飞盘导出 bbox / crop 级 hard negative。
  - 不确定样本不进入训练。
- 命名词典和代码审计：
  - 建立公共术语和变量职责说明。
  - 只对当前 P2 数据闭环必须接触的代码做小范围整理。

### 本轮不包含

- 不部署 CVAT、Label Studio 或其他完整标注平台。
- 不做全仓库变量名大重构。
- 不修改 P2 模型结构。
- 不启动长时间训练。
- 不把固定评估片段加入训练。
- 不实现完整多目标跟踪标注系统；只为未来 `track_segment` 保留字段。

## 数据泄露规则

数据泄露是本轮硬约束。训练数据污染比例很小也不能忽视，因为关键问题不是训练集占比，而是评估可信度。

### 评估片段元数据

每个评估片段必须登记：

```yaml
eval_segments:
  - eval_segment_id: videoplayback_first60s
    eval_video: movie/videoplayback_first60s.mp4
    source_video: movie/videoplayback_trimmed.mp4
    source_start_sec: 0.0
    source_end_sec: 60.044
    buffer_sec: 240.0
```

由此自动生成：

```yaml
exclude_ranges:
  - source_video: movie/videoplayback_trimmed.mp4
    start_sec: 0.0
    end_sec: 300.044
    reason: eval_segment_buffer
    eval_segment_id: videoplayback_first60s
```

当前已用低分辨率 1 fps 帧哈希做只读校验：`movie/videoplayback_first60s.mp4` 与 `movie/videoplayback_trimmed.mp4` 的前 60 秒一致。因此当前可以把 `first60s` 映射到源视频开头，但未来不能靠文件名猜测。

### 硬规则

- `eval_video` 只评估，不抽帧、不打标签、不进入训练。
- 所有候选生成器必须先应用 `exclude_ranges`。
- 如果 `eval_video` 无法定位到 `source_video` 的真实时间区间，不能从这个 `source_video` 挖训练样本。
- 训练导出前必须再运行泄露检查，检查来源视频、时间戳和评估区间。
- 模型改进结论必须同时看：
  - `movie/videoplayback_first60s.mp4` 开发复测。
  - 至少一个未参与挖样本的独立片段 sanity check。

## 数据模型

### `annotation_project.yaml`

配置职责：描述本轮标注项目，不保存人工结果。

建议字段：

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

### 任务字段

所有任务共享以下字段：

| 字段 | 说明 |
|------|------|
| `task_id` | 稳定任务 ID，不依赖列表顺序 |
| `task_type` | `frame_label`、`bbox_review`、`quality_check`、`track_segment` |
| `source_video` | 候选来自哪个完整源视频 |
| `timestamp_sec` | 源视频时间戳 |
| `frame_index` | 源视频帧号 |
| `eval_segment_id` | 若与评估片段相关，记录来源；普通训练候选为空 |
| `bbox_xyxy` | 候选框，像素坐标；整帧任务可为空 |
| `crop_path` | crop 任务的裁剪图路径 |
| `frame_path` | 整帧任务的图片路径 |
| `model_name` | 生成候选的模型名 |
| `model_conf` | 候选置信度 |
| `sample_role` | `positive_candidate`、`hard_negative_candidate`、`quality_check` |
| `review_status` | `pending`、`accepted`、`rejected`、`skipped` |
| `reviewer_decision` | `frisbee`、`not_frisbee`、`uncertain` |
| `tags` | `shadow`、`sideline`、`person`、`white_object` 等 |
| `created_at` | 候选生成时间 |
| `reviewed_at` | 人工复核时间 |

### 任务类型

| 任务类型 | 用途 | 审核动作 | 导出结果 |
|----------|------|----------|----------|
| `frame_label` | 阴影漏检正样本 | 补框、修框、删除误框、跳过 | 确认飞盘后导出 YOLO 正样本 |
| `bbox_review` | 场边误检 hard negative | 标记 `frisbee` / `not_frisbee` / `uncertain` | `not_frisbee` 导出 hard negative |
| `quality_check` | 抽样复核已标样本 | 通过 / 驳回 / 不确定 | 生成质量报告 |
| `track_segment` | 未来轨迹审核 | 审核 track、断点、outside | 未来导出 MOT / 轨迹数据 |

## 工作流

### 1. 生成候选

`frame_label` 队列：

- 从安全区间抽帧。
- 用亮度、颜色、场地区域或低置信模型输出筛选阴影候选。
- 保存整帧和候选提示信息。

`bbox_review` 队列：

- 在安全区间运行当前 P2 模型。
- 提取检测框 crop，并保存原图上下文。
- 将候选写入统一任务池。

所有候选进入任务池之前必须通过泄露过滤。

### 2. 人工复核

通用 reviewer 根据 `task_type` 渲染不同界面：

- `frame_label` 显示原图、模型候选框、阴影提示，人工补框或修框。
- `bbox_review` 显示 crop 和原图上下文，人工标记飞盘、非飞盘或不确定。

复核结果只更新任务池，不直接写训练集。

### 3. 导出训练样本

导出器读取任务池，执行：

- 泄露检查。
- 决策过滤。
- 文件复制和标签生成。
- 导出报告生成。

导出规则：

- `frame_label + frisbee`：导出图片和 YOLO 标签。
- `bbox_review + not_frisbee`：导出 hard negative crop 和空标签。
- `uncertain`：保留记录，但不训练。
- `skipped` / `rejected`：不训练。

### 4. 训练与复测

第一轮控制规模：每类约 50-150 个有效样本，避免一次性引入过多偏差。

复测要求：

- 使用与上一阶段一致的视频、阈值和统计方式。
- `first60s` 只作为开发复测片段。
- 独立片段用于 sanity check。
- 不接受靠整体压低检测率换来的“干净”。

## 成功标准

### 数据流程成功

- 候选生成器不会产出评估禁区内任务。
- 每个导出样本都能追溯到 `source_video`、`timestamp_sec`、`frame_index` 和人工决策。
- `uncertain` 样本不会进入训练。
- 导出前泄露检查能阻止违规样本。

### 模型效果成功

- 阴影区域真实飞盘召回提升。
- 场边物品和人的误检下降。
- 总检测率不能明显塌缩。
- `first60s` 和独立片段的趋势一致。

### 工程成功

- 新增任务类型时，不需要重写审核系统。
- 公共字段名、配置 key 和导出列名使用命名词典。
- 当前旧工具保留兼容入口，现有 review 结果不被破坏。

## 命名规范

本轮同时建立项目命名规则：新代码优先复用命名词典，不为同一概念创建多个变量名。

首批标准术语：

| 标准名 | 说明 | 避免混用 |
|--------|------|----------|
| `source_video` | 可用于挖样本的完整源视频 | `input_video`、`target_video` |
| `eval_video` | 只用于评估的视频 | `test_video` 在训练上下文中容易歧义 |
| `eval_segment` | 评估片段在源视频中的映射记录 | `test_clip` |
| `exclude_range` | 禁止进入候选和训练的时间区间 | `skip_range`、`blacklist_range` |
| `candidate_frame` | 待复核整帧候选 | `frame`、`image` |
| `candidate_bbox` | 待复核检测框候选 | `box` 在跨文件接口中过短 |
| `hard_negative_crop` | 明确非飞盘的误检裁剪图 | `fp_crop` 可作为局部变量，不作为公共字段 |
| `reviewer_decision` | 人工复核决策 | `result`、`label` |
| `sample_role` | 样本在训练闭环中的角色 | `type`、`kind` |

命名整理策略：

- 先做全项目只读审计。
- 只改当前 P2 数据闭环必须接触的代码。
- 公共 CLI 参数、CSV 列、配置 key 如需改名，保留兼容别名。
- 局部临时变量不强制进入词典。

## 测试与验证

### 单元测试

- `eval_segments` 生成 `exclude_ranges`。
- 时间戳落入排除区间时，候选被过滤。
- 无法定位评估片段时，候选生成失败并给出明确错误。
- `uncertain` 不导出到训练数据。
- `not_frisbee` 只导出为 hard negative，不导出正样本标签。

### 集成验证

- 构造小型任务池，跑完整导出流程。
- 校验导出目录中文件和标签数量。
- 校验导出报告统计与任务池一致。
- 对当前 `videoplayback_first60s` 映射执行泄露检查。

### 人工检查

- 随机抽查导出的正样本和 hard negative。
- 确认 crop 有原图上下文可追溯。
- 确认阴影标签来自评估禁区之外。

## 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| 任务系统过度抽象 | 拖慢当前训练闭环 | 第一阶段只实现 `frame_label` 和 `bbox_review` |
| 评估片段定位错误 | 防泄露失效 | 必须记录 `eval_segments`，必要时用帧哈希验证 |
| 大范围重命名破坏旧脚本 | 现有流程断裂 | 只改当前闭环相关代码，公共接口保留兼容 |
| hard negative 太多 | 召回下降 | 小批量导入，每轮 50-150 个有效样本 |
| 阴影候选噪声大 | 人工复核负担增加 | 先做候选排序和上限控制 |
| Streamlit reviewer 能力不足 | 审核效率受限 | 第一阶段复用，任务量变大后再考虑导出到 CVAT / Label Studio |

## 实现顺序建议

后续实现计划应按以下顺序拆分：

1. 建立命名词典和只读代码审计报告。
2. 定义 `annotation_project.yaml` 和任务数据模型。
3. 实现 `eval_segments` / `exclude_ranges` 防泄露过滤。
4. 改造 detection crop 候选生成，输出 `bbox_review` 任务。
5. 增加阴影候选生成，输出 `frame_label` 任务。
6. 改造 reviewer，支持通用任务池和 `uncertain`。
7. 实现导出器和泄露检查。
8. 运行小样本闭环，生成评估表。

## 参考记录

- llm-wiki：`/mnt/e/Agent_memory/agent-memory/queries/frisbee-annotation-tool-design-2026-06-06.md`
- 当前计划评估：`docs/superpowers/plans/2026-06-05-current-plan-evaluation.md`
- 现有工具：`tools/review_labels.py`、`tools/review_web.py`、`tools/extract_detection_crops.py`
