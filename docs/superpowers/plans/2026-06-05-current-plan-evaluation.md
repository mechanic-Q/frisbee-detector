# 当前 Superpowers 计划评估 - 2026-06-05

本文档把当前激活的 Superpowers 计划映射到仓库现状。
评估只使用已有本地 artifact；`data/` 和 `runs/` 仍然是未跟踪内容，
禁止提交。

## 当前计划状态

| 计划项 | 当前状态 | 证据 | 下一步 |
| --- | --- | --- | --- |
| `review_labels.py` 通用 review 工具 | 已实现 | `tools/review_labels.py`、`tools/_label_utils.py`、`tests/test_label_utils.py` 存在；`label_frames.py` 已移除 | 除非 UI review 发现问题，否则保持现状 |
| GSAM label review | 本地已清理 | `review_result.json` 当前为 46 accept、68 reject、0 skip、114 reviewed；accept/reject 无重叠 | 不提交 `data/`；训练决策使用本地清理后的 label |
| 合并 reviewed `game1080` label | 已被 product/pool merge 路径替代 | `frisbee-data/products/v1.yaml` 包含 `game1080`；`data/datasets/frisbee_merged/images/train` 中有 200 张 `game1080_*` 图片 | 继续以 `product YAML` 路径作为数据源事实 |
| P2 模型训练 | 已超过原始 v1 计划 | `runs/detect/frisbee_det_p2_game_v2/weights/best.pt` 存在 | 重训前先评估 |
| 1080p first60 评估 | 已完成正式同命令 benchmark | 已记录 `frisbee_det_p2_game_v2`、`frisbee_det_s_v7`、`frisbee_det_s_v3` 在 `videoplayback_first60s` 上的结果 | 选择默认模型前可继续补充新模型对比 |
| 数据集校验 | 代码已修复 | 空 YOLO label 文件现在会被视为有效负样本 / 背景样本 | 训练前运行 `tools/verify_dataset.py` |

## Review Result 清理

本地 `data/datasets/game1080/review_result.json` 之前存在重复项，
并且有 1 个 accept/reject 冲突：

| 问题 | 处理方式 |
| --- | --- |
| accept/reject 中存在重复条目 | 保持原顺序去重 |
| `frame_0112` 同时出现在 accept 和 reject | 保留为 accept，因为 `labels/frame_0112.txt` 非空 |
| `frame_0002` 有非空 reviewed label，但不在 accept 中 | 加入 accept，使状态与 reviewed label 输出一致 |

本地最终 review 状态：

| 桶 | 数量 |
| --- | ---: |
| Accept | 46 |
| Reject | 68 |
| Skip | 0 |
| Unique reviewed | 114 |

## 既有 Artifact 评估表

这些指标来自已有 `runs/detect/runs/eval/*/labels` 文件：
统计有 detection 的 frame 数，以及每个 label 文件中的 detection 行数。
这些结果适合判断方向，但 label 目录没有记录完整运行参数；
正式模型选择仍应使用显式命令重新跑 benchmark。

| 模型 artifact | 视频 | 有 detection 的帧数 | Detection rate | Detections/frame | 计划解读 |
| --- | --- | ---: | ---: | ---: | --- |
| `frisbee_det_p2_game_v2` | `videoplayback_first60s.mp4` | 309 / 3646 | 8.5% | 0.10 | P2 v2 在 1080p holdout artifact 上偏弱 |
| `frisbee_det_s_v7-3` | `videoplayback_first60s.mp4` | 797 / 3646 | 21.9% | 0.26 | 比 P2 v2 更好，但 recall 仍低 |
| `frisbee_det_s_v3-10` | `25866279684-1-192_55-56min.mp4` | 1048 / 1500 | 69.9% | 1.86 | Recall 高，但 detection 数量多，FP 风险仍在 |
| `frisbee_det_s_v7_quick` | `25866279684-1-192_55-56min.mp4` | 803 / 1500 | 53.5% | 0.86 | 在该 clip 上达到 v7 quick recall 目标 |
| `frisbee_det_s_v7-2` | `25866279684-1-192_55-56min.mp4` | 311 / 1500 | 20.7% | 0.22 | Production v7 artifact 过于保守 |
| `frisbee_det_s_v3-11` | `clip_20-23min.mp4` | 2690 / 4378 | 61.4% | 1.28 | v3 baseline 达到 recall 目标，但 FP 风险仍在 |
| `frisbee_det_s_v7_quick-2` | `clip_20-23min.mp4` | 2047 / 4378 | 46.8% | 0.67 | 略低于 v7 quick recall 目标 |
| `frisbee_det_s_v7` | `clip_20-23min.mp4` | 516 / 4378 | 11.8% | 0.12 | Production v7 artifact 过于保守 |

## 正式同命令 Benchmark

2026-06-05 已使用同一个视频和同一个置信度阈值完成：

```bash
python3 inference/predict_video.py --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
python3 inference/predict_video.py --model runs/detect/frisbee_det_s_v7/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
python3 inference/predict_video.py --model runs/detect/frisbee_det_s_v3/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
```

| 模型 artifact | 有 detection 的帧数 | Detection rate | Detections/frame | 保存的 eval 目录 | 解读 |
| --- | ---: | ---: | ---: | --- | --- |
| `frisbee_det_p2_game_v2` | 309 / 3598 | 8.6% | 0.10 | `runs/detect/runs/eval/frisbee_det_p2_game_v2-2` | P2 v2 在 1080p holdout 上仍然偏弱 |
| `frisbee_det_s_v7` | 797 / 3598 | 22.2% | 0.27 | `runs/detect/runs/eval/frisbee_det_s_v7-4` | 比 P2 v2 更好，但 recall 仍低 |
| `frisbee_det_s_v3` | 1217 / 3598 | 33.8% | 0.41 | `runs/detect/runs/eval/frisbee_det_s_v3-14` | 这 3 个模型里 recall 最好，但仍低于早期 720p v3 recall |

## 推荐的下一步计划

P2 仍接近 artifact 推导出的 8.5% detection rate，
所以 Superpowers 的下一步是重训 cleaned-label P2 模型。
这是长时间训练任务，不能在没有明确授权的情况下启动。

## P2 重训 Preflight

在启动任何长时间训练前已完成：

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 分支 | OK | `git branch --show-current` -> `feat/p2-detector` |
| 数据集有效性 | OK | `python3 tools/verify_dataset.py configs/frisbee_merged.yaml` -> `Total issues: 0` |
| `tmux` | OK | `tmux 3.4` |
| GPU 可见性 | sandbox 外 OK | `nvidia-smi` -> RTX 5080，16 GB；escalated PyTorch 可见 CUDA device 0 |
| P2 YAML | OK | `YOLO("yolov8s-p2.yaml")` 可构造 `DetectionModel` |
| `game1080` 样本 | OK | `data/datasets/frisbee_merged/images/train` 中有 200 张 `game1080_*` 图片：46 个正样本，154 个负样本 |
| 测试视频全帧 hard negatives | 训练前必须修复 | 当前 merged data 包含 202 张空标签 `hardneg_*` 全帧图片，尺寸 1280x720，其中 train 有 165 张，且文件名来自 `55-56min` / `clip_20-23min` 测试视频 |

merge 脚本现在已经支持 `product` 级别的 `exclude`。
本地 `product` 配置已经调整为排除 `hardneg`，
但还没有重建 merged dataset，因为这会替换已有生成数据。

## Cleaned Dataset 只读预估

以下结果来自 `frisbee-data/products/v1.yaml` 和 `/mnt/e/frisbee-pool`
的只读 preflight；没有调用 `merge_from_product()`，
也没有重写 `data/datasets/frisbee_merged`。

当前本地 `product` 配置：

| 字段 | 值 |
| --- | --- |
| `exclude` | `hardneg` |
| 有效 `train_only` | `game1080` |
| 排除后总样本数 | 4948 |
| 排除后是否仍有 `hardneg` | false |

排除后的 source 计数：

| Source | Total | Positive | Negative |
| --- | ---: | ---: | ---: |
| `coco` | 2268 | 2268 | 0 |
| `coconeg` | 445 | 0 | 445 |
| `game1080` | 200 | 46 | 154 |
| `kaggle` | 851 | 851 | 0 |
| `negatives` | 80 | 0 | 80 |
| `pseudo` | 103 | 10 | 93 |
| `ultimateml` | 1001 | 1001 | 0 |

使用 seed 42 的 split 预估：

| Split | Total | Positive | Negative | Sources |
| --- | ---: | ---: | ---: | --- |
| `train` | 3998 | 3348 | 650 | `coco`、`coconeg`、`game1080`、`kaggle`、`negatives`、`pseudo`、`ultimateml` |
| `val` | 474 | 411 | 63 | `coco`、`coconeg`、`kaggle`、`negatives`、`pseudo`、`ultimateml` |
| `test` | 476 | 417 | 59 | `coco`、`coconeg`、`kaggle`、`negatives`、`pseudo`、`ultimateml` |
