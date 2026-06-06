# 命名词典

## 适用范围

本词典约束公共 CLI 参数、配置 key、JSONL / CSV 字段、跨文件函数参数和文档术语。局部临时变量可以保持简洁，但不能把同一公共概念写成多个名字。

## Annotation 数据闭环术语

| 标准名 | 说明 | 避免混用 |
|--------|------|----------|
| `source_video` | 可用于挖样本的完整源视频 | `input_video`、`target_video` |
| `eval_video` | 只用于评估的视频，不进入训练 | `test_video` |
| `eval_segment` | 评估片段在源视频中的映射记录 | `test_clip` |
| `exclude_range` | 禁止候选生成和训练导出的单条源视频时间区间概念 | `skip_range`、`blacklist_range` |
| `exclude_ranges` | 配置 / JSON 字段中承载多条 `exclude_range` 的集合 key | `skip_ranges`、`blacklist_ranges` |
| `candidate_frame` | 待人工复核的整帧候选 | 公共接口中避免只写 `frame` |
| `candidate_bbox` | 待人工复核的检测框候选 | 公共接口中避免只写 `box` |
| `hard_negative_crop` | 明确非飞盘的误检裁剪图 | `fp_crop` 只允许作局部变量 |
| `review_status` | 复核任务流转状态 | `status`、`state` |
| `reviewer_decision` | 人工复核语义结论 | `result`、`label` |
| `sample_role` | 样本在训练闭环中的角色 | `type`、`kind` |

## review_status 允许值

| 字段值 | 说明 |
|--------|------|
| `pending` | 未复核 |
| `accepted` | 已给出有效复核结论 |
| `rejected` | 明确不采用 |
| `skipped` | 暂时跳过 |

## reviewer_decision 允许值

| 字段值 | 说明 |
|--------|------|
| `frisbee` | 人工确认是飞盘 |
| `not_frisbee` | 人工确认不是飞盘 |
| `uncertain` | 不确定，不进入训练 |
