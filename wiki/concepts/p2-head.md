---
type: concept
title: P2 小目标检测头
tags: [architecture, small-object]
related: [detector-architecture-benchmark-v2]
created: 2026-09-10
updated: 2026-09-10
---

# P2 小目标检测头

在 YOLO 标准三尺度头之上追加更高分辨率的 P2 检测层，服务 ~10-25px 的极小目标（飞盘）。

## 实测增益（frisbee_merged_v2, 30ep, v8s）

| 配置 | mAP50 | 差值 |
|---|---|---|
| v8s-P2 | 0.701 | **+6 分** |
| v8s 标准头 | 0.639 | — |

## 交互发现（26s 代际）

- 26s 标准头（0.570）**低于** v8s 标准头（0.639）——新一代架构的端到端设计（MuSGD、无 NMS）在 P2 小目标组合上反而弱
- ultralytics 8.4.46：v8/26 有 P2 yaml，11 无
- 结论：**架构选型按任务实测，不按代际新旧**

## P2 的代价

- 计算量增大（高分辨率特征图）：~37 GFLOPs vs 标准头更低
- 训练显存需求上浮——batch=2 上限的根因之一
