---
type: entity
title: D-FINE-S
tags: [detector, apache-2.0, detr]
related: [detector-architecture-benchmark-v2]
created: 2026-09-10
updated: 2026-09-10
---

# D-FINE-S

ICLR 2025 检测器（DETR 家族），Apache-2.0 纯开源。仓库：Peterande/D-FINE。

## 架构特点

- **FDR（Fine-grained Distribution Refinement）**：框坐标建模为概率分布，decoder 逐层精化——定位精度（mAP50-95）显著优于 YOLO 系的 DFL 一次性回归
- HGNetv2-B0 backbone，3 层 decoder，10M 参数（S 档）
- 分辨率约束：必须是 32 的倍数；**位置编码绑定训练分辨率**（640 权重不能直接推 1280，需重训）

## 在本项目的实测（frisbee_merged_v2）

- 72ep 官方配方（@640）：**test mAP50 = 0.8655 / mAP50-95 = 0.6470**——30+ep 组最高，超全部 100ep ultralytics 模型
- 推理速度 @640：52.8 FPS（vs YOLOv8s 167.9，慢 3.2 倍；离线分析场景可接受）
- License Apache-2.0：无 AGPL 商用传染

## 集成状态

- 训练：Windows 原生跑通（win_gpu_queue），配置 `D-FINE/configs/dfine/dfine_hgnetv2_s_frisbee.yml`
- 评测：[[detector-architecture-benchmark-v2]]
- 待办：@1280 重训、+硬负样本迭代、TensorRT 导出
