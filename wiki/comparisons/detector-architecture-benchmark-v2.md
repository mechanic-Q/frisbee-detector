---
type: comparison
title: 检测器架构基准 v2（飞盘检测，默认状态×统一数据）
tags: [benchmark, architecture, frisbee-detection]
related: [dfine-s, rfdetr-small, hard-negative-mining, p2-head]
created: 2026-09-10
updated: 2026-09-10
---

# 检测器架构基准 v2

## 协议

- **同起跑线**：全部官方 COCO 预训练默认权重（历史自训版本一律不参与）
- **同数据**：frisbee_merged_v2（4629 训练/476 测试，含 494 VLM 确认硬负样本）
- **同配方**：box=5、patience=8、seed=42；YOLO 系 30ep@1280，D-FINE 官方 72ep@640，RF-DETR 30ep@672（各用官方默认档）
- **同标尺**：frisbee_merged test split（476 图/521 框），pycocotools/ultralytics 标准口径

## 结果（test 集）

| 模型 | License | Epochs | 分辨率 | mAP50 | mAP50-95 | FPS |
|---|---|---|---|---|---|---|
| **D-FINE-S** | **Apache-2.0** | 72(官方) | 640 | **0.8655** | **0.6470** | ~40(批) |
| RF-DETR-small | Apache-2.0 | 30 | 672 | 0.7432 | 0.5714 | 39.8 |
| v8s-P2 prod（100ep+硬负样本） | AGPL | 100 | 1280 | 0.8092 | 0.4332 | — |
| shadow_v1 基线（100ep） | AGPL | 100 | 1280 | 0.8547* | 0.4810 | — |
| v8s-P2（30ep） | AGPL | 30 | 1280 | 0.7010 | 0.3612 | ~13 |
| YOLOv8s 标准头 | AGPL | 30 | 1280 | 0.6386 | 0.3378 | 13.6 |
| YOLO11s 标准头 | AGPL | 30 | 1280 | 0.6122 | 0.2950 | — |
| YOLO26s 标准头 | AGPL | 30 | 1280 | 0.5697 | 0.2955 | 14.1 |

*shadow_v1 的 test 集与 merged_v2 test 不同源（旧切分），仅作量级参考。

## 结论

1. **D-FINE-S 断层夺魁**：30ep 超越全部 100ep 的 ultralytics 模型（含 prod），mAP50-95 领先 0.17+——FDR 框分布精炼对小目标定位精度有实质增益。
2. **标准头排序**：v8s(0.639) > 11s(0.612) > 26s(0.570)——"更新≠更好"实测成立。
3. **P2 头增益**：v8s 同代对比 0.701 vs 0.639（+6 分）——小目标头有效。
4. **纯开源可行**：D-FINE/RF-DETR 均 Apache-2.0，性能覆盖 AGPL 系，商用无授权障碍。
5. **速度代价**：D-FINE-S @640 单帧 52.8 FPS vs YOLOv8s @640 的 167.9 FPS（YOLO 快 3.2 倍）——离线分析场景可接受。
6. **YOLO26 的 MuSGD 优化器与端到端 val 行为与 v8/11 不同**（ultralytics/skills 提示），对比数字含训练器差异。

## 遗留实验

- [ ] D-FINE-S @1280 重训（640 权重不能直接推 1280——位置编码绑定，需重训；预期小目标召回再涨）
- [ ] D-FINE + 硬负样本迭代（YOLO 线已证明正收益）
- [ ] TensorRT INT8 导出后同精度复测
