# Wiki Log

## 2026-09-10（域外验证）

- 完成 [[golden-set-100-domain-validation]]：9 模型在真实转播画面 100 帧上的域外评测
- 关键发现：D-FINE-S 0.659 域外仍第一（领先扩大）；RF-DETR 从 test 第2跌到域外第8（过拟合）；YOLO26s 两处垫底
- 附带：9 模型全共识框 = 高精度自动采标信号（人工裁剪确认，不依赖 VLM）
- VLM 通道不可用（GLM key 已清理）→ 改共识分层+人工目检，已记入风险处置


## 2026-09-10（下午续）

- 11s 权重遗失解决：Windows 原生重训完成（30ep），test mAP50 = 0.6122（与首次同配方数字一致，配方稳定性交叉验证）
- D-FINE-S test 评测完成：mAP50 = 0.8655 / mAP50-95 = 0.647（WSL 侧，checkpoint 名修正 best_stg1.pth）
- prod v8s-P2 权重确认完整（best/last/11 快照，ComfyUI 侧），test mAP50 = 0.8092 已验证；"prod 需恢复"为状态误判
- D-FINE @1280 直接推理验证：不可行（位置编码绑定），@1280 需重训（待排）
- 速度公平对比（同 640）：YOLOv8s 167.9 FPS vs D-FINE-S 52.8 FPS（3.2x）——离线场景可接受
- bench_summary.json 全部 4 模型数字补齐，无 ERROR

## 2026-09-10

- 创建飞盘项目 wiki：purpose.md + schema.md + index.md + log.md（按 llm-wiki nashsu 模板）
- 沉淀 [[detector-architecture-benchmark-v2]]：8 模型对比完成（D-FINE-S 0.866 夺魁，Apache 纯开源；26s 标准头垫底 0.570；"更新≠更好"实测）
- 沉淀 [[dfine-s]] 实体页：FDR 架构原理 + 本项目实测数字 + 位置编码分辨率绑定约束
- 沉淀 [[hard-negative-mining]]：VLM 确认负样本 −55% 误检 + 标准 mAP 反升（对照实验）
- 沉淀 [[p2-head]]：+6 分实测增益；26 代无 P2 yaml
- 沉淀 [[gpu-environment-reliability]]：WSL2 多进程并发不稳定定案；6 个环境坑与修复；队列正确形态（单串行+锁，绝不叠编排器）
- 开放问题 [[monocular-speed-accuracy]]：单目测速精度路线（低空段/平均速度/物理拟合）

## 2026-09-09

- 基准前置：四模型架构基准启动（v8s/11s/26s/RF-DETR 官方权重同起跑线）
- D-FINE-S 引入评估（源码安装 + frisbee 数据配置适配）
