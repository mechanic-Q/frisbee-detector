# Wiki Log

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
