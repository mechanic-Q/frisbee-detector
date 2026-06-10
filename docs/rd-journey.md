# frisbee-detector 研发全记录

> 从零构建一个极限飞盘检测模型的完整历程 —— 技术选型、数据闭环、踩坑与解法。

---

## 一、项目背景与目标

**目标：** 从极限飞盘（Ultimate Frisbee）比赛视频中，实时检测高速飞行中的飞盘，输出像素坐标，最终映射到场地真实坐标（米），为运动轨迹分析和数据统计提供底层能力。

**核心难点：**

- 飞盘是**极小目标**（1080p/60fps 视频中通常仅占几十像素）
- 高速运动（可达 25m/s），帧间位移大
- 复杂背景：球员、白帽子、反光椅子、记分板、场地线、阴影
- **单类别检测**：只有一个"frisbee"类，分类损失权重调优空间非常有限

---

## 二、研发流程总览

```
数据采集 → 格式转换 → 自动标注 → 数据集合并 → 模型训练 → 推理评估 → 误检挖掘 → 数据闭环迭代
    ↑                                                                                    |
    └──────────────────────── 标注任务层 (annotation pipeline) ←─────────────────────────┘
```

项目遵循**数据驱动 AI** 的研发范式。核心信念：**模型架构已足够好，瓶颈在数据质量**。迭代方向不是换更大的模型，而是持续优化训练数据。

### Phase 1: 数据采集与清洗 (2026-05-11 ~ 2026-05-12)

收集了 7+ 数据源：

| 数据源 | 图片数 | 处理方式 |
|--------|:------:|----------|
| UltimateML（Roboflow 导出） | ~1001 | 多类合并为单类 firsbee |
| Kaggle frisbee | 851 | 2 类合并为 1 类，80/10/10 划分 |
| COCO 2017 frisbee (category=34) | ~1781 | 剔除损坏标签，20 线程并行下载 |
| frisbee_dataset | 80 | **全部标注不可靠 → 全部转负样本** |
| Pseudo-labels (v1 模型) | ~1600 | conf≥0.60 保留，0.25-0.60 丢弃 |
| Hard negatives (VLM 筛选) | 动态 | 从测试视频 FP 帧提取 |
| game1080 (GroundedSAM + 人工复核) | 200 | autodistill_grounded_sam 自动标注 |

### Phase 2: 模型训练与迭代

| 版本 | 关键变化 | mAP50 | 帧检测率 | dets/帧 | 问题 |
|------|----------|:-----:|:--------:|:-------:|------|
| v1 | UltimateML only, 激进的伪标签 | — | — | — | 误检过多 |
| v2 | 合并数据集，box=10，close_mosaic=15 | — | 4.3% | — | **数据泄露**导致指标虚高，recall 极差 |
| v3 | 修复泄露，box=5，诚实验证 | — | 69.9% | 1.86 | **召回跳跃但 FP 爆炸（60%+）** |
| v5/v6/v6b | cls=1.3 调参 | — | — | — | ❌ **cls 调参路线失败** |
| v7_quick | 数据驱动：FP crops + TP frames (imgsz=640) | — | 53.5% | 0.86 | 召回下降（分辨率原因），dets/帧改善 |
| P2 shadow_v1 | YOLOv8s-P2，阴影/误检数据闭环 | — | — | — | ✅ **当前默认模型** |

### Phase 3: 追踪与场地映射 (2026-05-18)

在检测模型基础上构建了两层下游能力：

1. **单飞盘 Kalman 追踪器** — 从 ByteTrack 多目标追踪切换为自定义单目标 Kalman 滤波 + 候选评分
2. **Homography 场地标定** — 4 点透视变换将像素坐标映射到 100m×37m 场地真实坐标

### Phase 4: 标注任务层 (2026-06-06 ~ 2026-06-07)

建立了通用的 annotation 数据闭环流水线：

```
YAML项目配置 → 候选生成(generate) → Streamlit 人工复核(review) → 导出YOLO标签(export) → 合并训练集(merge) → 重训
```

支持两类任务：
- `bbox_review`：模型检测框 TP/FP 复核
- `frame_label`：阴影帧 VLM 辅助标注

---

## 三、技术选型及理由

### 检测模型：YOLOv8s → YOLOv8s-P2

| 选择 | 理由 |
|------|------|
| **YOLOv8** | 成熟稳定的单阶段检测器，Ultralytics 生态完善（训练/验证/推理/导出一条龙） |
| **s (small)** | RTX 5080 16GB 显存限制，m/l 模型 batch=2 都难跑，且飞盘检测不需要深层语义 |
| **P2 变体** | YOLOv8s-P2 增加了一个高分辨率特征图层（P2），对小目标检测有显著提升。飞盘在 1280px 视频中通常只有 20-50px，标准 P3-P5 特征图感受野太大 |
| **不用 YOLO11/12** | 尝试过 yolo11n、yolo12n，无明显提升。YOLOv8 生态兼容性最好（SAHI、autodistill） |

### 图像尺寸：imgsz=1280

飞盘太小。640px 输入下飞盘缩到不足 15px，几乎不可检测。1280 是 RTX 5080 16GB + batch=2 的极限。

### SAHI 切片推理

对大图分块推理（640×640 tiles，20% overlap），让小飞盘在切片中占据更大比例。召回可提升 5-15%，代价是推理速度下降 3-10 倍。

**遇到的问题：** SAHI 与 ultralytics 8.4 的兼容性需验证，安装时需 `--break-system-packages`。

### 追踪器：自定义 Kalman Filter

| 方案 | 为什么不选 |
|------|-----------|
| ByteTrack (YOLO 内置) | 多目标追踪器，为行人/车辆设计。飞盘场景每帧只有 1 个目标，ByteTrack 会产生大量虚假 tracklet |
| DeepSORT | 需要外观特征提取，飞盘外观特征极弱（白色圆盘没有纹理） |
| **自定义 Kalman** | ✅ 4 状态 (x, y, vx, vy) → 后升级为 **6 状态** (+ax, +ay)，加速度建模使 Mahalanobis d² 分布改善 60-80% |

候选评分权重：35% 运动一致性 + 25% 置信度 + 25% 面积一致性 + 15% 宽高比（飞盘是扁圆的）。

### Homography：4 点透视变换

使用 OpenCV `cv2.getPerspectiveTransform`，从场地标线交叉点（角旗/边线交点）计算 3×3 单应矩阵。Streamlit 标定工具支持可视化验证（场地线叠绘 + 鸟瞰图）。

### 自动标注与审核

| 工具 | 用途 | 理由 |
|------|------|------|
| **autodistill_grounded_sam** | game1080 初始标注 | GroundedSAM 零样本能力，不需要训练 |
| **GLM-4V-Flash (智谱)** | VLM 辅助 FP 判定 / 阴影帧复核 | 成本可控（¥0.0001/图），API 稳定 |
| **Streamlit** | 人工复核 UI | 快速搭建，键盘快捷键驱动（y/n/u/s/r, Backspace 撤销） |

### VLM vs 分类器判定 FP

曾尝试两条自动判定 FP 的路线：

1. **ResNet18 二分类器**：22 TP + 178 FP 标注数据训练，val_acc=95%。但推理 v7_quick 的 4228 个检测时，99.8% 被判为 FP。**正样本严重不足，严重过拟合**。失败。

2. **SigLIP 零样本分类**：`google/siglip-so400m-patch14-384`，零样本能力在飞盘 vs 白帽子场景下也不可靠。

结论：**对于飞盘检测的 FP 判定，人工复核 + VLM 辅助比训练分类器更可靠。** 单类正样本稀缺是根本瓶颈。

---

## 四、踩坑记录

### 坑 1: 数据泄露（v2 指标虚高）

**现象：** v2 模型在验证集上 mAP 很高，但在测试视频上帧检测率只有 4.3%。

**原因：** pseudo-labeling 时用了**已经在训练集中的视频帧**做自动标注。模型在训练时见过这些帧，验证集失去了独立性。

**解法：**
- 在 v3 中彻底清理了训练/验证/测试的数据泄露
- 在 annotation 任务配置中引入 `exclude_range` 机制：禁止将 eval_segment 时间范围内的帧用于训练
- 建立了**防泄露区间**（leakage range）概念，所有候选生成和导出都必须遵守

**教训：** Pseudo-labeling 必须严格隔离。用模型标注过的帧，不能再用同一帧来验证。数据泄露是 CV 项目中最隐蔽也最致命的 bug。

### 坑 2: cls 调参路线全面失败（v5/v6/v6b）

**假设：** 提高分类损失权重 `cls`（从默认 0.5 到 1.3），让模型更谨慎地判断"是飞盘"，从而减少误检。

**实验过程：** 训练 v5（cls=1.3）、v6（加 OpenImages 数据）、v6b（cls=2.0），耗时约 6 GPU 小时。

**结果：** 所有 cls 调参版本均未改善 FP 率。根本原因：

> **单类别检测任务中，`cls` 损失权重没有区分度。** 模型不会"更谨慎地判断是飞盘"，而是**全面降低所有检测的置信度**——真假飞盘一起被抑制。对于多类任务（如 COCO 80 类），`cls` 调参有效是因为类别之间有竞争；单类任务只有一个类别 + background，提高 `cls` 等同于降低 `box`。

**教训：** 在动手调参前，先理解损失函数的语义。单类检测的精度提升只能靠数据（更多负样本/hard negatives），不能靠损失权重。

### 坑 3: YOLO 双嵌套输出路径

**现象：** 训练输出跑到 `runs/detect/runs/detect/frisbee_det_s_vX/` 而不是 `runs/detect/frisbee_det_s_vX/`。

**原因：** `train.py` 中设置 `project="runs"`，但 YOLO 默认 `project="runs/detect"`，两者拼接导致双嵌套。

**解法：** 训练结束后 `mv runs/detect/runs/detect/frisbee_det_s_vX runs/detect/frisbee_det_s_vX`。修复方式是改 `project="runs/detect"` 或直接用 `project="."`。

### 坑 4: frisbee_dataset 标注质量极差

**现象：** 381 个标注中，149 个（39%）面积超过图像 50%——是全图标注或超大框错误。

**决策：** 全部 80 张图转为负样本（空标签）。宁可损失可能有效的正样本，也不让脏标注污染训练数据。

**理由：** 无法可靠区分哪些小框是真正有效的飞盘标注。在 80 张图 vs 污染风险之间，选择了安全的一边。

### 坑 5: VLM 判定 FP 不可靠

**现象：** 用 GLM-4V-Flash 判定 100 张检测裁剪图，结论 74% TP / 26% FP。但人工复核发现 VLM 把白帽子、人头、反光物体大量误判为飞盘。

**结论：** VLM 不适合直接做 fine-grained 的飞盘/非飞盘判定。它的语义理解偏向"这个物体看起来像圆形 = 可能是飞盘"。

**当前策略：** VLM 仅用作阴影帧的辅助初筛（判断红框内是否为飞盘），不做最终裁决。关键判定仍依赖人工复核。

### 坑 6: NTFS 挂载 I/O 性能

**现象：** 项目数据存储在 `/mnt/e/`（Windows NTFS 分区挂载到 WSL2）。YOLO 训练的 DataLoader 在 NTFS 上读取 5000+ 张训练图时，I/O 成为瓶颈。

**解法：**
- 启用 `--cache` 将图片缓存到 RAM（RTX 5080 的 16GB 显存不够，但系统 RAM 充足）
- `workers=2`（更多 workers 在 NTFS 上反而因为锁竞争变慢）
- 大文件操作（mv 5.1GB 模型目录）在 NTFS 上极慢，需耐心等待

### 坑 7: Kalman 4→6 状态升级

**现象：** 4 状态 Kalman（位置 + 速度）在飞盘变速运动时追踪不稳定，预测位置和检测位置之间的 Mahalanobis 距离分布偏大。

**解法：** 升级为 6 状态 Kalman（位置 + 速度 + **加速度**），process noise 和 measurement noise 针对 1080p/60fps 重新调参。

**效果：** Mahalanobis d² 分布改善 60-80%，追踪更稳定。

### 坑 8: SAHI 集成兼容性

**现象：** `pip install sahi` 在系统 Python 环境中与已有 ultralytics 版本冲突。

**解法：** `pip install sahi --break-system-packages`。SAHI 的 `AutoDetectionModel.from_pretrained()` 需要传入 YOLO 模型路径，文档示例与 ultralytics 8.x 的实际 API 有细微差异，需查看 SAHI 源码确认参数名。

### 坑 9: 训练必须在 tmux 中运行

这不是 bug，而是 Claude Code 环境的约束：Bash 工具有 10 分钟超时，YOLO 训练需要 1-3 小时。所有训练命令必须通过 `tmux new-session -d -s train` 在后台运行，训练结束后通过日志确认结果。

### 坑 10: 正样本极度稀少

这是项目最根本的挑战。飞盘在视频中出现的频率低（可能几十秒才出现一次），且每次出现只有 2-5 帧可见。这导致：

- 自动标注（pseudo-labeling）产出的正样本有限
- 二分类器训练时正样本不足（22 TP vs 178 FP）
- 数据不平衡严重

**对策：**
- 多源数据聚合（7+ 数据源）
- GroundedSAM 零样本标注扩充
- 人工复核优先保障正样本质量
- 负样本策略：用 hard negatives 替换通用背景图

---

## 五、项目架构概览

```
frisbee-detector/
├── configs/              # 集中配置（模型路径、数据集YAML、annotation项目）
│   ├── models.py         # 模型版本管理与默认参数
│   ├── paths.py          # 路径常量
│   ├── annotation/       # 标注任务项目配置（YAML）
│   └── homography/       # 场地标定结果（JSON）
├── models/               # 训练与验证
│   └── train.py          # YOLOv8 训练 CLI（支持 --product, --resume, P2-variant）
├── inference/            # 推理管线
│   ├── predict_video.py  # 视频推理（YOLO + SAHI + 统计输出）
│   ├── predict_track.py  # 单飞盘 Kalman 追踪 + 轨迹可视化
│   └── predict_image.py  # 单图推理
├── tools/                # 工具链（27个脚本）
│   ├── annotation_core.py           # 标注数据模型、防泄露、任务IO
│   ├── generate_annotation_tasks.py # 候选生成
│   ├── review_tasks.py              # Streamlit 通用复核器
│   ├── export_annotation_tasks.py   # 导出YOLO训练样本
│   ├── merge_annotation_exports.py  # 合并导出到训练集
│   ├── merge_datasets.py            # 多源数据集合并
│   ├── auto_label.py / auto_label_gsam.py  # 自动标注
│   ├── vlm_review_crops.py          # VLM辅助FP判断
│   ├── collect_hard_negatives.py    # VLM硬负样本挖掘
│   ├── convert_*.py                 # 各数据集格式转换
│   └── calibrate_field.py           # Streamlit场地标定
├── utils/                # 核心库
│   ├── homography.py     # 单应矩阵计算、坐标变换、可视化
│   ├── tracker_utils.py  # Kalman初始化、候选评分、轨迹管理
│   └── dataset.py        # YAML配置生成、数据集划分
├── docs/                 # 文档与计划
│   ├── development.md                   # 开发指南
│   ├── conventions/naming-glossary.md   # 命名词典
│   └── superpowers/{plans,specs}/       # 14个实现计划 + 7个设计规格
└── tests/                # 37+单元测试
```

---

## 六、关键经验总结

1. **数据质量 > 模型架构。** P2 变体的提升远不如一次干净的数据清洗。v2→v3 的泄露修复让召回从 4.3% 跳到 69.9%。

2. **单类检测不能用 cls 调参治 FP。** 是一个反直觉但数学上合理的结论。单类场景增加 `cls` 权重等同于降低 `box` 权重，不会选择性抑制 FP。

3. **Hard negatives 比 generic backgrounds 更有效。** 用 178 个特定误检场景（白帽子、反光椅子）替代 283 个通用背景图，模型能更精准地学习"什么不是飞盘"。

4. **Pseudo-labeling 必须防泄露。** 用模型标注的帧不能用来验证同一模型。`exclude_range` 机制是数据闭环的生命线。

5. **VLM 不适合做精准判别。** GLM-4V 对飞盘/非飞盘的判定准确率不到 70%，比不上人工。它的价值在初筛和辅助，不在最终裁决。

6. **小目标检测的关键在输入分辨率。** imgsz=640 vs 1280，在 20-50px 飞盘场景下是质的区别。RTX 5080 16GB 是硬约束，batch=2 已经是极限。

7. **标注工具的人机工程学至关重要。** Streamlit reviewer 的键盘快捷键（y/n/u/s/r, Backspace 撤销）让标注效率提升 3-5 倍。

---

## 七、模型版本演进时间线

| 日期 | 版本 | 里程碑 |
|------|------|--------|
| 2026-05-11 | v1 | 初始版本（UltimateML only） |
| 2026-05-12 | v2 | 合并 4 数据源 → 数据泄露 → recall 崩溃 |
| 2026-05-12 | v3 | 修复泄露 + box=5 → recall=69.9% 但 FP=60%+ |
| 2026-05-13 | v5/v6 | cls 调参路线 → **失败** |
| 2026-05-14 | v7_quick | 数据驱动（FP crops + TP frames）→ 改善 |
| 2026-05-14 | v7 | 全分辨率（imgsz=1280）生产训练 |
| 2026-05-14 | — | ResNet18/SigLIP FP 分类器 → **失败** |
| 2026-05-18 | — | 单飞盘 Kalman 追踪器 + Homography 标定 |
| 2026-06-06 | — | P2 通用标注任务层建立 |
| 2026-06-07 | P2 shadow_v1 | YOLOv8s-P2 + 阴影/误检数据闭环 → **当前默认模型** |
| 2026-06-10 | — | Kalman 6 状态升级 + Mahalanobis d² 分析 |

---

*最后更新：2026-06-10*
