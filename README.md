# frisbee-detector

**YOLOv8s-P2 极限飞盘检测模型** — 从比赛视频中检测高速飞行中的飞盘，输出像素坐标 → 场地真实坐标映射。

## 核心特性

- 🎯 YOLOv8s-P2 小目标检测（飞盘在 1080p 视频中仅 20-50px）
- 🔍 SAHI 切片推理（召回 +5-15%）
- 📐 Homography 场地标定（像素 → 100m×37m 场地坐标）
- 🧭 单飞盘 6 状态 Kalman 追踪（位置 + 速度 + 加速度）
- 🔄 完整数据闭环（标注 → 训练 → 评估 → 挖掘 → 反馈）

## 快速开始

```bash
# 环境
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sahi --break-system-packages   # 可选，切片推理

# 推理
python inference/predict_video.py --video movie/test.mp4
python inference/predict_track.py --video movie/test.mp4   # 追踪 + 轨迹可视化

# 训练（需 tmux，训练 1-3h）
tmux new-session -d -s train -c /mnt/e/frisbee-detector
tmux send-keys -t train "python models/train.py --data configs/frisbee_merged.yaml \
  --model yolov8s-p2.yaml --epochs 100 --imgsz 1280 --batch 2" Enter
```

## 项目结构

```
frisbee-detector/
├── configs/              # 集中配置（模型路径、数据集YAML、标注项目）
│   ├── models.py         # 模型版本管理（目前默认 P2 shadow_v1）
│   ├── annotation/       # 标注任务项目配置
│   └── homography/       # 场地标定结果
├── models/train.py       # YOLOv8 训练 CLI
├── inference/            # 推理管线（视频推理、追踪、SAHI）
├── tools/                # 27 个工具脚本（标注、转换、合并、自动标注）
├── utils/                # 核心库（homography、tracker、dataset）
├── docs/
│   ├── rd-journey.md     # 📖 研发全记录（流程、技术选型、踩坑、时间线）
│   ├── development.md    # 开发指南
│   └── conventions/      # 命名词典
└── tests/                # 37+ 单元测试
```

## 详细文档

👉 **[研发全记录](docs/rd-journey.md)** — 必读。完整的研发历程、技术选型理由、10 个踩坑记录、模型版本演进时间线。

## 技术栈

| 层 | 技术 |
|----|------|
| 检测模型 | YOLOv8s-P2 (Ultralytics) |
| 切片推理 | SAHI |
| 自动标注 | GroundedSAM (autodistill) |
| VLM 辅助 | GLM-4V-Flash (智谱) |
| 追踪 | cv2.KalmanFilter (6 状态) |
| 标定 | cv2.findHomography / getPerspectiveTransform |
| 复核 UI | Streamlit |
| 测试 | pytest |

## 许可证

MIT
