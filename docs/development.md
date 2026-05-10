# 开发指南

## 环境搭建

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/WSL
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
# SAHI optional: pip install -r requirements-optional.txt
```

## 训练流程

```bash
# Evaluate existing datasets and models (critical decision point)
python tools/analyze_labels.py /mnt/e/firsbee/03_datasets/UltimateML/ultimate_train/Ultimate-Frisbee-Game-7
python tools/convert_ultimateml.py

# Train (Plan A: fine-tune from existing weights)
python models/train.py --resume /mnt/e/firsbee/03_datasets/UltimateML/models/best.pt

# Train (Plan B: from COCO pretrained)
python models/train.py

# Validate
python models/train.py --validate-only --model-path runs/detect/frisbee_det_s/weights/best.pt
```

## 推理

```bash
python inference/predict_video.py /path/to/video.mp4
python inference/predict_image.py /path/to/image.jpg
```

## 分支策略

`main` (stable) ← `dev` (integration) ← `feat/*`, `fix/*`

Use Conventional Commits: `feat:`, `fix:`, `docs:`, `data:`, `chore:`, `test:`
