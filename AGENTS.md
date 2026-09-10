# AGENTS.md — frisbee-detector

YOLOv8s frisbee detection. Real-time frisbee spotting in ultimate frisbee game footage.

## Quick start

```bash
# Run inference (scripts have sys.path bootstrap, run from project root; Windows 原生)
python inference/predict_video.py --video movie/clip_20-23min.mp4 --conf 0.35
python inference/predict_video.py --sahi --video movie/clip_20-23min.mp4 --conf 0.35
python inference/predict_image.py /path/to/image.jpg

# Train (1-3h — Bash 工具用 run_in_background 后台跑；Windows 无 tmux)
python models/train.py --data configs/frisbee_merged.yaml --box 5 --name frisbee_det_s_vN

# Validate
python3 models/train.py --validate-only --model-path runs/detect/frisbee_det_s_v3/weights/best.pt

# Tests
python -m pytest tests/ -v

# Merge datasets (preserves source data — uses copy not mv)
python tools/merge_datasets.py

# Review auto-labels (GSAM, pseudo, etc.)
streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames
streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt

# Interactive detection review (Web — Streamlit)
streamlit run tools/review_web.py -- --img-dir data/fp_spotcheck_50/

# Interactive detection review (Desktop — Tkinter)
python tools/review_desktop.py --model runs/detect/frisbee_det_s_v3/weights/best.pt --source data/fp_spotcheck_50/ --conf 0.35
yololabeler data/fp_spotcheck_50/

# Collect hard negatives via VLM (GLM-4V-Flash; needs GLM_API_KEY env var — never commit the key)
export GLM_API_KEY=<your-zhipu-key>
python tools/collect_hard_negatives.py \
  --model runs/detect/frisbee_det_s_v3/weights/best.pt \
  --videos movie/25866279684-1-192_55-56min.mp4 \
  --output data/datasets/frisbee_merged/images/train_hard_neg
```

## Match-analysis GUI v0（已并入主线；原 feat/gui-v0 → .worktrees/gui-v0 的 worktree 已于 2026-09-11 移除，直接在主 checkout 运行）

```bash
# PySide6 MUST be on python.org Python 3.11 — conda 3.13 hits a Qt DLL load failure
py -3.11 -m pip install PySide6 opencv-python numpy   # once
py -3.11 -m gui.main                                  # GUI: open video → analyze → overlay replay
QT_QPA_PLATFORM=offscreen py -3.11 -m gui.main --smoke VIDEO [TRACKS_JSON]   # headless smoke

# Analysis worker (Windows 原生 GPU; stdout = JSON-lines per protocol.py; GPU 排队锁)
python tools/gpu_run.py gui-analysis python -m frisbee_analyzer.pipeline ^
  --video E:/frisbee-detector/data/bili_final_test/testclip_60_120s.mp4 ^
  --weights E:/frisbee-detector/yolo26x.pt ^
  --output-dir E:/frisbee-detector/runs/gui_analysis/<stem>
# artifact: runs/gui_analysis/<stem>/tracks.json  {video,fps,frames,team_overrides}
```
Zero-shot COCO person
weights (yolo26x) are the v0 placeholder — swap in player/referee finetuned weights via
`--weights` without GUI changes. Known limits: zero-shot tracking includes spectators;
tracks sampled between team-sampling frames stay `team_id: null`.

## Architecture

```
gui/              → PySide6 桌面端（Windows 原生；worker 经 QProcess 原生启动，JSON-lines 协议）
frisbee_analyzer/ → pipeline.py (worker CLI) + protocol.py (消息协议) + tracking.py (BoT-SORT)
                    + team.py (SigLIP+UMAP+KMeans 分队) + events.py (事件引擎骨架，Phase 3 接入)
configs/          → paths.py (RESEARCH_ROOT, PROJECT_ROOT), models.py (DEFAULT_MODEL)
utils/            → dataset.py (YAML gen, split), io.py (safe copy/write)
tools/            → data conversion & prep scripts (auto_label, merge, extract_frames)
                → review_labels.py (Streamlit label reviewer, P2 overlay)
                → review_web.py (Streamlit FP/TP reviewer)
                → review_desktop.py (yololabeler prediction export)
                → collect_hard_negatives.py (GLM-4V-Flash VLM filtering)
models/           → train.py (train_frisbee_detector + argparse)
inference/        → predict_video.py (YOLO + SAHI), predict_image.py, visualize.py
data/datasets/    → 7 source datasets (kaggle, ultimateml, coco, negatives, pseudo, coco_neg, hard_neg)
runs/detect/      → trained models: frisbee_det_s, frisbee_det_s_v2, frisbee_det_s_v3
movie/            → test & source videos
```

## Critical gotchas

### Annotation naming and leakage rules
- Before adding annotation tools or public fields, read `docs/conventions/naming-glossary.md`.
- Use `source_video`, `eval_video`, `eval_segment`, `candidate_frame`, `candidate_bbox`, `hard_negative_crop`, `review_status`, `reviewer_decision`, and `sample_role` consistently.
- Use `exclude_range` for one blocked source-video time interval and `exclude_ranges` for config / JSON collections of those intervals.
- `eval_video` is evaluation-only. Training candidates must pass annotation `exclude_ranges`.
- Do not commit generated annotation assets under `data/annotation/`.

### Windows 原生环境（2026-09-11 起，WSL 退役）
- **python 3.13 + torch 2.11+cu128** = 训练/推理/分析 worker 主力（PATH 上的 `python`）；**py -3.11** = GUI 专用（PySide6）。两者 CUDA 均可用（RTX 5080）。
- worker 解释器可用环境变量 `FRISBEE_PYTHON` 覆盖（gui/worker_paths.py）。
- **KMP_DUPLICATE_LIB_OK=TRUE 必须在 torch 导入前设置**——torch 与 numba/umap 各带一份 OpenMP 运行时，缺了会 OMP Error #15 直接崩。pipeline.py / tools/gpu_run.py / GUI worker env 已内置。
- GPU 排队锁：`python tools/gpu_run.py <task> <cmd>`（msvcrt 独占锁，`%TEMP%/frisbee_gpu.lock`）。`tools/gpu_run.sh` 仅为 bash 惯用保留，两把锁不互通。
- 不再有 `/mnt/` 路径、wsl.exe 桥接、tmux；历史 WSL 记录见 docs/2026-09-08-match-analysis-survey-and-plan.md §9-11。

### Training constraints
- **RTX 5080 16GB**: batch=2 max, workers=2 (batch=4 OOMs during validation, batch=8 OOMs immediately)——Windows 原生 CUDA 直跑（torch 2.11+cu128, python 3.13）
- YOLO training 1-3h——**Bash 工具用 run_in_background 后台跑**（Windows 无 tmux；多任务先过 `python tools/gpu_run.py <task> <cmd>` 排队锁

### Model save path — FIXED (2026-09-09), no more manual mv
`models/train.py` passes an ABSOLUTE `project=<PROJECT_ROOT>/runs/detect`, so runs land
directly in `runs/detect/<name>`. Root cause of the old trap: a RELATIVE `project` is
resolved against ultralytics' global settings.json `runs_dir`（历史坑出在 WSL 侧的
`~/comfy/ComfyUI/runs`；WSL 已退役，教训保留）, producing nested `runs/detect/runs/<name>/`.
Historical runs still live there. `PROJECT_ROOT` 现由 `configs/paths.py` 按文件位置自适应，仓库移动不用改配置。

### Data leakage — test video frames in training
Frames extracted for pseudo-labeling must NEVER come from test videos.
The v2 model accidentally trained on test-adjacent frames — this inflates validation mAP.
Always verify: `find data/frames/ -name "*.jpg" | grep <test_video_name>` before using frames.

### COCO frisbee category_id is 34
Not 29. Found in `/tmp/coco_frisbee/annotations/annotations/instances_train2017.json`.
Use 20-thread parallel download (individual images) — never try the 18GB zip.

### sys.path bootstrap
All 13 entry scripts have `sys.path.insert(0, parent)` bootstrap.
Run from project root. No `PYTHONPATH` needed.（历史 Linux 提示 --break-system-packages 已随 WSL 退役；Windows 侧 pip 直装。）

## Model naming convention

```
frisbee_det_{size}_v{N}
frisbee_det_s      → v1 (high recall, many FP)
frisbee_det_s_v2   → v2 (data leak inflated, high precision, terrible recall)
frisbee_det_s_v3   → v3 (no leak, box=5, best model — use with conf=0.35)
frisbee_det_s_v4   → v4 (re-run with default settings, early stop at 31 epochs)
frisbee_det_s_v5   → v5 (cls=1.3, recall collapsed to 1.4% — cls too aggressive for single-class)
frisbee_det_s_v6   → v6 (cls=0.8, 8.9% detection — better but still too conservative)
```

`configs/models.py` holds DEFAULT_MODEL (currently v3). Update after training new models.
`DEFAULT_CONF=0.35` — use with v3 for best precision/recall balance.

## Key config

- `configs/models.py`: DEFAULT_MODEL, DEFAULT_CONF=0.35, SEED=42
- `configs/paths.py`: RESEARCH_ROOT=E:/firsbee, PROJECT_ROOT=仓库根(按文件位置自适应), POOL_ROOT=E:/frisbee-pool
- Override via env: `RESEARCH_ROOT`, `PROJECT_ROOT`

## False positive mitigation

The best model is **v3 with conf=0.35** — 69.9% frame detection rate, 1.86 dets/frame on 55-56min test video.

Previous approaches tried cls loss tuning (0.5→1.3), but cls acts as a confidence calibrator
that suppresses ALL detections, not just FPs. The nonlinear relationship:
- cls=0.5 (v3): 79.5% detection, 60%+ FPs
- cls=0.6 (v6b): 24.0% detection, low FP
- cls=0.8 (v6): 8.9% detection, near-zero FP
- cls=1.3 (v5): 1.4% detection, near-zero FP

The correct solution: keep cls=0.5 (no change), raise confidence threshold to 0.35 at
inference time. This filters low-confidence FPs without suppressing real detections.

Hard negatives must be bbox-level (cropped object regions), NOT full frames from test
videos. Frame-level hard negatives from the test distribution kill recall because the
model learns "this game scene = no frisbee".

COCO dataset: 2179 out of 3960 training images had corrupt labels ("license" text).
These were removed during v6 development. Clean dataset: 1781 images, 15% backgrounds.
Solution from research:
- **`cls` loss weight**: currently 0.5, should be **1.3** (Ultralytics hyperparameter tuning optimal: 1.33)
- **Hard negative mining**: run v3 on test videos → collect FP frames → add as empty-label negatives
- **Background ratio**: target 10-20% (currently 8.5%)
- Source: [Ultralytics #3466](https://github.com/ultralytics/ultralytics/issues/3466), [Hyperparameter Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
- Full plan: `.sisyphus/plans/reduce-false-positives-v5.md`

## Git workflow

```
→ main（dev 分支已于 2026-09-11 删除，主线直推；大型改动开 feat/* 分支合回后即删）
```
Only commit code/config. Never commit data/, runs/, or .pt files (gitignored).
