# AGENTS.md — frisbee-detector

YOLOv8s frisbee detection. Real-time frisbee spotting in ultimate frisbee game footage.

## Quick start

```bash
# Run inference (scripts have sys.path bootstrap, run from project root)
python3 inference/predict_video.py --video movie/clip_20-23min.mp4 --conf 0.35
python3 inference/predict_video.py --sahi --video movie/clip_20-23min.mp4 --conf 0.35
python3 inference/predict_image.py /path/to/image.jpg

# Train (use tmux — training takes 1-3 hours and Bash tool times out at 10min)
tmux new-session -d -s train -c $PWD
tmux send-keys -t train "python3 models/train.py --data configs/frisbee_merged.yaml --box 5 --cls 1.3 --name frisbee_det_s_v5" Enter

# Validate
python3 models/train.py --validate-only --model-path runs/detect/frisbee_det_s_v3/weights/best.pt

# Tests
python3 -m pytest tests/ -v

# Merge datasets (preserves source data — uses copy not mv)
python3 tools/merge_datasets.py

# Review auto-labels (GSAM, pseudo, etc.)
streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames
streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt

# Interactive detection review (Web — Streamlit)
streamlit run tools/review_web.py -- --img-dir data/fp_spotcheck_50/

# Interactive detection review (Desktop — Tkinter)
python3 tools/review_desktop.py --model runs/detect/frisbee_det_s_v3/weights/best.pt --source data/fp_spotcheck_50/ --conf 0.35
yololabeler data/fp_spotcheck_50/

# Collect hard negatives via VLM (GLM-4V-Flash)
python3 tools/collect_hard_negatives.py \
  --model runs/detect/frisbee_det_s_v3/weights/best.pt \
  --videos movie/25866279684-1-192_55-56min.mp4 \
  --output data/datasets/frisbee_merged/images/train_hard_neg
```

## Architecture

```
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

### Training constraints
- **RTX 5080 16GB**: batch=2 max, workers=2 (batch=4 OOMs during validation, batch=8 OOMs immediately)
- YOLO training **must run in tmux** — Bash tool has 10-min timeout, training takes 1-3h

### Model save path DOUBLE-NESTING bug
YOLO creates output at `runs/detect/runs/detect/<name>/` when `project="runs/detect"`.
Always move the model after training:
```bash
mv runs/detect/runs/detect/frisbee_det_s_vN runs/detect/frisbee_det_s_vN
```

### Data leakage — test video frames in training
Frames extracted for pseudo-labeling must NEVER come from test videos.
The v2 model accidentally trained on test-adjacent frames — this inflates validation mAP.
Always verify: `find data/frames/ -name "*.jpg" | grep <test_video_name>` before using frames.

### COCO frisbee category_id is 34
Not 29. Found in `/tmp/coco_frisbee/annotations/annotations/instances_train2017.json`.
Use 20-thread parallel download (individual images) — never try the 18GB zip.

### sys.path bootstrap
All 13 entry scripts have `sys.path.insert(0, parent)` bootstrap.
Run from project root. No `PYTHONPATH` needed. `opencv` installs at system level, use `--break-system-packages`.

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
- `configs/paths.py`: RESEARCH_ROOT=/mnt/e/firsbee, PROJECT_ROOT=/mnt/e/frisbee-detector
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
improve-precision → dev → main
```
Only commit code/config. Never commit data/, runs/, or .pt files (gitignored).
