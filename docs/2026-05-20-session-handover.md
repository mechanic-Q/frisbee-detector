# Session Handover — frisbee-detector (2026-05-20)

> **Purpose:** New AI session reads this + `AGENTS.md` to get full context without prior conversation history.

---

## Project Summary

YOLOv8s frisbee detector for ultimate frisbee game footage. 1080p/60fps source video.
Goal: detect the flying disc reliably enough to power a tracking + field-position pipeline.

---

## Branches & What's On Them

| Branch | Status | Description |
|--------|--------|-------------|
| `main` | stable | v1 MVP — training, inference, basic tools |
| `dev` | merged | merged from improve-precision |
| `feat/p2-detector` | **active (current)** | P2 model + game1080 labeling + tracker + homography |
| `feat/single-frisbee-tracker` | merged into p2 | Kalman single-target tracker |
| `improve-precision-v5` | worktree, stalled | v5-v7 cls tuning experiments (dead end) |
| `siglip-classifier` | worktree, stalled | SigLIP binary classifier experiment |

**Active branch:** `feat/p2-detector` — 60 commits ahead of dev.

---

## Trained Models

| Model | Epochs | mAP50 | Notes |
|-------|:------:|:-----:|-------|
| `frisbee_det_s` (v1) | 100 | — | High recall, many FP |
| `frisbee_det_s_v2` | — | — | Data leak inflated metrics |
| `frisbee_det_s_v3` | 100 | — | **Best v3 model. conf=0.35. 69.9% detection, 1.86 dets/frame** |
| `frisbee_det_s_v4` | 31 | — | Early stop, default settings |
| `frisbee_det_s_v5` | — | — | cls=1.3, recall collapsed to 1.4% |
| `frisbee_det_s_v6` | — | — | cls=0.8, 8.9% detection |
| `frisbee_det_s_v7_quick` | — | — | Binary classifier experiment |
| **`frisbee_det_p2_game_v2`** | ~100 | **0.578** | **P2 (high-res small object) + 200 game1080 frames. P=0.796, R=0.604** |

**Best model overall:** `frisbee_det_p2_game_v2` — P2 architecture, trained on merged dataset + 200 labeled 1080p game frames.

---

## Dataset

**Location:** `data/datasets/frisbee_merged/` (8 source datasets merged)

| Split | Count | Source |
|-------|:-----:|--------|
| `train` | 1695 | kaggle + ultimateml + coco + negatives + pseudo + coco_neg + hard_neg |
| `train_game` | 200 | 1080p game footage (60s-90min segment, no overlap with test) |
| `val` | 495 | holdout from merged sources |
| `test` | 495 | holdout from merged sources |

**Config:** `configs/frisbee_merged.yaml` — includes `train_game` in training list.

### game1080 Labeling Status

- **200 frames** extracted from 90min video (post-60s segment)
- **GroundedSAM auto-labeled** → 113/200 had detections → `data/datasets/game1080/labels_gsam/`
- **Review tool:** `tools/review_labels.py` (Streamlit) — universal label reviewer with P2 overlay
  - Run: `streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt`
  - Supports: Accept/Reject/Skip/Prev, P2 overlay, sort by IoU, progress tracking
  - Output: `labels/` (never modifies `labels_gsam/`)
  - Result: `review_result.json`
- **Status:** Review not yet completed by user

---

## Pipeline Components (all on feat/p2-detector)

### 1. Detection
- `inference/predict_video.py` — YOLO + optional SAHI sliced inference + CSV output
- `inference/predict_image.py` — single image inference
- `inference/visualize.py` — drawing utilities

### 2. Tracking
- `utils/tracker_utils.py` — Kalman filter, candidate scorer, Trajectory ring buffer
- `inference/predict_track.py` — single-frisbee tracker with trajectory visualization
  - Replaces ByteTrack multi-object approach
  - States: searching → tracking → predicting (lost)
  - Output: CSV with px, py, vx, vy, conf, status + optional world coords

### 3. Homography (field coordinates)
- `utils/homography.py` — calibration load + pixel↔world conversion
- `configs/homography/25866279684-1-192.json` — calibration for test video
- Quality gate: field bounds check + velocity filter (>25 m/s rejected)

### 4. Data Tools
- `tools/auto_label_gsam.py` — GroundedSAM auto-labeling (produced 113/200 labels)
- `tools/review_labels.py` — Streamlit label reviewer (accept/reject/skip, P2 overlay, sort by IoU)
- `tools/_label_utils.py` — shared label I/O, IoU, P2 cache, sort utilities
- `tools/review_web.py` — Streamlit FP/TP reviewer for detection results
- `tools/review_desktop.py` — Desktop yololabeler export
- `tools/collect_hard_negatives.py` — VLM-filtered hard negative mining (GLM-4V-Flash)
- `tools/merge_datasets.py` — merges 8 source datasets
- `tools/extract_frames.py` — ffmpeg frame extraction

### 5. Training
- `models/train.py` — train + validate with argparse
- RTX 5080 16GB: batch=2, workers=2 max
- Must use tmux (training 1-3h, Bash tool 10min timeout)

---

## Test Videos

| Video | Resolution | Duration | Purpose |
|-------|:----------:|:--------:|---------|
| `25866279684-1-192_55-56min.mp4` | 720p | 60s | Primary test clip |
| `clip_20-23min.mp4` | 720p | 180s | Longer test clip |
| `videoplayback_first60s.mp4` | 1080p | 60s | **1080p test** (no overlap with training frames) |
| `videoplayback_trimmed.mp4` | 1080p | 90min | Source for training frame extraction |

---

## Key Lessons Learned

1. **cls loss tuning is a dead end for single-class** — cls=0.5→1.3 suppresses ALL detections, not just FPs. Nonlinear cliff: 0.5→79.5% detection, 0.8→8.9%, 1.3→1.4%.
2. **conf=0.35 at inference time** is the right FP filter for v3. Don't retrain to fix FPs.
3. **Frame-level hard negatives kill recall** — model learns "game scene = no frisbee". Must use bbox-level crops only.
4. **COCO frisbee category_id is 34**, not 29. 2179/3960 labels were corrupt ("license" text). Clean: 1781 images.
5. **P2 architecture helps** — higher-resolution feature maps for small objects (frisbee at distance).
6. **YOLO double-nesting bug** — output at `runs/detect/runs/detect/<name>/`. Always move after training.
7. **sys.path bootstrap** — all entry scripts have `sys.path.insert(0, parent)`. Run from project root.

---

## What's Done ✓

- [x] MVP: dataset tools, training, inference, evaluation
- [x] v3 model: best standard YOLOv8s (conf=0.35)
- [x] Merged 8 source datasets (1695 train + 495 val + 495 test)
- [x] cls loss tuning experiments v5-v7 (concluded: dead end)
- [x] P2 model with game1080 data: `frisbee_det_p2_game_v2` (mAP50=0.578)
- [x] GroundedSAM auto-labeling: 114/200 frames labeled
- [x] Streamlit label review tool: `tools/review_labels.py`
- [x] Single-frisbee Kalman tracker: `inference/predict_track.py`
- [x] Homography calibration + field coordinate conversion
- [x] Quality gate: field bounds + velocity filter
- [x] 37 unit tests passing

---

## What's In Progress / Next Steps

### Immediate (blocking)
1. **Complete GSAM label review** — 114 labeled frames in Streamlit reviewer. Accept/reject each.
2. **Merge accepted labels into training set** — copy accepted frames to `frisbee_merged/images/train_game/` + labels
3. **Retrain P2 with cleaned labels** — after review, retrain `frisbee_det_p2_v3`
4. **Evaluate on 1080p test video** (`videoplayback_first60s.mp4`)

### Short-term
5. **P2 vs v3 comparison on same test set** — quantitative benchmark
6. **Tracker integration with P2 model** — swap model path in predict_track.py
7. **End-to-end demo video** — detection + tracking + trajectory + world coords on 1080p clip

### Longer-term
8. **SAHI + P2** — test if sliced inference further helps small frisbee detection
9. **False positive reduction** — bbox-level hard negative mining from P2 FPs
10. **Merge feat/p2-detector → dev → main** after validation

---

## File Reference

```
configs/
  frisbee_merged.yaml        — dataset config (train + train_game + val + test)
  models.py                  — DEFAULT_MODEL, DEFAULT_CONF=0.35, SEED=42
  paths.py                   — RESEARCH_ROOT, PROJECT_ROOT
  homography/25866279684-1-192.json  — field calibration

data/datasets/
  frisbee_merged/            — merged training dataset (8 sources + game1080)
  game1080/                  — 200 extracted frames + GSAM labels (114/200)

inference/
  predict_video.py           — YOLO + SAHI + CSV output
  predict_track.py           — single-frisbee Kalman tracker + trajectory
  predict_image.py           — single image inference

models/
  train.py                   — train + validate CLI

utils/
  tracker_utils.py           — Kalman, scorer, Trajectory
  homography.py              — calibration + pixel↔world
  dataset.py, io.py          — dataset YAML gen, safe IO

tools/
  auto_label_gsam.py         — GroundedSAM auto-labeling
  review_labels.py           — Streamlit label reviewer (accept/reject/skip)
  merge_datasets.py          — merge 8 source datasets
  collect_hard_negatives.py  — VLM hard negative mining

docs/superpowers/
  plans/                     — 10 implementation plans (Chinese)
  specs/                     — 6 design specs (Chinese)

tests/                       — 37 tests (paths, utils, tracker, homography, verify)
```

---

## Git Workflow

```
improve-precision → dev → main
feat/p2-detector → dev (pending merge)
```

Only commit code/config. `data/`, `runs/`, `*.pt` are gitignored.
