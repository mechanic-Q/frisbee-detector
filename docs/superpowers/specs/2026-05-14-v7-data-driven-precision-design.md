# v7 Data-Driven Precision Improvement — Design Spec

**Date**: 2026-05-14
**Status**: Approved
**Branch**: `improve-precision-v5`

## TL;DR

After 6 training iterations failed to reduce false positives through hyperparameter tuning (cls,
box, patience), we now have concrete evidence: **89% FP rate at the bbox level** and **TP/FP
confidence distributions completely overlap** (both mean 0.52). Confidence threshold filtering
cannot separate real frisbees from false alarms — this is a training data quality problem.

The fix: replace generic background images with 178 real false-positive examples from manual
review, plus add 22 true-positive frames with proper YOLO annotations. Train at imgsz=640
for fast (15min) validation, then scale to imgsz=1280 for production if the direction works.

## 1. Problem Root Cause

### What we know now

| Finding | Evidence |
|---------|----------|
| Per-box FP rate at conf=0.35 | 89% (178/200 from manual spot-check) |
| TP confidence distribution | Mean 0.52, range 0.36-0.68 |
| FP confidence distribution | Mean 0.52, range 0.35-0.70 |
| cls tuning effectiveness | Nonlinear failure — 0.5→79%, 0.6→24%, 0.8→9%, 1.3→1.4% |
| Frame-level hard negatives | Kill recall entirely (model learns "game scene = no frisbee") |
| COCO dataset | 2179/3960 training images had corrupt labels ("license" text) |

### Why hyperparameter tuning failed

cls loss weight acts as a **confidence calibrator**, not a **discriminative feature
enhancer**. Increasing cls doesn't teach the model what a frisbee looks like — it just
suppresses all detections uniformly. TP and FP confidence scores move together.

### Why this design is different

This design uses **data quality**, not hyperparameter gambling:
- Replace generic backgrounds with actual FP examples the model gets wrong
- Add real TP examples with accurate bounding boxes
- Fast validation cycle (15min) to check direction before committing to 1h training

## 2. Data Changes

### Current training set (after COCO cleanup)

| Category | Count |
|----------|:-----:|
| Labeled images (non-COCO, valid annotations) | 1,498 |
| Background images (non-COCO, empty labels) | 283 |
| Total training images | 1,781 |
| Background ratio | 15.9% |

### New training set

| Category | Count | Change |
|----------|:-----:|--------|
| Labeled images (existing) | 1,498 | — |
| TP frames (from test video, with YOLO bbox) | 22 | New |
| FP crops (from manual review, empty labels) | 178 | New |
| Background images (original, removed) | 283 | Deleted |
| **Total training images** | **1,698** | -83 |
| **Background ratio** | **10.5%** | ↓ |

### Key principle: quality over quantity

- 178 new background images are **hard negatives** — specific objects that currently fool v3
- 22 new TP frames add real game-scene frisbee appearances with accurate bounding boxes
- Replacing 283 generic backgrounds with 178 targeted FPs improves FP relevance while
  reducing total background count (less risk of suppressing recall)

## 3. Data Preprocessing

### TP frames (22 images)

From `review_results.csv` → extract 22 TP entries with frame number and bbox coordinates:
1. Read source frame from 55-56min test video
2. Convert pixel coordinates `(x1,y1,x2,y2)` to YOLO normalized format `(cx,cy,w,h)`
3. Save frame as `.jpg` + label as `.txt` (class_id=0)
4. Filename prefixed `tp_` for traceability

### FP crops (178 images)

Already extracted as individual cropped images. Copy directly as empty-label backgrounds:
1. Copy `.jpg` to `images/train/`
2. Create empty `.txt` in `labels/train/`
3. Filename prefixed `fp_` for traceability

## 4. Training Configuration

### Fast validation parameters

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| Starting point | v3 best.pt | Preserve proven frisbee feature extraction |
| cls | 0.5 | No tuning — data quality is the fix |
| box | 7.5 | No tuning — keep default |
| imgsz | 640 | 4x faster training, small frisbee accepted for validation |
| epochs | 30 | Quick direction check |
| patience | 10 | Early stop if no improvement after 10 epochs |
| batch | 2 | RTX 5080 16GB limit |
| workers | 2 | Stable, no OOM |
| optimizer | AdamW | Standard |
| lr0 | 0.001 | Fine-tune rate |
| Name | `frisbee_det_s_v7_fast` | — |

**Expected training time**: ~10-15 minutes

### Production parameters (if fast validation passes)

| Parameter | Value |
|-----------|:-----:|
| imgsz | 1280 |
| epochs | 100 |
| patience | 15 |
| Name | `frisbee_det_s_v7` |

**Expected training time**: ~1 hour

## 5. Evaluation Plan

### Setup

- Inference at **imgsz=1280** and **conf=0.35** (same as baseline)
- Evaluate on both test videos

### Baseline comparison

| Metric | v3 (baseline) | v7 fast (target) | v7 prod (target) |
|--------|:---:|:---:|:---:|
| Frame detection rate (55-56min) | 69.9% | ≥ 50% | ≥ 60% |
| Dets/frame | 1.86 | ≤ 1.5 | ≤ 1.3 |
| Per-box FP rate (50-box sample) | 89% | ≤ 50% | ≤ 25% |
| Frame detection rate (20-23min) | 61.4% | ≥ 40% | ≥ 50% |

### Decision gates

| Condition | Action |
|-----------|--------|
| Detection ≥ 50% AND FP ≤ 50% | ✅ Direction confirmed. Train production version (imgsz=1280, 100 epochs). |
| Detection ≥ 50% BUT FP > 50% | ⚠️ Partial improvement. Add VLM-batch hard negatives → train again. |
| Detection < 50% | ❌ Approach B failed. Fall back to v3 + conf threshold or try Approach A. |

## 6. Implementation Tasks

### Task 1: Data preprocessing script

Create `tools/prep_v7_data.py` that:
1. Reads `review_results.csv`
2. Extracts 22 TP → source frames from test video → YOLO labels
3. Copies 178 FP crops with empty labels
4. Cleans 283 original backgrounds
5. Updates dataset structure

### Task 2: Batch dataset updates

Run the preprocessing script to prepare the clean dataset.

### Task 3: Fast training (tmux)

```bash
tmux new-session -d -s train -c /mnt/e/frisbee-detector/.worktrees/improve-precision-v5
tmux send-keys -t train "python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --cls 0.5 --box 7.5 --batch 2 --workers 2 \
  --imgsz 640 --epochs 30 --patience 10 \
  --resume /mnt/e/frisbee-detector/runs/detect/frisbee_det_s_v3/weights/best.pt \
  --name frisbee_det_s_v7_fast" Enter
```

### Task 4: Evaluation

```bash
python3 /mnt/e/frisbee-detector/inference/predict_video.py \
  --model /mnt/e/frisbee-detector/.worktrees/improve-precision-v5/runs/detect/frisbee_det_s_v7_fast/weights/best.pt \
  --video /mnt/e/frisbee-detector/movie/25866279684-1-192_55-56min.mp4 \
  --conf 0.35
```

Repeat for 20-23min clip.

### Task 5: FP spot-check

Extract 50 random crop samples from conf=0.35 detections for per-box FP estimation.

### Task 6: Decision

Apply decision gates → commit changes, update wiki.

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| 22 TP frames from test video cause scene overfit | Medium | Medium | 20-23min eval catches this; cropping hides scene context |
| imgsz=640 misses small frisbees → direction looks wrong | Medium | High | Production version uses imgsz=1280; 640 only for validation |
| 178 FPs not enough to improve precision | Low | Low | If direction valid but magnitude low, VLM-scale to 500+ |
| FP crops too small (20-40px upscaled) for training | Low | Medium | Resize to 640×640 with padding, not stretch |

## 8. File Changes

| File | Action | When |
|------|--------|------|
| `tools/prep_v7_data.py` | Create | Before training |
| `data/datasets/frisbee_merged/images/train/` | Modify | During data prep |
| `data/datasets/frisbee_merged/labels/train/` | Modify | During data prep |
| `configs/models.py` | Modify | After training (add V7 model path) |
| `docs/superpowers/specs/2026-05-14-v7-data-driven-precision-design.md` | This file | — |
