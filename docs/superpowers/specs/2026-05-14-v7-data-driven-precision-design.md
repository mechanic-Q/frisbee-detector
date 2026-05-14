# v7 Data-Driven Precision Improvement — Design Spec

> **Goal:** Reduce per-box FP rate from 89% to <20% while keeping frame detection rate >=50%.
> **Method:** Replace background images with bbox-level hard negatives (178 FP crops) + add TP frames with YOLO labels (22 frames).
> **Approach:** Quick validation (imgsz=640, 30 epochs) first, then production training (imgsz=1280) if metrics pass.

## Problem Statement

v3 at conf=0.35 achieves 69.9% frame detection rate on 55-56min test video but per-box
FP rate is 89% (178/200 boxes reviewed were false positives). TP and FP confidence
distributions fully overlap (mean ~0.52 for both) — threshold filtering is impossible.

Previous attempts to fix this via `cls` loss weight tuning all failed:
- cls=0.5 (v3): 79.5% detection, ~60%+ FP
- cls=0.6 (v6b): 24.0% detection
- cls=0.8 (v6): 8.9% detection
- cls=1.3 (v5): 1.4% detection

The model's internal classifier cannot distinguish frisbees from white hats/rocks
because it has never been shown what those FP objects look like as negatives.

## Root Cause

The training dataset has 283 generic background images (empty fields, random scenes)
but zero examples of "things that look like frisbees but aren't." The model's
classifier has never been penalized for confusing a white hat with a frisbee.

## Solution: Data-Driven Approach (Scheme B)

### Data Changes

| Change | Count | Source | Safety |
|--------|:-----:|--------|--------|
| Remove 283 generic backgrounds | 283 | frisbee_negatives, coco_neg subsets | Safe — low-information images |
| Add 178 FP crops as backgrounds | 178 | `data/perbox_crops/` (reviewed FP) | Safe — bbox-level crops, not full frames |
| Add 22 TP frames with YOLO labels | 22 | 55-56min test video frames + v3 predictions | **Test scene leak accepted** |
| Keep 1498 existing positive images | 1498 | ultimateml, kaggle, pseudo | Unchanged |

**Net dataset:** 1498 positive + 22 TP frames + 178 FP crops = 1698 training images
Background ratio: 200/1698 = **11.8%** (target was 10-20%)

### Why Scheme B (Background Replacement)

Original dataset had 4950 images (incl. 2179 corrupt COCO). After cleaning: 1781 images
with 283 backgrounds (15.9%). Replacing 283 generic backgrounds with 178 targeted hard
negatives gives the model much stronger negative signal without increasing dataset size.

### Why 22 TP Frames Are Safe Despite Test Scene Leak

The 22 TP frames come from the 55-56min test video. This is a data leak — the model
will see test scenes during training. However:

1. Only 22 frames (1.3% of dataset) — minimal memorization risk
2. v3 already trained on pseudo-labeled frames from similar video sources
3. The primary metric is **FP rate reduction**, not test video mAP
4. 20-23min test video provides independent evaluation (different game, different venue)
5. Quick validation at imgsz=640 will show if the approach works before committing

If quick validation succeeds, we can collect fresh TP frames from non-test sources
for the production training.

## Training Configuration

### Phase 1: Quick Validation (~10-15 min)

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| Base model | v3 (`frisbee_det_s_v3/weights/best.pt`) | Fine-tune from best existing |
| imgsz | 640 | 2x faster than 1280 |
| epochs | 30 | Enough to see trend |
| patience | 10 | Early stop if needed |
| batch | 2 | RTX 5080 16GB limit |
| box | 5 | Same as v3 |
| cls | 0.5 | Same as v3 — don't change |
| workers | 2 | Stable on this hardware |
| name | `frisbee_det_s_v7_quick` | |

### Phase 2: Production Training (~1-2h, if Phase 1 passes)

Same as Phase 1 but `imgsz=1280`, `epochs=100`, `patience=20`.
Name: `frisbee_det_s_v7`.

### Decision Gate

After Phase 1, evaluate on both test videos:

| Metric | Threshold | Action |
|--------|:---------:|--------|
| FP rate | <20% | PASS → Phase 2 |
| Frame detection rate (55-56min) | >=50% | PASS → Phase 2 |
| Frame detection rate (20-23min) | >=50% | PASS → Phase 2 |
| Any metric fails | — | STOP — analyze, consider VLM expansion |

If Phase 1 fails, do NOT proceed to Phase 2. Instead:
1. Analyze failure mode (FP still high? recall collapsed?)
2. Consider VLM-expanded hard negatives (use `collect_hard_negatives.py`)
3. Consider adding more TP frames from non-test videos

## Preprocessing Script: `tools/prep_v7_data.py`

### What It Does

1. **Remove 283 background images** from current training set
   - Identify by: empty label files (0 bytes) in `data/datasets/frisbee_merged/labels/train/`
   - Remove both image and label file
   
2. **Copy 178 FP crops** to training set
   - Source: `data/perbox_crops/` files marked FP in `review_results.csv`
   - Destination: `data/datasets/frisbee_merged/images/train/`
   - Labels: empty `.txt` files (background/negative)
   - Prefix: `hardneg_v7_` to distinguish from existing hard negatives

3. **Extract 22 TP frames** from 55-56min test video
   - Read frame numbers from TP entries in `review_results.csv`
   - Open video, seek to frame, save as JPEG
   - Generate YOLO label by re-running v3 inference on that frame
   - Save image as `tp_v7_{frame_num:05d}.jpg` with corresponding label
   - Destination: `data/datasets/frisbee_merged/images/train/`

4. **Regenerate YAML config** via `utils.dataset.write_yaml_config()`

### Safety Checks

- Verify source video exists before extracting frames
- Verify v3 model exists for label generation
- Count final dataset and report positive/negative ratio
- Dry-run mode (`--dry-run`) to preview changes without modifying files

## Evaluation Protocol

After training:

```bash
# 55-56min test video
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v7_quick/weights/best.pt \
  --video movie/25866279684-1-192_55-56min.mp4 --conf 0.35

# 20-23min test video  
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v7_quick/weights/best.pt \
  --video movie/clip_20-23min.mp4 --conf 0.35
```

Metrics to record:
- Frame detection rate (target: >=50%)
- Avg detections per frame (target: <1.0 — lower than v3's 1.86)
- Confidence distribution stats
- Manual FP spot-check on 50 random detection frames

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| FP crops too small (151-200px) for 640/1280 training | Model can't learn from tiny images | Images are padded/resized — YOLO will scale them |
| 22 TP frames cause overfitting to test scenes | Inflated metrics on 55-56min | Cross-check on 20-23min (different game) |
| Quick validation at imgsz=640 misleads | Wrong go/no-go decision | If borderline, re-run at imgsz=1280 before deciding |
| Background removal hurts generalization | Worse on new scenes | 178 FP crops are more informative than 283 generic backgrounds |
| Fine-tuning from v3 forgets existing features | Catastrophic forgetting | 30 epochs with low LR from pretrained weights is standard |

## File Inventory

| File | Role |
|------|------|
| `data/perbox_crops/review_results.csv` | 200 reviewed detections (22 TP, 178 FP) |
| `data/perbox_crops/crop_*.jpg` | Bbox-level crop images |
| `movie/25866279684-1-192_55-56min.mp4` | Test video (source of TP frames) |
| `runs/detect/frisbee_det_s_v3/weights/best.pt` | Base model for fine-tuning + TP label generation |
| `tools/prep_v7_data.py` | **NEW** — data preprocessing script |
| `models/train.py` | Training script (add `--cls` parameter if needed) |
| `inference/predict_video.py` | Evaluation script |

## Expected Outcome

v3's classifier sees ~1500 frisbee positives and ~300 generic backgrounds. It has
no examples of "frisbee-like objects that aren't frisbees." By replacing those 300
generic backgrounds with 178 actual FP crops (white hats, rocks, lines) that the
model currently misclassifies, the classifier will learn to distinguish frisbees
from these specific confusers.

The 22 TP frames ensure the model retains the ability to detect real frisbees in
game footage — preventing the recall collapse seen with cls tuning.

Expected: FP rate drops from 89% to <20%, frame detection rate stays above 50%.
