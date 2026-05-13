# v6 Progressive Precision Improvement — Design Spec

**Date**: 2026-05-14
**Status**: Approved
**Branch**: `improve-precision-v5`

## TL;DR

v5 (cls=1.3 from scratch) collapsed recall to 1.4% on test video. The fix is a progressive
two-round strategy: first fine-tune from v3 with moderate cls=0.8 to validate the cls tuning
direction; if FPs remain high, add hard negatives + validation set improvements in Round 2.

Built on learning from v5 failure: cls=1.3 is too aggressive for single-class detection,
and training from scratch discards v3's proven frisbee-finding capability.

## 1. Problem Context

### History

| Version | Approach | Outcome |
|---------|----------|---------|
| v3 | Clean data, box=5, no data leak | 79.5% recall on test video, but ~60% FPs |
| v4 | cls=0.5 (default), box=5 | mAP50=0.815, baseline for tuning |
| v5 | cls=1.3, box=5, from scratch | mAP50=0.906 on val, **1.4% recall on test video** |

### v5 Failure Root Causes

1. **cls=1.3 too aggressive for single-class detection** — Ultralytics' recommended cls=1.33
   comes from 80-class COCO benchmarks. For a single-class detector, the classification loss
   has only one class to distinguish from background, making high cls penalties suppress
   nearly all detections.

2. **Training from scratch discarded v3's knowledge** — v3 had already learned to find
   frisbees (79.5% recall on test video). Starting from yolov8s.pt threw away this proven
   feature extraction capability.

3. **Validation set doesn't represent deployment** — val set is COCO/kaggle close-up
   frisbees. Test videos contain distant, fast-moving frisbees. v5 appeared to succeed
   on validation metrics while actually failing on real footage.

### v3 Proven Capability (keep this)

- 79.5% frame detection rate on 55-56min test clip
- The backbone's feature extractors reliably find frisbee-like objects
- Problem is **classification precision**, not **detection recall**

## 2. Overview

```
v3 weights ──→ [Round 1] cls=0.8 fine-tune ──→ v6 ──→ evaluate
                                                       ↓
                                        detection≥50% & FP≤25%?
                                                       ↓ yes → tune conf=0.30 → done
                                                       ↓ no
                                          [Round 2] hard negatives
                                            + two-stage fine-tune
                                            → v6b → evaluate → done
```

### Decision Rules at Evaluation Gate

| Condition | Action |
|-----------|--------|
| Detection ≥ 50% AND FP ≤ 25% | Round 1 success. Tune conf to 0.30. Done. |
| Detection 30-50% OR FP 25-40% | Partial improvement. Proceed to Round 2. |
| Detection < 30% | cls still too high. Round 2: lower cls to 0.6-0.7. |

## 3. Round 1 — Moderate cls Fine-Tune

### Training Configuration

| Parameter | v5 (failed) | v6 R1 | Rationale |
|-----------|:-----------:|:-----:|-----------|
| Starting point | yolov8s.pt | v3 best.pt | Preserve v3's frisbee feature detectors |
| cls | 1.3 | 0.8 | Midpoint between 0.5 (good recall) and 1.3 (killed recall) |
| box | 5.0 | 7.5 | v3 used box=5 to boost recall; restore default for box quality |
| epochs | 100 | 100 | Unchanged; patience=15 will early-stop |
| imgsz | 1280 | 1280 | Critical for distant small frisbees (50-100px) |
| batch | 2 | 2 | Hardware limit (RTX 5080 16GB) |
| optimizer | AdamW | AdamW | Stable convergence on small dataset |
| lr0 | 0.001 | 0.001 | Fine-tune rate — low to preserve learned features |
| warmup_epochs | 3 | 3 | Prevents early batches from damaging v3 features |
| mosaic | 0.5 | 0.5 | Keeps small-object augmentation |
| close_mosaic | 10 | 10 | Last 10 epochs on normal images for stable convergence |
| patience | 20 | 15 | Tighter early-stopping to save time |
| Dataset | frisbee_merged | frisbee_merged | Unchanged — 4950 images, 16.6% negatives |

### Key Design Decisions

- **Fine-tune from v3, not from scratch**: v3's backbone already knows how to locate
  frisbees. The goal is to refine classification decision boundaries, not relearn detection.
- **cls=0.8**: Computational compromise between 0.5 (too permissive) and 1.3 (too
  restrictive). For single-class detection, the effective penalty is higher than the
  same value on multi-class problems.
- **box=7.5** (default): Restore normal box quality requirements. False positives tend to
  have poorer bounding box quality than true positives, so higher box weight indirectly
  suppresses FPs.
- **No data changes in R1**: First validate whether cls=0.8 alone can achieve acceptable
  FP rates. Adding hard negatives is the plan B.

### Train Command

```bash
python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --box 7.5 --batch 2 --workers 2 \
  --imgsz 1280 --epochs 100 --patience 15 \
  --resume runs/detect/frisbee_det_s_v3/weights/best.pt \
  --name frisbee_det_s_v6
```

Note: `--cls` and `--freeze` are already on the `improve-precision-v5` branch.

### Evaluation

Same conditions as v3 baseline for fair comparison:

```bash
# 55-56min test clip
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v6/weights/best.pt \
  --video movie/clip_55-56min.mp4 --conf 0.20

# 20-23min test clip
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v6/weights/best.pt \
  --video movie/clip_20-23min.mp4 --conf 0.20
```

FP estimation: sample 50 random detection frames, manual visual inspection.

### Success Criteria

| Metric | v3 (baseline) | v6 R1 Target | R2 Trigger |
|--------|:-------------:|:------------:|:----------:|
| Frame detection rate | 79.5% | ≥ 50% | < 30% |
| Dets/frame | 2.5 | ≤ 1.8 | > 2.0 |
| Estimated FP rate | ~60% | ≤ 25% | > 30% |

## 4. Round 2 — Hard Negative Mining + Two-Stage Fine-Tune

Only executed if Round 1 does not achieve detection ≥ 50% AND FP ≤ 25%.

### 4.1 Data Preparation

**Source**: Run v3 on both test videos at conf=0.10 (low threshold to surface all suspect
detections), extract detection frames.

**Manual review**: Review extracted frames, tag false positives (white hats, rocks, light
patches, etc.). Target: 200-400 hard negative frames.

**Labeling**: Each hard negative frame gets an **empty label file** (no bounding boxes).
This teaches the model: "this scene contains no frisbee, do not detect anything here."

**Focus on quality**: Prioritize frames containing specific confusers (white hats, round
rocks) over random empty field frames. The model learns most from the scenes it currently
fails on.

**Game-footage validation set** (time permitting): Annotate 100-200 frames from game
footage with frisbee bounding boxes for a deployment-representative validation set.

### 4.2 Updated Dataset

```
data/datasets/frisbee_merged/
├── images/
│   ├── train/           # Original 3960 images
│   ├── train_hard_neg/  # 200-400 hard negative frames (empty labels)
│   ├── val/             # Original 495 images
│   ├── val_game/        # 100-200 game-footage validation frames (optional)
│   └── test/            # Original 495 images
└── labels/              # Parallel structure to images/
```

### 4.3 Two-Stage Fine-Tuning

**Why two-stage**: Stage 1 preserves v3's proven backbone while teaching the detection
head from hard negatives. Stage 2 then gently refines everything with a lower learning
rate to avoid catastrophic forgetting.

#### Stage 1: Freeze Backbone (10 epochs)

```bash
python3 models/train.py \
  --data configs/frisbee_merged_v6.yaml \
  --cls 0.8 --box 7.5 --batch 2 --workers 2 \
  --imgsz 1280 --epochs 10 --patience 5 \
  --resume runs/detect/frisbee_det_s_v3/weights/best.pt \
  --freeze 10 \
  --name v6_stage1
```

- `freeze=10`: Freezes first 10 layers (backbone). Only detection head trains.
- Short run — just long enough to learn from hard negatives without overfitting.

#### Stage 2: Unfreeze All (30 epochs)

```bash
python3 models/train.py \
  --data configs/frisbee_merged_v6.yaml \
  --cls 0.8 --box 7.5 --batch 2 --workers 2 \
  --imgsz 1280 --epochs 30 --patience 10 \
  --resume runs/detect/v6_stage1/weights/best.pt \
  --lr0 0.0005 \
  --name v6_stage2
```

- `lr0=0.0005`: Half the normal fine-tune rate — gentle refinement of all layers.
- Unfreeze by not passing `--freeze`.

### 4.4 Evaluation (Same as Round 1)

| Metric | v3 (baseline) | v6 R2 Target |
|--------|:-------------:|:------------:|
| Frame detection rate | 79.5% | ≥ 50% |
| Dets/frame | 2.5 | ≤ 1.5 |
| Estimated FP rate | ~60% | ≤ 20% |

If R2 still fails to achieve targets, fallback to the v3 model with inference-side
confidence threshold tuning (conf=0.30+) as a pragmatic minimum-viable solution.

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| cls=0.8 still too aggressive for single-class | Medium | High | Track detection rate closely; if <30%, Round 2 uses cls=0.6-0.7 |
| Fine-tune from v3 instead of scratch changes loss landscape | Low | Medium | warmup_epochs=3 + low lr0 protect v3 features |
| Hard negative volume insufficient (don't collect 200+ frames) | Medium | Medium | Start collection early during R1 training |
| Game-footage annotation too time-consuming | High | Low | Annotate only if time permits; R2 can work without it |
| RTX 5080 16GB OOM | Low | High | workers=2 proven safe; batch=2 is max; never use workers=4 |

## 6. File Changes

| File | Change | When |
|------|--------|------|
| `models/train.py` | Add `--cls` parameter (already done in v5) | v5 commit |
| `models/train.py` | Add `--freeze` parameter for R2 Stage 1 | v6 commit |
| `configs/frisbee_merged.yaml` | Add `train_hard_neg` paths (R2 only) | R2 only |
| `configs/models.py` | Add V6_MODEL path | After training |
| `inference/predict_video.py` | No changes needed | — |

### New `--freeze` Parameter in train.py

Add to `train_frisbee_detector()` signature:
```python
freeze: int | None = None  # Number of layers to freeze (None = train all)
```

Add to `model.train()` call (conditionally — only when freeze is not None):
```python
**({"freeze": freeze} if freeze is not None else {}),
```

Add to `argparse`:
```python
parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers")
```

## 7. References

- [Ultralytics Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/) — cls optimal search space
- [Ultralytics Fine-Tuning Guide](https://docs.ultralytics.com/guides/finetuning-guide/) — two-stage freeze/unfreeze strategy
- [Ultralytics #3466](https://github.com/ultralytics/ultralytics/issues/3466) — Glenn Jocher on hard negative mining
- [Ultralytics #18809](https://github.com/ultralytics/ultralytics/issues/18809) — background image ratio recommendations
- [Ultralytics #10207](https://github.com/ultralytics/ultralytics/issues/10207) — cls loss weight for class imbalance
- `.sisyphus/plans/reduce-false-positives-v5.md` — Original v5 FP mitigation plan
- `docs/superpowers/specs/2026-05-13-improve-precision-v5-design.md` — v5 design spec
