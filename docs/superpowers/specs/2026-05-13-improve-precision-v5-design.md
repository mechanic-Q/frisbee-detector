# v5 Precision Improvement — Design Spec

> **Goal:** Reduce false positives (white hats → frisbee) from ~60% to <15%
> while keeping frame detection rate at 60-70%.

## Summary

v3/v4 models achieve good recall (68-79%) but produce ~60% false positives —
mostly white hats and rocks mistaken for frisbees. The root cause is `cls` loss
weight at YOLOv8's default 0.5, which barely penalizes classification errors.

**Solution:** Retrain with `cls=1.3` (Ultralytics official optimal value),
using the existing 4950-image merged dataset. If that's not enough, expand
with OpenImages Flying disc images as backup (v6).

## Architecture

```
主线 ──→ 训 v5 (cls=1.3, 现有合并4950张, 其余参数同v4)
              │
              ▼
         评估 v5 vs v4（验证集 + 两个测试视频）
              │
         ┌────┴────┐
         │ FP<15%? │
         ├────┬────┤
         │ 是 │ 否 │
         ▼    ▼
       完成  OpenImages 865张 → 训 v6

支线 ──→ 重新下载 OpenImages Flying disc 865张
         (重做 ID 收集 → 并发下载 → 转 YOLO 格式 → 记录 wiki)
```

## Data Inventory (Verified)

| Source | Images | Type |
|--------|:------:|------|
| frisbee_ultimateml | 1001 | positive |
| frisbee_kaggle | 851 | positive |
| frisbee_coco (COCO frisbee) | 2713 | positive |
| frisbee_negatives | 80 | negative |
| frisbee_pseudo | 103 | mixed |
| frisbee_coco_neg | 445 | negative |
| frisbee_hard_neg | 202 | negative |
| **Total merged** | **4950** | — |
| **Negative ratio** | **16.6%** | ✅ already in range |

## Training Parameters

| Parameter | v4 | v5 | Change |
|-----------|:--:|:--:|--------|
| `cls` | 0.5 (default) | **1.3** | ✅ key change |
| `box` | 5 | 5 | same |
| `close_mosaic` | 10 | 10 | same |
| `epochs` | 100 | 100 | same |
| `batch` | 2 | 2 | same |
| `imgsz` | 1280 | 1280 | same |
| `patience` | 20 | 20 | same |
| `optimizer` | AdamW | AdamW | same |
| `data` | `configs/frisbee_merged.yaml` | same | same |

**Prerequisite:** `train.py` must add `--cls` argparse parameter (currently missing).

## Success Criteria

| Metric | v3/v4 | v5 Target | Measurement |
|--------|:-----:|:---------:|-------------|
| Frame detection rate | 79.5% | 60-70% | Script auto |
| Dets per frame | 2.5 | 1.0-1.3 | Script auto |
| FP rate | ~60% | **<15%** | Manual spot-check 100 frames |
| Test video 1 | 55-56min | 55-56min | `predict_video.py` |
| Test video 2 | — | 20-23min | `predict_video.py` |

FP rate spot-check method: sample 50 frames from each test video (100 total),
manually verify each detection — frisbee vs non-frisbee.

## Steps

### Step 1: Add `--cls` to `models/train.py`

Add `--cls` argparse argument with default 0.5, wire into `train_frisbee_detector()`.

### Step 2: Train v5

```bash
python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --model-size s \
  --name frisbee_det_s_v5 \
  --epochs 100 --imgsz 1280 --batch 2 \
  --patience 20 --box 5 --cls 1.3 --close-mosaic 10
```

Run in tmux (~1.5-2.5h). Handle double-nesting bug (move model after training).

### Step 3: Validate

```bash
python3 models/train.py --validate-only \
  --model-path runs/detect/frisbee_det_s_v5/weights/best.pt \
  --data configs/frisbee_merged.yaml
```

### Step 4: Evaluate on test videos

```bash
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v5/weights/best.pt \
  --video movie/25866279684-1-192_55-56min.mp4 --conf 0.20

python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v5/weights/best.pt \
  --video movie/clip_20-23min.mp4 --conf 0.20
```

### Step 5: Decision

- **PASS** (dets/frame < 1.3, FP < 15%): Done.
- **FAIL**: Proceed to OpenImages download → train v6.

## OpenImages Download (Backup Plan)

1. Collect all Flying disc image IDs from existing text files in `openimages_frisbee/`
2. Download via CVDF mirror (not Google Storage — blocked)
3. Parse bbox from `oidv6-train-annotations-bbox.csv`, convert to YOLO format
4. Manual QC 20 images, record download summary to wiki
5. Add `frisbee_openimages` as new source in `merge_datasets.py` SOURCES list
6. Re-merge → train v6 with same params

**Risk:** CVDF mirror may also return 403. If so, terminate this path.

## Risks

| Risk | Mitigation |
|------|-----------|
| cls=1.3 alone insufficient | OpenImages backup (v6) |
| v5 recall drops too far | v5 plan accepts tradeoff; frame detection 60-70% is target |
| v4 early-stopped at 31 epochs | v5 may also stop early — accept and evaluate |
| OpenImages download blocked | Terminate gracefully, record failure to wiki |

## References

- [Ultralytics Hyperparameter Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/) — cls optimal 1.33
- [Ultralytics #3466](https://github.com/ultralytics/ultralytics/issues/3466) — hard negative mining
- [Ultralytics #10207](https://github.com/ultralytics/ultralytics/issues/10207) — cls loss weight for class imbalance
- `.sisyphus/plans/reduce-false-positives-v5.md` — prior FP mitigation plan
