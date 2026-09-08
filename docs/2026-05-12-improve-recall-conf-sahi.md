# Improve Recall: Lower Confidence Threshold + SAHI Sliced Inference

## Goal
- Fix v2 model's severe missed detections (4.3% frame detection rate) by lowering
  confidence threshold and applying SAHI sliced inference for small/distant frisbees

## Prerequisites
- v2 model at `runs/detect/runs/detect/frisbee_det_s_v2/weights/best.pt` (needs path fix)
- RTX 5080 16GB GPU
- Test videos: `movie/clip_20-23min.mp4`, `movie/25866279684-1-192_55-56min.mp4`

## Plan

### Step 0: Fix model paths (pre-flight)
**Why**: Model weights are at the doubled path `runs/detect/runs/detect/` (5.1GB)
but our config references the fixed path `runs/detect/`. Models won't load until
the files are moved.

**Actions**:
1. Move `runs/detect/runs/detect/frisbee_det_s_v2/` → `runs/detect/frisbee_det_s_v2/`
2. Move `runs/detect/runs/detect/frisbee_det_s/` → `runs/detect/frisbee_det_s/`
3. Move `runs/detect/runs/eval/` contents to appropriate location under `runs/detect/`
4. Remove empty `runs/detect/runs/` directory
5. Verify: `python -c "from configs.models import DEFAULT_MODEL; from ultralytics import YOLO; YOLO(str(DEFAULT_MODEL)); print('OK')"`

### Step 1: Baseline evaluation (conf=0.45)
**Why**: Establish comparison baseline with current default threshold.

**Actions**:
1. Run on clip_20-23min.mp4:
   ```
   python inference/predict_video.py --conf 0.45 --video movie/clip_20-23min.mp4
   ```
2. Run on 55-56min clip:
   ```
   python inference/predict_video.py --conf 0.45 --video movie/25866279684-1-192_55-56min.mp4
   ```
3. Record: frame detection rate, total detections, confidence distribution

### Step 2: Low confidence evaluation (conf=0.20)
**Why**: Test whether simply lowering the threshold improves recall without
introducing too many false positives.

**Actions**:
1. Run on clip_20-23min.mp4:
   ```
   python inference/predict_video.py --conf 0.20 --video movie/clip_20-23min.mp4
   ```
2. Run on 55-56min clip:
   ```
   python inference/predict_video.py --conf 0.20 --video movie/25866279684-1-192_55-56min.mp4
   ```
3. Compare with Step 1 results: frame detection rate diff, new det stats

**Expected**: Frame detection rate improves from ~4% to 15-30%. Multi-det
frames may appear. False positives possible at low confidence.

### Step 3: Install SAHI + add SAHI inference to predict_video.py
**Why**: SAHI slices large images into smaller overlapping tiles, enabling the
model to detect very small or distant frisbees that are invisible at full
resolution.

**Actions**:
1. Install via pip:
   ```
   pip install sahi --break-system-packages
   ```
2. Add `evaluate_sahi()` function to `inference/predict_video.py`:
   - Use `cv2.VideoCapture` to read video frame-by-frame
   - For each frame: `sahi.predict.get_sliced_prediction()` with:
     - `slice_height=640, slice_width=640`
     - `overlap_height_ratio=0.2, overlap_width_ratio=0.2`
     - `detection_model` from `sahi.AutoDetectionModel.from_pretrained()`
   - Collect same statistics as `evaluate_trained_model()`:
     - total frames, frames with detections, per-frame distribution, confidence stats
3. Add `--sahi` flag to CLI argparse
4. Add `sahi>=0.11.0` to `requirements.txt`

**Risk**: SAHI + ultralytics 8.4 compatibility needs verification.
SAHI processes frames 3-10x slower than plain YOLO.

### Step 4: SAHI evaluation (conf=0.20)
**Why**: Test whether sliced inference catches frisbees missed by full-image
inference.

**Actions**:
1. Run on clip_20-23min.mp4:
   ```
   python inference/predict_video.py --conf 0.20 --sahi --video movie/clip_20-23min.mp4
   ```
2. Run on 55-56min clip:
   ```
   python inference/predict_video.py --conf 0.20 --sahi --video movie/25866279684-1-192_55-56min.mp4
   ```
3. Compare with Step 2 results (same conf, no SAHI)

**Expected**: Additional 5-15% frame detection improvement for very small or
distant frisbees. Processing speed drops to ~1-3 fps.

### Step 5: Compare & decide
**Why**: Determine whether recall is now acceptable, or if further action
(training v3, YOLOv8m) is needed.

**Comparison matrix**:

| Run | Config | Expected Recall | Speed | FP Risk |
|-----|--------|----------------|-------|---------|
| Baseline | v2 + conf=0.45 | ~4% (known) | Fast | Low |
| Low conf | v2 + conf=0.20 | 15-30%? | Fast | Medium |
| SAHI | v2 + conf=0.20 + SAHI | 20-45%? | Slow | Medium |

**Decision tree**:
- If conf=0.20 achieves >80% recall on test clips → done, update DEFAULT_CONF
- If SAHI adds significant recall over conf=0.20 → make SAHI the default
- If false positives are too high at conf=0.20 → proceed to plan "方案2" (re-train v3 with lower pseudo-label threshold)
- If recall still <50% even with SAHI → proceed to plan "方案2" (re-train v3)

## Relevant Files
- `configs/models.py` — DEFAULT_MODEL, DEFAULT_CONF
- `configs/paths.py` — PROJECT_ROOT, EXTERNAL
- `inference/predict_video.py` — evaluate_trained_model, new evaluate_sahi
- `requirements.txt` — Python dependencies
- `requirements-optional.txt` — SAHI dependency (existing)
- `runs/detect/runs/detect/frisbee_det_s_v2/weights/best.pt` — v2 model (22MB)
- `movie/clip_20-23min.mp4` — test video 1
- `movie/25866279684-1-192_55-56min.mp4` — test video 2

## Risks
1. SAHI + ultralytics 8.4 compatibility unverified
2. SAHI on video: extrapolates to ~2-5min per 3-min clip at 1280px
3. conf=0.20 may produce thousands of false positives — need manual spot-check
4. Model path migration touches 5.1GB of data on NTFS mount (slow `mv`)
