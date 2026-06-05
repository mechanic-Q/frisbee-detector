# Current Superpowers Plan Evaluation - 2026-06-05

This note maps the active superpowers plans to the current repository state.
It uses existing local artifacts only; `data/` and `runs/` remain untracked and
must not be committed.

## Active Plan Status

| Plan item | Current status | Evidence | Next action |
| --- | --- | --- | --- |
| `review_labels.py` universal reviewer | Implemented | `tools/review_labels.py`, `tools/_label_utils.py`, `tests/test_label_utils.py` exist; `label_frames.py` removed | Keep as-is unless UI review finds issues |
| GSAM label review | Locally cleaned | `review_result.json` now has 46 accept, 68 reject, 0 skip, 114 reviewed; no accept/reject overlap | Do not commit `data/`; use cleaned local labels for training decisions |
| Merge reviewed game1080 labels | Superseded by product/pool merge | `frisbee-data/products/v1.yaml` includes `game1080`; `data/datasets/frisbee_merged/images/train` has 200 `game1080_*` images | Keep product-YAML path as source of truth |
| P2 model training | Completed beyond original v1 plan | `runs/detect/frisbee_det_p2_game_v2/weights/best.pt` exists | Evaluate before retraining |
| 1080p first60 evaluation | Partially done | Existing eval labels for `frisbee_det_p2_game_v2` on `videoplayback_first60s` | Rerun formal same-command benchmark if selecting a default model |
| Dataset verification | Fixed in code | Empty YOLO label files are now treated as valid negative/background samples | Run `tools/verify_dataset.py` before training |

## Review Result Cleanup

The local `data/datasets/game1080/review_result.json` had duplicates and one
accept/reject conflict:

| Issue | Resolution |
| --- | --- |
| Duplicate entries in accept/reject | Deduplicated while preserving order |
| `frame_0112` appeared in both accept and reject | Kept as accept because `labels/frame_0112.txt` is non-empty |
| `frame_0002` had a non-empty reviewed label but was absent from accept | Added to accept so state matches reviewed label output |

Final local review state:

| Bucket | Count |
| --- | ---: |
| Accept | 46 |
| Reject | 68 |
| Skip | 0 |
| Unique reviewed | 114 |

## Evaluation Table From Existing Artifacts

These metrics were computed from existing `runs/detect/runs/eval/*/labels`
files by counting frames with saved detections and detection lines per label
file. They are useful for direction, but exact run arguments are not recorded
in the label folders, so formal model selection should rerun the benchmark with
explicit commands.

| Model artifact | Video | Frames with detections | Detection rate | Detections/frame | Plan interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `frisbee_det_p2_game_v2` | `videoplayback_first60s.mp4` | 309 / 3646 | 8.5% | 0.10 | P2 v2 is weak on the 1080p holdout artifact |
| `frisbee_det_s_v7-3` | `videoplayback_first60s.mp4` | 797 / 3646 | 21.9% | 0.26 | Better than P2 v2 on this artifact, still low recall |
| `frisbee_det_s_v3-10` | `25866279684-1-192_55-56min.mp4` | 1048 / 1500 | 69.9% | 1.86 | High recall, many detections |
| `frisbee_det_s_v7_quick` | `25866279684-1-192_55-56min.mp4` | 803 / 1500 | 53.5% | 0.86 | Meets v7 quick recall target on this clip |
| `frisbee_det_s_v7-2` | `25866279684-1-192_55-56min.mp4` | 311 / 1500 | 20.7% | 0.22 | Production v7 artifact appears too conservative |
| `frisbee_det_s_v3-11` | `clip_20-23min.mp4` | 2690 / 4378 | 61.4% | 1.28 | v3 baseline recall target met, FP risk remains |
| `frisbee_det_s_v7_quick-2` | `clip_20-23min.mp4` | 2047 / 4378 | 46.8% | 0.67 | Slightly below v7 quick recall target |
| `frisbee_det_s_v7` | `clip_20-23min.mp4` | 516 / 4378 | 11.8% | 0.12 | Production v7 artifact appears too conservative |

## Formal Same-Command Benchmark

Completed on 2026-06-05 with the same video and confidence threshold:

```bash
python3 inference/predict_video.py --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
python3 inference/predict_video.py --model runs/detect/frisbee_det_s_v7/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
python3 inference/predict_video.py --model runs/detect/frisbee_det_s_v3/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
```

| Model artifact | Frames with detections | Detection rate | Detections/frame | Saved eval dir | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| `frisbee_det_p2_game_v2` | 309 / 3598 | 8.6% | 0.10 | `runs/detect/runs/eval/frisbee_det_p2_game_v2-2` | P2 v2 remains weak on the 1080p holdout |
| `frisbee_det_s_v7` | 797 / 3598 | 22.2% | 0.27 | `runs/detect/runs/eval/frisbee_det_s_v7-4` | Better than P2 v2, still low recall |
| `frisbee_det_s_v3` | 1217 / 3598 | 33.8% | 0.41 | `runs/detect/runs/eval/frisbee_det_s_v3-14` | Best recall of these three at `conf=0.35`, but still below earlier 720p v3 recall |

## Recommended Next Plan Step

P2 remains near the artifact-derived 8.5% detection rate, so the next
superpowers step is to retrain a cleaned-label P2 model. This is a long
training task and should not be started without explicit approval.

## P2 Retrain Preflight

Completed before starting any long training:

| Check | Result | Evidence |
| --- | --- | --- |
| Branch | OK | `git branch --show-current` -> `feat/p2-detector` |
| Dataset validity | OK | `python3 tools/verify_dataset.py configs/frisbee_merged.yaml` -> `Total issues: 0` |
| `tmux` | OK | `tmux 3.4` |
| GPU visibility | OK outside sandbox | `nvidia-smi` -> RTX 5080, 16GB; escalated PyTorch sees CUDA device 0 |
| P2 YAML | OK | `YOLO("yolov8s-p2.yaml")` constructs a `DetectionModel` |
| `game1080` samples | OK | `data/datasets/frisbee_merged/images/train` has 200 `game1080_*` images: 46 positive, 154 negative |
| Full-frame test-video hard negatives | Must fix before training | Current merged data contains 202 empty-label `hardneg_*` full-frame images at 1280x720, including 165 in train from `55-56min` / `clip_20-23min` test video names |

The merge script now needs to honor product-level `exclude` entries before the
next P2 retrain. The local product config has been adjusted to exclude
`hardneg`, but the merged dataset has not been regenerated yet because that
would replace existing generated data.
