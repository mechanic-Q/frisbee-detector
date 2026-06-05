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

## Recommended Next Plan Step

Do not start long P2 retraining yet. First run a formal same-command benchmark
on `videoplayback_first60s.mp4` for the candidate models:

```bash
python3 inference/predict_video.py --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
python3 inference/predict_video.py --model runs/detect/frisbee_det_s_v7/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
python3 inference/predict_video.py --model runs/detect/frisbee_det_s_v3/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
```

If P2 remains near the artifact-derived 8.5% detection rate, the next
superpowers step is to retrain a cleaned-label P2 model, not to merge the branch.
