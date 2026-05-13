# v6 Progressive Precision Improvement — Implementation Plan

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 从 v3 fine-tune 训练 v6 模型（cls=0.8, box=7.5），在测试视频上评估，根据结果决定是否需要 Round 2

**架构：** 渐进式两轮策略。Round 1 从 v3 权重 fine-tune，仅调 cls/box 参数。Round 2（条件触发）加入 hard negative mining + 两阶段 fine-tune。代码改动集中在 `models/train.py` 添加 `--freeze` 参数。

**技术栈：** YOLOv8 (ultralytics), Python 3, tmux (长时间训练)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `models/train.py` | Modify | Add `--freeze` parameter to function signature + argparse + model.train() call |
| `tests/test_train.py` | Create | Test that `--freeze` and `--cls` args are passed through correctly |
| `configs/models.py` | Modify | Add V6_MODEL path constant |
| `docs/superpowers/specs/2026-05-14-v6-progressive-precision-design.md` | Already exists | Design spec (no changes) |

---

### Task 1: Add `--freeze` parameter to train.py

**Files:**
- Modify: `models/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write failing test for freeze parameter passthrough**

Create `tests/test_train.py`:

```python
import os
import sys
import argparse
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train import train_frisbee_detector


def test_freeze_passed_to_model_train():
    with patch("models.train.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = "/tmp/test_run"
        mock_model.train.return_value = mock_results
        MockYOLO.return_value = mock_model

        train_frisbee_detector(
            data_yaml="fake.yaml",
            freeze=10,
            resume_from=None,
        )

        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["freeze"] == 10


def test_freeze_none_not_passed():
    with patch("models.train.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = "/tmp/test_run"
        mock_model.train.return_value = mock_results
        MockYOLO.return_value = mock_model

        train_frisbee_detector(
            data_yaml="fake.yaml",
            freeze=None,
            resume_from=None,
        )

        call_kwargs = mock_model.train.call_args[1]
        assert "freeze" not in call_kwargs or call_kwargs.get("freeze") is None


def test_cls_passed_to_model_train():
    with patch("models.train.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = "/tmp/test_run"
        mock_model.train.return_value = mock_results
        MockYOLO.return_value = mock_model

        train_frisbee_detector(
            data_yaml="fake.yaml",
            cls=0.8,
            resume_from=None,
        )

        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["cls"] == 0.8
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 -m pytest tests/test_train.py -v`
Expected: `test_freeze_passed_to_model_train` FAILS (TypeError: unexpected keyword argument 'freeze')

- [ ] **Step 3: Implement freeze parameter in train_frisbee_detector**

Modify `models/train.py` function signature — add `freeze` parameter:

```python
def train_frisbee_detector(data_yaml, model_size="s", epochs=100, imgsz=1280,
                           batch=8, resume_from=None, device=0, workers=4,
                           box=7.5, cls=0.5, close_mosaic=10, patience=20,
                           run_name=None, freeze=None):
```

Add to `model.train()` call — add this line after `exist_ok=True,`:

```python
        **({"freeze": freeze} if freeze is not None else {}),
```

Add to argparse section (after `--patience` line):

```python
    parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers for transfer learning")
```

Add `freeze=args.freeze,` to the `train_frisbee_detector()` call in `__main__` (after `run_name=args.name,`).

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_train.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run full test suite, verify no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: All tests PASS (no existing tests broken)

- [ ] **Step 6: Commit**

```bash
git add models/train.py tests/test_train.py
git commit -m "feat: add --freeze parameter to train.py for transfer learning"
```

---

### Task 2: Train v6 Round 1

**Files:**
- No code changes — training run only

- [ ] **Step 1: Launch training in tmux**

The training takes ~1.5h. Must use tmux because Bash tool has 10-min timeout.

```bash
tmux new-session -d -s train -c /mnt/e/frisbee-detector/.worktrees/improve-precision-v5
tmux send-keys -t train "python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --cls 0.8 --box 7.5 --batch 2 --workers 2 \
  --imgsz 1280 --epochs 100 --patience 15 \
  --resume /mnt/e/frisbee-detector/runs/detect/frisbee_det_s_v3/weights/best.pt \
  --name frisbee_det_s_v6" Enter
```

Note: `--resume` points to v3 model in the main repo (not worktree) because v3 weights are at `/mnt/e/frisbee-detector/runs/detect/frisbee_det_s_v3/weights/best.pt`. YOLO will output to `runs/detect/frisbee_det_s_v6/` in the worktree.

- [ ] **Step 2: Monitor training progress**

Check tmux session periodically:

```bash
tmux capture-pane -t train -p | tail -30
```

Look for: epoch progress, loss decreasing, no OOM errors. Expected: training starts with warmup (3 epochs), then converges. Patience=15 may early-stop around epoch 30-50.

- [ ] **Step 3: Wait for training to complete, verify output**

```bash
tmux capture-pane -t train -p | tail -10
```

Look for: "Best model saved to:" message. Then verify:

```bash
ls -la /mnt/e/frisbee-detector/.worktrees/improve-precision-v5/runs/detect/frisbee_det_s_v6/weights/best.pt
```

Expected: File exists, ~22MB.

- [ ] **Step 4: Fix double-nesting if needed**

YOLO's `project="runs/detect"` may create output at `runs/detect/runs/detect/frisbee_det_s_v6/`. Check and fix:

```bash
# Check if double-nested
ls /mnt/e/frisbee-detector/.worktrees/improve-precision-v5/runs/detect/runs/detect/frisbee_det_s_v6/ 2>/dev/null && echo "DOUBLE-NESTED" || echo "OK"
```

If double-nested:

```bash
mv /mnt/e/frisbee-detector/.worktrees/improve-precision-v5/runs/detect/runs/detect/frisbee_det_s_v6 /mnt/e/frisbee-detector/.worktrees/improve-precision-v5/runs/detect/frisbee_det_s_v6
```

- [ ] **Step 5: Record training metrics**

```bash
tmux capture-pane -t train -p -S -100 | grep -E "(mAP|Precision|Recall|Best|epoch)"
```

Record: mAP50, mAP50-95, Precision, Recall from validation.

---

### Task 3: Evaluate v6 on test videos

**Files:**
- No code changes — inference runs only

**Important:** Use the main repo's `predict_video.py` (the worktree version is outdated and missing argparse).

- [ ] **Step 1: Evaluate on 55-56min test clip**

```bash
python3 /mnt/e/frisbee-detector/inference/predict_video.py \
  --model /mnt/e/frisbee-detector/.worktrees/improve-precision-v5/runs/detect/frisbee_det_s_v6/weights/best.pt \
  --video /mnt/e/frisbee-detector/movie/clip_55-56min.mp4 \
  --conf 0.20
```

Record from output:
- Total frames processed
- Frames with detections (count + percentage)
- Average detections per frame
- Confidence distribution (how many < 0.50)

- [ ] **Step 2: Evaluate on 20-23min test clip**

```bash
python3 /mnt/e/frisbee-detector/inference/predict_video.py \
  --model /mnt/e/frisbee-detector/.worktrees/improve-precision-v5/runs/detect/frisbee_det_s_v6/weights/best.pt \
  --video /mnt/e/frisbee-detector/movie/clip_20-23min.mp4 \
  --conf 0.20
```

Record same metrics as Step 1.

- [ ] **Step 3: Compare against v3 baseline**

Create comparison table:

| Metric | v3 (baseline) | v6 R1 | Target |
|--------|:---:|:---:|:---:|
| Frame detection rate (55-56min) | 79.5% | ? | ≥ 50% |
| Dets/frame (55-56min) | 2.5 | ? | ≤ 1.8 |
| Frame detection rate (20-23min) | (unknown) | ? | — |
| Dets/frame (20-23min) | (unknown) | ? | — |

---

### Task 4: Decision gate — evaluate results

**Files:**
- No code changes

- [ ] **Step 1: Apply decision rules**

Based on 55-56min results:

**If detection ≥ 50% AND estimated FP ≤ 25%:**
→ Round 1 SUCCESS. Skip to Task 6.

**If detection 30-50% OR FP 25-40%:**
→ Partial improvement. Proceed to Task 5 (Round 2).

**If detection < 30%:**
→ cls still too aggressive. Round 2 should use cls=0.6-0.7. Proceed to Task 5.

- [ ] **Step 2: FP estimation (manual)**

If needed for decision: sample 50 random detection frames from the output video, visually inspect and count false positives. FP rate = FP_count / 50.

---

### Task 5: Round 2 — Hard Negative Mining + Two-Stage Fine-Tune (CONDITIONAL)

> **Only execute if Task 4 decides Round 1 did not meet success criteria.**
> This task involves significant manual work (reviewing frames). Detailed sub-steps
> will be planned after Round 1 results are known, using the design spec as reference:
> `docs/superpowers/specs/2026-05-14-v6-progressive-precision-design.md` Section 4.

- [ ] **Step 1: Collect FP frames from v3/v6 inference**

Run v3 on test videos at conf=0.10 to capture ALL suspect detections:

```bash
python3 /mnt/e/frisbee-detector/inference/predict_video.py \
  --model /mnt/e/frisbee-detector/runs/detect/frisbee_det_s_v3/weights/best.pt \
  --video /mnt/e/frisbee-detector/movie/clip_55-56min.mp4 \
  --conf 0.10 --save-dir /tmp/fp_collection_55
```

Repeat for 20-23min clip.

- [ ] **Step 2: Manual review — tag FP frames**

Review extracted frames. Target: 200-400 hard negatives. Focus on:
- White hats (most common FP)
- Round rocks
- Light patches on grass
- Other round/white objects

Copy selected frames to `data/datasets/frisbee_merged/images/train/` with empty label files.

- [ ] **Step 3: Stage 1 — Freeze backbone fine-tune (10 epochs)**

```bash
tmux new-session -d -s train-r2 -c /mnt/e/frisbee-detector/.worktrees/improve-precision-v5
tmux send-keys -t train-r2 "python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --cls 0.8 --box 7.5 --batch 2 --workers 2 \
  --imgsz 1280 --epochs 10 --patience 5 \
  --freeze 10 \
  --resume /mnt/e/frisbee-detector/runs/detect/frisbee_det_s_v3/weights/best.pt \
  --name v6b_stage1" Enter
```

- [ ] **Step 4: Stage 2 — Unfreeze fine-tune (30 epochs)**

```bash
tmux send-keys -t train-r2 "python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --cls 0.8 --box 7.5 --batch 2 --workers 2 \
  --imgsz 1280 --epochs 30 --patience 10 \
  --resume runs/detect/v6b_stage1/weights/best.pt \
  --name v6b_stage2" Enter
```

- [ ] **Step 5: Evaluate v6b on test videos**

Same evaluation as Task 3. Target: detection ≥ 50%, FP ≤ 20%.

---

### Task 6: Finalize — update configs and commit results

**Files:**
- Modify: `configs/models.py`

- [ ] **Step 1: Add V6_MODEL to configs/models.py**

Add after V3_MODEL line:

```python
V6_MODEL = RUNS_DIR / "frisbee_det_s_v6" / "weights" / "best.pt"
```

- [ ] **Step 2: Update DEFAULT_MODEL if results are good**

If v6 Round 1 succeeds (or Round 2 succeeds), update:

```python
DEFAULT_MODEL = V6_MODEL
```

- [ ] **Step 3: Commit**

```bash
git add configs/models.py
git commit -m "feat: add v6 model path to configs"
```

- [ ] **Step 4: Update wiki**

Update `/mnt/e/Agent_memory/agent-memory/concepts/frisbee-recognition-project.md`:
- Update model iteration table with v6 results
- Update log.md with training + evaluation results

- [ ] **Step 5: Merge improve-precision-v5 branch**

```bash
# From main repo
cd /mnt/e/frisbee-detector
git merge improve-precision-v5
```
