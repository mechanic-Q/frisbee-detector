#!/usr/bin/env python3
"""GPU 空闲自动队列（F1 第三层：零样本轮换优先）。

顺序:
  1. wait: GPU 空闲守卫（显存 <2GB 且无 compute app）
  2. obj365 零样本: 金标准 100 帧推理 → --gt-v2 终评
  3. obj365 连续片段: 55-56min 300 帧检出 → 融合层（格⑥）
  4. 格④⑤: D-FINE(COCO)/race_v8sp2_v3 + 融合
完成后结果 JSON+对照视频留在 results/f1_matrix/，用户在会话查看。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MARK = ROOT / "results/f1_matrix/gpu_queue"
LOGD = ROOT / "results/f1_matrix/gpu_queue/logs"
PY = sys.executable
OBJ365 = ROOT / "data/bili_final_test/dfine_s_obj365.pth"
VIDEO = ROOT / "movie/25866279684-1-192_55-56min.mp4"


def gpu_idle() -> bool:
    """图形会话下 compute-apps 恒返回桌面进程列表（N/A 显存）——按显存增量判空闲。

    基线：队列启动时记录当前 used（桌面占用）。之后 used 超基线+2GB 视为忙。
    """
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        used = int(r.stdout.strip().splitlines()[0])
        return used < _baseline_mb + 2000
    except Exception:
        return False


LOCK_FILE = None  # 占位：gpu_idle 用模块级基线变量
_baseline_mb = 1200


def run(name: str, cmd: list[str], cwd: Path = ROOT) -> bool:
    mark = MARK / f"{name}.done"
    if mark.exists():
        print(f"[q] {name} 已完成，跳过", flush=True)
        return True
    print(f"[q] WAIT GPU idle for {name} ... {time.strftime('%H:%M:%S')}", flush=True)
    while not gpu_idle():
        time.sleep(120)
    log = open(LOGD / f"{name}.log", "w", encoding="utf-8")
    print(f"[q] START {name}", flush=True)
    rc = subprocess.run([PY, str(ROOT / "tools/gpu_run.py"), name, *map(str, cmd)],
                        cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT).returncode
    log.close()
    if rc == 0:
        mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        print(f"[q] DONE {name}", flush=True)
        return True
    print(f"[q] FAIL {name} exit={rc}（继续下一项）", flush=True)
    return False


def main() -> int:
    global _baseline_mb
    MARK.mkdir(parents=True, exist_ok=True)
    LOGD.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    _baseline_mb = int(r.stdout.strip().splitlines()[0])
    print(f"[q] 显存基线: {_baseline_mb} MB（超基线+2000MB 视为忙）", flush=True)

    # 1) obj365 零样本：金标准 100 帧
    if OBJ365.exists():
        run("obj365_zs_golden", [PY, "tools/golden_inference_detr.py",
                                 "--dfine-ckpt", str(OBJ365), "--dfine-name", "dfine_s_obj365_zs",
                                 "--dfine-cfg", "configs/dfine/dfine_hgnetv2_s_obj365cls.yml",
                                 "--skip-rfdetr"])
        run("obj365_metrics", [PY, "tools/golden_metrics.py", "--gt-v2"])
    else:
        print(f"[q] obj365 权重不存在: {OBJ365}", flush=True)

    # 2) obj365 连续片段 + 融合（格⑥）——复用 f1_cells45 的原生推理，但对象是 obj365 权重
    run("obj365_seq_fusion", [PY, "-c", f"""
import sys, json, time
sys.path.insert(0, r"{ROOT}")
sys.path.insert(0, r"{ROOT / 'D-FINE'}")
import os; os.chdir(r"{ROOT / 'D-FINE'}")
import cv2, torch
import torchvision.transforms.functional as TF
from src.core import YAMLConfig
from frisbee_analyzer.disc_fusion import fuse_disc_detections

cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_s_obj365cls.yml")
model = cfg.model
ck = torch.load(r"{OBJ365}", map_location="cpu")
model.load_state_dict(ck.get("model", ck)); model.eval().cuda()
cap = cv2.VideoCapture(r"{VIDEO}")
imgs = []
while len(imgs) < 300:
    ok, f = cap.read()
    if not ok: break
    imgs.append(f)
cap.release()
t0 = time.time()
dets = []
for img in imgs:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = TF.to_tensor(TF.resize(TF.to_pil_image(rgb), [640, 640])).unsqueeze(0).cuda()
    with torch.no_grad():
        r = model(t)
    s = r["pred_logits"].sigmoid().squeeze(-1)
    boxes = r["pred_boxes"][0]; h, w = img.shape[:2]
    frame_d = []
    for b, sc in zip(boxes.cpu().numpy(), s[0].cpu().numpy()):
        if float(sc) < 0.05: continue
        cx, cy, bw, bh = [float(v) for v in b]
        frame_d.append({{"bbox": [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h], "conf": round(float(sc),4)}})
    dets.append(frame_d)
dt = time.time() - t0
fused, st = fuse_disc_detections(dets, fps=25.0)
out = {{"raw": sum(len(d) for d in dets), "tracking": st.n_tracking, "predicting": st.n_predicting,
       "gated": st.n_gated, "longest": st.longest_track_frames, "detect_runtime_s": round(dt,1)}}
p = r"{ROOT / 'results/f1_matrix/obj365_seq_fusion.json'}"
open(p, "w", encoding="utf-8").write(json.dumps(out, indent=1, default=str))
print("saved", p, out)
"""])

    # 3) 格④⑤
    run("cells45", [PY, "tools/f1_cells45.py", "--max-frames", "300"])

    print("[q] ALL DONE — 结果见 results/f1_matrix/ + golden_metrics_v2.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
