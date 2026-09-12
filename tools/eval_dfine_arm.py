#!/usr/bin/env python3
"""§13.20 三臂复测：D-FINE 权重在 chunk4 上跑原生推理 + 融合 + 可见区间口径。

用法: python tools/eval_dfine_arm.py <WEIGHTS_PTH> <TAG> [--res 640]
产物: results/f1_matrix/<TAG>_chunk4/  (disc 序列 + 指标 json)

对照基线: P(在管|可见)=49.3% (shadow_v1 全管线), tracking 17.7%
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "D-FINE"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from frisbee_analyzer.disc_fusion import fuse_disc_detections  # noqa: E402
from frisbee_analyzer.events_runner import load_calibration, px_to_world  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("tag")
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--low-conf", type=float, default=0.05)
    ap.add_argument("--conf", type=float, default=0.35, help="入档阈值（与 shadow 0.35 对齐）")
    ap.add_argument("--video", default=str(ROOT / "data/bili_final_test/accept_chunk_4.mp4"))
    ap.add_argument("--calib", default=str(ROOT / "results/f1_matrix/phaseA_chunk4/segment_calib.json"))
    args = ap.parse_args()

    out_dir = ROOT / "results/f1_matrix" / f"{args.tag}_chunk4"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_abs = str(Path(args.weights).resolve())

    # D-FINE 原生推理
    cwd = Path.cwd()
    os.chdir(ROOT / "D-FINE")
    from src.core import YAMLConfig  # noqa: E402
    cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_s_frisbee.yml")
    model = cfg.model
    ck = torch.load(weights_abs, map_location="cpu")
    state = ck.get("model", ck)
    model.load_state_dict(state, strict=False)
    model.eval().cuda()
    os.chdir(cwd)

    import torchvision.transforms.functional as TF  # noqa: E402

    def preprocess(frame):
        """训练用 SANitize(strategy='aspect') → 推理必须保持比例+padding（§13.20 实测：
        拉伸 resize 有检出率 46% vs 保持比例 87%——D-FINE 与 YOLO letterbox 不同需显式对齐）。"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = min(args.res / w, args.res / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img = cv2.resize(rgb, (nw, nh))
        canvas = np.full((args.res, args.res, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = img
        return TF.to_tensor(TF.to_pil_image(canvas)).unsqueeze(0).cuda(), scale

    cap = cv2.VideoCapture(str(args.video))
    seq = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t, scale = preprocess(frame)
        with torch.no_grad():
            r = model(t)
        scores = r["pred_logits"].sigmoid().squeeze(-1)[0].cpu().numpy()
        boxes = r["pred_boxes"][0].cpu().numpy()
        dets = []
        for b, sc in zip(boxes, scores):
            if float(sc) < args.conf:   # 与 shadow 臂同入档口径（disc_conf=0.35）
                continue
            cx, cy, bw, bh = (float(v) for v in b)
            # padding 坐标 → 原图（除以 scale）
            x1 = (cx - bw / 2) * args.res / scale
            y1 = (cy - bh / 2) * args.res / scale
            x2 = (cx + bw / 2) * args.res / scale
            y2 = (cy + bh / 2) * args.res / scale
            dets.append({"bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                         "conf": round(float(sc), 4), "source": "full"})
        # top-K 限制（与 shadow 臂 disc_tile_topk 同量级：每帧最多 2 个候选）
        dets.sort(key=lambda d: d["conf"], reverse=True)
        seq.append(dets[:2])
        n += 1
        if n % 600 == 0:
            print(f"  inferred {n} frames", flush=True)
    cap.release()
    del model
    torch.cuda.empty_cache()
    print(f"inference done: {n} frames")

    # 融合（与 shadow 臂同参数：续接窗 30 + 门拒续接 + 场线先验）
    # 尺寸从视频探测（§13.20 教训：硬编码 852x480 而实际 1920x1080 → 标定错位 →
    # 场线先验全拒 → 假性 P=14%）
    cap2 = cv2.VideoCapture(str(args.video))
    VW = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    VH = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap2.release()
    H, _, _ = load_calibration(args.calib, image_size=(VW, VH))
    proj = lambda cx, cy: px_to_world(H, cx, cy)  # noqa: E731
    fused, stats = fuse_disc_detections(seq, fps=30.0, world_projector=proj,
                                        width=VW, height=VH, field_margin_m=5.0)
    disc = {}
    for i, f in enumerate(fused):
        if f.status in ("tracking", "predicting") and f.bbox is not None:
            disc[str(i)] = {"bbox": [round(v, 1) for v in f.bbox], "conf": round(f.conf, 4),
                            "cx": round(f.cx, 1), "cy": round(f.cy, 1), "status": f.status,
                            "source": f.source}
    (out_dir / "disc_frames.json").write_text(json.dumps(disc), encoding="utf-8")

    # 可见区间口径
    vis = {int(k) for k in json.loads(
        (ROOT / "results/f1_matrix/visible_candidates_v2.json").read_text(encoding="utf-8"))}
    inpipe = {int(k) for k in disc}
    trk = {int(k) for k, v in disc.items() if v.get("status") == "tracking"}
    result = {
        "tag": args.tag, "weights": str(args.weights), "conf_threshold": args.conf,
        "fusion_stats": {k: v for k, v in stats.__dict__.items() if not k.startswith("details")},
        "P_inpipe_given_visible": round(len(vis & inpipe) / len(vis), 4),
        "P_tracking_given_visible": round(len(vis & trk) / len(vis), 4),
        "inpipe_all": round(len(inpipe) / n, 4),
        "visible_frames": len(vis),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print("baseline(shadow_v1 full-pipeline): P(在管|可见)=0.493, tracking=0.177")
    return 0


if __name__ == "__main__":
    sys.exit(main())
