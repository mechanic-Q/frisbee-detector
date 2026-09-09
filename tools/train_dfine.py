"""D-FINE-S 基准训练（Windows 原生，Apache-2.0 纯开源第二候选）。
运行: python tools/train_dfine.py
- 官方 COCO 预训练 dfine_s_coco.pth 起跑（与其他候选同"默认状态"原则）
- 同一数据 frisbee_coco（frisbee_merged_v2 的 COCO 形态）
- epochs=30 与其他候选对齐（官方默认 72 是 COCO 全量档，小数据集 30ep 合理）
- 分辨率 640（官方 S 默认档；672 属 RF-DETR 档）
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DFINE = ROOT / "D-FINE"
OUT = ROOT / "data/bili_final_test/dfine_out"
OUT.mkdir(parents=True, exist_ok=True)


def wait_gpu_free():
    import subprocess as sp
    while True:
        r = sp.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                   capture_output=True, text=True)
        if not r.stdout.strip():
            return
        print("[gpu-queue] D-FINE 等待 GPU 空闲...", flush=True)
        time.sleep(60)


import time  # noqa: E402


def main():
    wait_gpu_free()
    sys.path.insert(0, str(DFINE))
    os.chdir(DFINE)
    # 数据集配置指向 frisbee
    ds_cfg = DFINE / "configs/dataset/frisbee_detection.yml"
    model_cfg = DFINE / "configs/dfine/dfine_hgnetv2_s_coco.yml"
    # 预训练权重（官方 HF 直链，缺则用 HF hub 自动下载）
    cmd = [
        sys.executable, str(DFINE / "train.py"),
        "-c", str(model_cfg),
        "--use-amp", "--seed=42",
        "-t", str(ROOT / "data/bili_final_test/dfine_s_coco.pth"),
        "-o", f"output_dir={OUT}",
        f"epoches=30",
        f"train_dataloader.total_batch_size=8",
        f"train_dataloader.dataset.img_folder=E:/frisbee-detector/data/datasets/frisbee_coco/train/",
        f"train_dataloader.dataset.ann_file=E:/frisbee-detector/data/datasets/frisbee_coco/train/_annotations.coco.json",
        f"train_dataloader.dataset.transforms={{'type': 'Compose', 'ops': [{{'type': 'RandomPhotometricDistort', 'p': 0.5}}, {{'type': 'RandomZoomOut', 'fill': 0}}, {{'type': 'RandomIoUCrop', 'p': 0.8}}, {{'type': 'SanitizeBoundingBoxes'}}, {{'type': 'RandomHorizontalFlip'}}, {{'type': 'ConvertBoxes', 'fmt': 'cxcywh', 'normalize': True}}, {{'type': 'ConvertPILImage', 'dtype': 'float32', 'scale': True}}, {{'type': 'RandomChoice', 'transforms': [{{'type': 'RandomHSV'}}, {{'type': 'RandomBlur'}}], 'p': [0.6, 0.2]}}]}}, {{'type': 'SanitizeBoundingBoxes'}}]}}",
        f"val_dataloader.dataset.img_folder=E:/frisbee-detector/data/datasets/frisbee_coco/valid/",
        f"val_dataloader.dataset.ann_file=E:/frisbee-detector/data/datasets/frisbee_coco/valid/_annotations.coco.json",
        f"train_dataloader.collate_fn.batch_size=8",
    ]
    # 数据集配置也要覆盖 num_classes/remap
    import yaml
    ds = yaml.safe_load(ds_cfg.read_text())
    ds["num_classes"] = 1
    ds["remap_mscoco_category"] = False
    ds_cfg.write_text(yaml.dump(ds, sort_keys=False))

    print("CMD:", " ".join(cmd[:8]), "...", flush=True)
    r = subprocess.run(cmd)
    print("dfine train exit:", r.returncode, flush=True)
    (OUT / "train_exit_code.txt").write_text(str(r.returncode))


if __name__ == "__main__":
    main()
