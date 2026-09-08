"""RF-DETR-small 微调基准（Apache-2.0，纯开源备选代表）。
在 WSL 运行（走 GPU 队列）: bash tools/gpu_run.sh bench-rfdetr python3 tools/train_rfdetr.py
数据: data/datasets/frisbee_coco（由 yolo_to_coco.py 生成）
"""
from pathlib import Path

ROOT = Path("/mnt/e/frisbee-detector")
DATA = ROOT / "data/datasets/frisbee_coco"
OUT = ROOT / "data/bili_final_test/rfdetr_out"


def main():
    from rfdetr import RFDETRSmall
    model = RFDETRSmall()  # COCO 预训练默认权重（与 ultralytics s 级同一起跑线）
    model.train(
        dataset_dir=str(DATA),
        output_dir=str(OUT),
        epochs=30,
        batch_size=4,
        grad_accum_steps=1,
        resolution=672,      # 56 的倍数；1080P 小目标适当提高输入
        early_stopping=False,
    )
    print("rfdetr train done ->", OUT)


if __name__ == "__main__":
    main()
