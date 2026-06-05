"""用当前 best.pt 模型对抽帧图片进行自动标注（pseudo-labeling）。

规则:
- conf >= 0.60 → 保存为 YOLO 标签（正样本）
- 无检测 → 创建空标签文件（负样本）
- 使用 ultralytics Boxes.xywhn 获取归一化坐标（0-1）

输出: /mnt/e/frisbee-detector/data/datasets/frisbee_pseudo/
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "/mnt/e/frisbee-detector/runs/detect/runs/detect/frisbee_det_s/weights/best.pt"
FRAMES_DIR = "/mnt/e/frisbee-detector/data/frames"
OUTPUT_DIR = "/mnt/e/frisbee-detector/data/datasets/frisbee_pseudo"
CONF_THRESHOLD = 0.60


def main():
    model = YOLO(MODEL_PATH)
    print(f"Model: {MODEL_PATH}")
    print(f"Conf threshold: {CONF_THRESHOLD}")

    frames_path = Path(FRAMES_DIR)
    image_files = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        image_files.extend(frames_path.rglob(f"*{ext}"))
    image_files = sorted(image_files)
    total = len(image_files)
    print(f"Found {total} frame images")

    if not image_files:
        print("No images found. Run tools/extract_frames.py first.")
        return

    out_img_dir = Path(OUTPUT_DIR) / "images" / "train"
    out_lbl_dir = Path(OUTPUT_DIR) / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    labeled_count = 0
    negative_count = 0

    # Process one by one to preserve file paths
    for idx, src_path in enumerate(image_files):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{total} ({idx*100//total}%)")

        rel_path = src_path.relative_to(frames_path)

        dst_img = out_img_dir / rel_path.with_suffix(".jpg")
        dst_lbl = out_lbl_dir / rel_path.with_suffix(".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(str(src_path), str(dst_img))

        results = model.predict(
            source=str(src_path),
            conf=CONF_THRESHOLD,
            save=False,
            save_txt=False,
            verbose=False,
        )

        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            lines = []
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x_center = float(box.xywhn[0][0])
                y_center = float(box.xywhn[0][1])
                width = float(box.xywhn[0][2])
                height = float(box.xywhn[0][3])
                lines.append(
                    f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
            if lines:
                with open(dst_lbl, "w") as f:
                    f.write("\n".join(lines) + "\n")
                labeled_count += 1
            else:
                dst_lbl.touch()
                negative_count += 1
        else:
            dst_lbl.touch()
            negative_count += 1

    print("\nAuto-labeling complete:")
    print(f"  Labeled (>= {CONF_THRESHOLD}): {labeled_count}")
    print(f"  Negatives (no detection):    {negative_count}")
    print(f"  Total:                       {labeled_count + negative_count}")

    # 统计 bboxes 大小分布
    areas = []
    for lbl in out_lbl_dir.rglob("*.txt"):
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    area = float(parts[3]) * float(parts[4])
                    areas.append(area)
    if areas:
        areas.sort()
        print(f"\nBbox stats ({len(areas)} total):")
        print(f"  Min:   {areas[0]:.6f}")
        print(f"  Max:   {areas[-1]:.6f}")
        print(f"  Mean:  {sum(areas)/len(areas):.6f}")
        print(f"  Median:{areas[len(areas)//2]:.6f}")
        too_big = sum(1 for a in areas if a > 0.10)
        print(f"  >10% area: {too_big} ({too_big/len(areas)*100:.1f}%)")

    yaml_content = f"path: {OUTPUT_DIR}\ntrain: images/train\nval: images/train\ntest: images/train\nnc: 1\nnames: ['frisbee']\n"
    with open(Path(OUTPUT_DIR) / "frisbee.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
