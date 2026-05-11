"""从多个视频按指定间隔抽帧，保存为 jpg 图片。

输出: /mnt/e/frisbee-detector/data/frames/<video_name>/
"""
import cv2
from pathlib import Path

VIDEO_CONFIGS = [
    {
        "path": "/mnt/e/frisbee-detector/movie/25866279684-1-192.mp4",
        "interval": 5,      # 每5秒抽1帧
        "max_frames": 1200,
    },
    {
        "path": "/mnt/e/frisbee-detector/movie/clip_20-23min.mp4",
        "interval": 1,      # 每1秒抽1帧
        "max_frames": 200,
    },
    {
        "path": "/mnt/e/firsbee/03_datasets/frisbee-tracking/clip-5.mp4",
        "interval": 0.2,    # 每0.2秒抽1帧（5fps）
        "max_frames": 100,
    },
    {
        "path": "/mnt/e/firsbee/03_datasets/frisbee-vision-project/Footage/backhand_2.mp4",
        "interval": 0.2,
        "max_frames": 100,
    },
]


def extract_frames(video_path, output_dir, interval_sec, max_frames):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  SKIP: {video_path} not found")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps * interval_sec))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    saved = 0
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            out_path = output_dir / f"{video_path.stem}_frame_{saved:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        count += 1

    cap.release()
    print(f"  {video_path.name}: {saved} frames (interval={interval_sec}s, fps={fps})")
    return saved


def main():
    output_base = Path("/mnt/e/frisbee-detector/data/frames")
    total = 0
    for config in VIDEO_CONFIGS:
        video_name = Path(config["path"]).stem
        out_dir = output_base / video_name
        n = extract_frames(config["path"], out_dir, config["interval"], config["max_frames"])
        total += n
    print(f"\nTotal frames extracted: {total}")


if __name__ == "__main__":
    main()
