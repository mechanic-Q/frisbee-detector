"""分析 worker CLI：stdout 输出 JSON-lines 进度，产物为 per-video tracks.json。

在 WSL 运行（也接受 Windows 路径，内部自动转换）：
    python3 -m frisbee_analyzer.pipeline \
        --video 'E:\\frisbee-detector\\data\\bili_final_test\\testclip_60_120s.mp4'

GUI 通过 QProcess 启动本模块并逐行解析 stdout（协议见 frisbee_analyzer/protocol.py）。
退出码：0 成功；1 出错；2 被 GUI 取消。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import protocol
from .protocol import win_to_wsl, wsl_to_win
from .tracking import WorkerCancelled, iter_player_tracks, probe_video

PROGRESS_EVERY = 30  # 每 N 帧上报一次进度（GUI 进度条足够平滑）


def run(args, stream=None) -> int:
    """执行一次分析。stream 参数便于测试注入。返回退出码。"""
    video = win_to_wsl(args.video)
    if not Path(video).exists():
        protocol.emit(protocol.error(f"video not found: {video}"), stream)
        return 1

    info = probe_video(video)
    total = info["total_frames"]
    if args.max_frames:
        total = min(total, args.max_frames)
    protocol.emit(protocol.meta(total, info["fps"], info["width"], info["height"]), stream)

    # M2 接入点：分队模块对 frames 做后处理，填充 team_id（见 team.py）
    from .team import assign_teams  # 惰性导入，保持 --help 轻量

    frames: dict[str, list] = {}
    for idx, dets in iter_player_tracks(
        video,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
    ):
        frames[str(idx)] = dets
        if idx % PROGRESS_EVERY == 0:
            protocol.emit(protocol.progress(idx, total), stream)
    protocol.emit(protocol.progress(total, total), stream)

    assign_teams(frames, video, cancel_check=None, log=lambda m: protocol.emit(protocol.log(m), stream))

    out_dir = Path(win_to_wsl(args.output_dir)) if args.output_dir else Path("runs/gui_analysis") / Path(video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tracks.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    doc = {
        "video": video,
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "team_overrides": {},
        "frames": frames,
    }
    tmp_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)  # 原子写：中断不会留下半截 JSON

    protocol.emit(protocol.result(wsl_to_win(str(out_path))), stream)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="frisbee match analysis worker (JSON-lines on stdout)")
    parser.add_argument("--video", required=True, help="视频路径（Windows 或 WSL 风格均可）")
    parser.add_argument("--output-dir", default=None, help="产物目录（默认 runs/gui_analysis/<视频名>）")
    parser.add_argument("--weights", default="yolo26x.pt", help="检测权重（零样本 COCO person 或 player/referee 微调）")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=None, help="调试用：只处理前 N 帧")
    args = parser.parse_args(argv)

    try:
        return run(args)
    except WorkerCancelled as e:
        protocol.emit(protocol.error(f"cancelled: {e}"))
        return 2
    except Exception as e:  # noqa: BLE001 —— worker 顶层兜底，任何崩溃都要回传 GUI
        protocol.emit(protocol.error(f"{type(e).__name__}: {e}"))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
