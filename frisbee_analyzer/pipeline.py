"""分析 worker CLI：stdout 输出 JSON-lines 进度，产物为 per-video tracks.json。

Windows 原生运行（不再经 WSL；路径按原样使用）：
    python -m frisbee_analyzer.pipeline --video E:/frisbee-detector/data/bili_final_test/testclip_60_120s.mp4

GUI 通过 QProcess 启动本模块并逐行解析 stdout（协议见 frisbee_analyzer/protocol.py）。
退出码：0 成功；1 出错；2 被 GUI 取消。
"""

from __future__ import annotations

import os

# Windows 原生: torch 与其他库各带一份 OpenMP 运行时(libiomp5md.dll),不设会 OMP Error #15 崩溃。
# 必须在导入 torch(.tracking)之前生效。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import protocol
from .tracking import WorkerCancelled, iter_player_tracks, probe_video

PROGRESS_EVERY = 30  # 每 N 帧上报一次进度（GUI 进度条足够平滑）


def run(args, stream=None) -> int:
    """执行一次分析。stream 参数便于测试注入。返回退出码。"""
    video = args.video
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
    disc_frames: dict[str, dict] = {}
    if getattr(args, "disc_weights", None):
        # 双模型路径：球员 + 飞盘联合跟踪（事件统计的前提，见 docs/2026-09-09-engine-validation.md）
        from .tracking import iter_player_and_disc_tracks

        for idx, p_dets, d_det in iter_player_and_disc_tracks(
            video,
            player_weights=args.weights,
            disc_weights=args.disc_weights,
            conf=args.conf,
            disc_conf=args.disc_conf,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
            player_classes=tuple(args.classes),
        ):
            frames[str(idx)] = p_dets
            if d_det is not None:
                disc_frames[str(idx)] = d_det
            if idx % PROGRESS_EVERY == 0:
                protocol.emit(protocol.progress(idx, total), stream)
    else:
        for idx, dets in iter_player_tracks(
            video,
            weights=args.weights,
            conf=args.conf,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
            classes=tuple(args.classes),
            tile_grid=args.tile_grid,
            tile_conf=args.tile_conf,
        ):
            frames[str(idx)] = dets
            if idx % PROGRESS_EVERY == 0:
                protocol.emit(protocol.progress(idx, total), stream)
    protocol.emit(protocol.progress(total, total), stream)
    if disc_frames:
        protocol.emit(protocol.log(f"disc: {len(disc_frames)}/{len(frames)} frames with detection"), stream)

    if args.team_from_cls:
        from .team import apply_team_from_cls

        n = apply_team_from_cls(frames)
        team_colors = {}
        protocol.emit(protocol.log(f"team: {n} dets assigned from detector classes "
                                   f"{list(args.classes)}"), stream)
    else:
        from .team import assign_teams

        team_colors = assign_teams(frames, video, cancel_check=None,
                                   log=lambda m: protocol.emit(protocol.log(m), stream))

    # 场内过滤（默认开启）：观众区/过小框不进入产物——方案 §8.1 场地过滤近似。
    # 有标定 → 场地多边形过滤（脚点反投影，远场球员不再被 y 阈值误切，迭代6 实测 89,933 框）；
    # 无标定 → 放宽启发式（min_y_frac 0.45→args.min_y_frac，默认 0.30）。
    from .filters import filter_dets, FIELD_POLYGON_M

    fh = info["height"]
    matrix = polygon = None
    if getattr(args, "calibration", None):
        try:
            from utils.homography import load_calibration

            calib = load_calibration(args.calibration)
            matrix, polygon = calib["matrix"], FIELD_POLYGON_M
        except Exception as e:  # noqa: BLE001 —— 标定读取失败则退回启发式
            protocol.emit(protocol.log(f"field filter: calibration ignored ({e})"), stream)
    if getattr(args, "no_field_filter", False):
        protocol.emit(protocol.log("field filter: DISABLED (--no-field-filter, raw dets saved)"), stream)
    else:
        total_before = sum(len(d) for d in frames.values())
        for key, dets in frames.items():
            frames[key] = filter_dets(
                dets, fh, matrix=matrix, polygon=polygon,
                min_height=fh * 0.083, min_y_frac=args.min_y_frac,
            )
        total_after = sum(len(d) for d in frames.values())
        route = "polygon(calibrated)" if matrix is not None else f"heuristic(min_y_frac={args.min_y_frac})"
        protocol.emit(protocol.log(f"field filter({route}): {total_before} -> {total_after} dets"), stream)

    out_dir = Path(args.output_dir) if args.output_dir else Path("runs/gui_analysis") / Path(video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tracks.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    doc = {
        "video": video,
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "team_colors": team_colors,
        "team_overrides": {},
        "frames": frames,
        "disc_frames": disc_frames,
    }

    # 事件统计（需飞盘轨迹；无盘则不产生事件——见验证报告）
    if disc_frames and getattr(args, "calibration", None):
        try:
            from .events_runner import compute_events

            events_doc = compute_events(doc, Path(args.calibration))
            doc["events"] = events_doc["events"]
            doc["score"] = events_doc["score"]
            protocol.emit(protocol.log(
                f"events: {len(events_doc['events'])} (score={events_doc['score']})"), stream)
        except Exception as e:  # noqa: BLE001 —— 事件统计失败不拖垮产物落盘
            protocol.emit(protocol.log(f"events: failed ({type(e).__name__}: {e})"), stream)

    tmp_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)  # 原子写：中断不会留下半截 JSON

    protocol.emit(protocol.result(str(out_path)), stream)
    return 0


def run_team_only(args, stream=None) -> int:
    """只重算分队（复用已有 tracks.json 的跟踪结果），用于换分队算法后免重跑 50 分钟跟踪。"""
    from .team import assign_teams

    doc_path = Path(args.team_only)
    if not doc_path.exists():
        protocol.emit(protocol.error(f"tracks.json not found: {doc_path}"), stream)
        return 1
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    video = doc.get("video")
    if not video or not Path(video).exists():
        protocol.emit(protocol.error(f"video not found: {video}"), stream)
        return 1
    frames = doc.get("frames", {})
    protocol.emit(protocol.meta(len(frames), doc.get("fps") or 30.0,
                                doc.get("width") or 0, doc.get("height") or 0), stream)
    team_colors = assign_teams(frames, video, log=lambda m: protocol.emit(protocol.log(m), stream))
    doc["team_colors"] = team_colors
    tmp = doc_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp.replace(doc_path)
    protocol.emit(protocol.result(str(doc_path)), stream)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="frisbee match analysis worker (JSON-lines on stdout)")
    parser.add_argument("--video", default=None, help="视频路径（--team-only 时可省）")
    parser.add_argument("--output-dir", default=None, help="产物目录（默认 runs/gui_analysis/<视频名>）")
    parser.add_argument("--weights", default="yolo26x.pt", help="检测权重（零样本 COCO person 或 player/referee 微调）")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=None, help="调试用：只处理前 N 帧")
    parser.add_argument("--classes", default="0", help="检测类别过滤（逗号分隔；players_e4 权重用 0,1,2）")
    parser.add_argument("--tile-grid", type=int, default=0,
                        help="切片检测网格（2=2x2，远场召回 +29~33%%；0=关闭）")
    parser.add_argument("--tile-conf", type=float, default=0.3, help="切片检测置信度")
    parser.add_argument("--team-from-cls", action="store_true",
                        help="players_e4 权重：检测类别即队伍（0=红队 1=蓝队 2=裁判），跳过聚类")
    parser.add_argument("--min-y-frac", type=float, default=0.30,
                        help="启发式场内过滤：框底 y2 ≥ H*该值（0.45 会误切远场球员，迭代6 实测）")
    parser.add_argument("--disc-weights", default=None,
                        help="飞盘检测权重（提供则启用双模型联合跟踪 + 事件统计）")
    parser.add_argument("--disc-conf", type=float, default=0.35, help="飞盘检测置信度阈值")
    parser.add_argument("--calibration", default=None,
                        help="场地标定 json：场内过滤升级为多边形过滤；事件统计需要")
    parser.add_argument("--no-field-filter", action="store_true",
                        help="保存未过滤原始检出（自动标定等用途）")
    parser.add_argument("--team-only", metavar="TRACKS_JSON", default=None,
                        help="跳过跟踪，只对已有 tracks.json 重算分队")
    args = parser.parse_args(argv)

    if not args.team_only and not args.video:
        parser.error("--video is required unless --team-only is used")

    try:
        if args.team_only:
            return run_team_only(args)
        args.classes = [int(c) for c in str(args.classes).split(",") if c.strip()]
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
