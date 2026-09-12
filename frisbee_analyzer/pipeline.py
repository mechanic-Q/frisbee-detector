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

import numpy as np

from . import protocol
from .tracking import WorkerCancelled, iter_player_tracks, probe_video

PROGRESS_EVERY = 30  # 每 N 帧上报一次进度（GUI 进度条足够平滑）


def _fused_to_disc_doc(fused) -> dict[str, dict]:
    """融合 DiscFrame 序列 → tracks.json 的 disc_frames 条目（仅 tracking/predicting 入档）。"""
    out: dict[str, dict] = {}
    for idx, f in enumerate(fused):
        if f.status in ("tracking", "predicting") and f.bbox is not None:
            entry = {
                "bbox": [round(v, 1) for v in f.bbox],
                "conf": round(f.conf, 4),
                "cx": round(f.cx, 1),
                "cy": round(f.cy, 1),
                "status": f.status,
            }
            if f.d2 is not None:
                entry["d2"] = f.d2
            if f.speed_ms is not None:
                entry["speed_ms"] = f.speed_ms
            if f.world_xy is not None:
                entry["world_xy"] = [round(f.world_xy[0], 2), round(f.world_xy[1], 2)]
            if f.source:
                entry["source"] = f.source
            if f.holder_track is not None:
                entry["holder_track"] = f.holder_track
            if f.gap_misses is not None:
                entry["gap_misses"] = f.gap_misses
            out[str(idx)] = entry
    return out


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
    disc_seq: list[tuple[int, list[dict]]] = []  # (frame_idx, 盘候选列表) 原始序列
    disc_raw: dict[str, list] = {}               # 持久化原始候选（§13.16：全链条 CPU 复算）
    fused_cur = None   # 当前最优融合结果（hand-roi/补全可能覆盖）
    seq_cur = None     # 与 fused_cur 对应的逐帧检测序列
    if getattr(args, "disc_weights", None):
        # 双模型路径：球员 + 飞盘联合跟踪（事件统计的前提，见 docs/2026-09-09-engine-validation.md）
        from .tracking import iter_player_and_disc_tracks

        for idx, p_dets, d_dets in iter_player_and_disc_tracks(
            video,
            player_weights=args.weights,
            disc_weights=args.disc_weights,
            conf=args.conf,
            disc_conf=args.disc_conf,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
            player_classes=tuple(args.classes),
            disc_tile_grid=getattr(args, "disc_tile_grid", 0) or 0,
            disc_tile_trigger=getattr(args, "disc_tile_trigger", None) or "conf<0.5",
            disc_tile_topk=getattr(args, "disc_tile_topk", None) or 2,
        ):
            frames[str(idx)] = p_dets
            disc_seq.append((idx, d_dets))
            if d_dets:
                disc_raw[str(idx)] = d_dets
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
    if disc_raw:
        n_tile = sum(1 for dets in disc_raw.values() for d in dets if d.get("source") == "tile")
        protocol.emit(protocol.log(f"disc: {len(disc_raw)}/{len(frames)} frames with detection "
                                  f"(tile-source dets: {n_tile})"), stream)
    if not disc_seq and (getattr(args, "disc_fusion", False) or getattr(args, "hand_roi", False)):
        protocol.emit(protocol.log("warning: --disc-fusion/--hand-roi need --disc-weights（未提供，跳过盘通道）"), stream)

    # disc_frames 兜底（未开融合时保持旧格式兼容：取每帧最高分候选）
    disc_frames: dict[str, dict] = {}
    if not getattr(args, "disc_fusion", False):
        for k, dets in disc_raw.items():
            top = max(dets, key=lambda d: d.get("conf", 0.0))
            disc_frames[k] = {"bbox": top["bbox"], "conf": top.get("conf", 0.0),
                              "cx": top.get("cx"), "cy": top.get("cy")}

    # F1 多维融合（默认关）：对盘检测序列做帧间关联+马氏门控+速度拒绝，重写 disc_frames。
    if getattr(args, "disc_fusion", False) and disc_seq:
        from .disc_fusion import fuse_disc_detections

        proj = None
        if getattr(args, "calibration", None):
            try:
                from utils.homography import load_calibration, pixel_to_world

                _calib = load_calibration(args.calibration)
                _m = _calib["matrix"]
                proj = lambda cx, cy: pixel_to_world(_m, cx, cy)  # noqa: E731
            except Exception as e:  # noqa: BLE001
                protocol.emit(protocol.log(f"disc-fusion: calibration ignored ({e})"), stream)

        # 还原成逐帧序列（含空帧；候选列表带 source/holder 直通融合层）
        seq: list[list[dict]] = []
        for _idx, d_dets in disc_seq:
            seq.append([{"bbox": list(d["bbox"]),
                         "conf": float(d.get("conf", 0.0)),
                         **({"source": d["source"]} if d.get("source") else {}),
                         **({"holder_track": d["holder_track"]} if d.get("holder_track") is not None else {})}
                        for d in d_dets])
        fused, fstats = fuse_disc_detections(
            seq, fps=info["fps"], world_projector=proj, field_margin_m=5.0 if proj else None,
            width=info["width"], height=info["height"],
        )
        fused_cur, seq_cur = fused, seq
        new_disc = _fused_to_disc_doc(fused)
        protocol.emit(protocol.log(
            f"disc-fusion: tracking={fstats.n_tracking} predicting={fstats.n_predicting} "
            f"gated={fstats.n_gated} rejected_speed={fstats.n_rejected_speed} "
            f"lost={fstats.n_lost} longest_track={fstats.longest_track_frames}f "
            f"({len(disc_raw)} -> {len(new_disc)} frames)"), stream)
        disc_frames = new_disc

        # F1 第二检测通道（--hand-roi，默认关）：持盘贴身遮挡的盘整帧看不见（§13.11 G4
        # 遗留根因）。对融合态非 tracking 帧，取最近持盘人的手部区 crop 放大复检，
        # 检出并入序列后二次融合。健康 tracking 帧不动。
        if getattr(args, "hand_roi", False) and fused:
            from .hand_roi import merge_hand_roi_detections, recheck_video

            players_by_frame = {int(k): v for k, v in frames.items()}
            extra, hr_stats = recheck_video(
                video, fused, players_by_frame, disc_weights=args.disc_weights,
                k=args.hand_roi_k, conf=args.hand_roi_conf, imgsz=args.hand_roi_imgsz,
                max_gap=args.hand_roi_max_gap,
                width=info["width"], height=info["height"],
                log=lambda m: protocol.emit(protocol.log(m), stream))
            seq2, mstats = merge_hand_roi_detections(seq, extra)
            fused2, fstats2 = fuse_disc_detections(
                seq2, fps=info["fps"], world_projector=proj, field_margin_m=5.0 if proj else None,
                width=info["width"], height=info["height"],
            )
            fused_cur, seq_cur = fused2, seq2
            new_disc2 = _fused_to_disc_doc(fused2)
            n_roi_src = sum(1 for v in new_disc2.values() if v.get("source") == "hand_roi")
            protocol.emit(protocol.log(
                f"hand-roi: frames_with_anchor={hr_stats.frames_with_anchor} "
                f"rechecked={hr_stats.frames_rechecked} rois={hr_stats.rois_checked} "
                f"dets={mstats.dets_added} (deduped {mstats.dets_deduped}) -> "
                f"tracking {fstats.n_tracking}->{fstats2.n_tracking}, "
                f"longest {fstats.longest_track_frames}->{fstats2.longest_track_frames}f, "
                f"disc_frames {len(new_disc)}->{len(new_disc2)} (roi-source {n_roi_src})"), stream)
            disc_frames = new_disc2

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

    # F1: 段内自动标定（--auto-calibrate，默认关）——用本次 run 的未过滤脚点云
    # 现场标定，消除"标定文件与素材时段错位"（§13.2/§13.9 三次实证）。在过滤前执行，
    # 标定 json 落输出目录；若 --calibration 同时给出，段内标定优先（时段匹配）并记录覆盖。
    if getattr(args, "auto_calibrate", False):
        from .segment_calib import auto_calibrate_segment_recover

        calib_doc, creport = auto_calibrate_segment_recover(frames, info["width"], info["height"])
        out_dir_ac = Path(args.output_dir) if args.output_dir else Path("runs/gui_analysis") / Path(video).stem
        out_dir_ac.mkdir(parents=True, exist_ok=True)
        if calib_doc is not None:
            calib_path = out_dir_ac / "segment_calib.json"
            calib_path.write_text(json.dumps(calib_doc, ensure_ascii=False, indent=1), encoding="utf-8")
            matrix, polygon = np.asarray(calib_doc["matrix"], dtype=np.float64), FIELD_POLYGON_M
            protocol.emit(protocol.log(
                f"auto-calibrate: PASS inlier={creport['optimized_inlier_ratio']} "
                f"points={creport['points']} "
                f"window={creport.get('recovery_window', 'full')} -> {calib_path.name}"
                + (" (overrides --calibration)" if getattr(args, "calibration", None) else "")), stream)
        else:
            protocol.emit(protocol.log(
                f"auto-calibrate: FAIL ({creport.get('reason')}) — 回退"
                + ("--calibration" if getattr(args, "calibration", None) else "启发式过滤")), stream)

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

    # F1 第三通道（--possession-impute，默认关）：持盘补全。融合 degraded 段若像素
    # 锚点稳定落在唯一球员框内（同 track_id、步行以下速度），合成该球员手部点的
    # 盘观测二次融合——状态机因此能持续看见被身体遮挡的持盘（§13.14）。需要标定
    # （auto-calibrate 产物或 --calibration）做视差校正，无标定则跳过。
    if getattr(args, "possession_impute", False) and fused_cur is not None:
        if matrix is None:
            protocol.emit(protocol.log(
                "possession-impute: 需要标定（--auto-calibrate PASS 或 --calibration），跳过"), stream)
        else:
            from .possession_impute import impute_possession
            from .hand_roi import merge_hand_roi_detections
            from utils.homography import world_to_pixel, pixel_to_world as _p2w

            players_by_frame = {int(k): v for k, v in frames.items()}
            extra, imp_stats = impute_possession(
                fused_cur, players_by_frame,
                world_to_pixel=lambda wx, wy: world_to_pixel(matrix, wx, wy),
                pixel_to_world=lambda px, py: _p2w(matrix, px, py),
                max_gap=args.impute_max_gap, max_streak=args.impute_max_streak,
            )
            seq3, _ = merge_hand_roi_detections(seq_cur, extra)
            fused3, fstats3 = fuse_disc_detections(
                seq3, fps=info["fps"], world_projector=proj, field_margin_m=5.0 if proj else None,
                width=info["width"], height=info["height"],
            )
            before = len(disc_frames)
            disc_frames = _fused_to_disc_doc(fused3)
            n_imp = sum(1 for v in disc_frames.values() if v.get("source") == "possession_imputed")
            protocol.emit(protocol.log(
                f"possession-impute: anchors={imp_stats.anchor_frames} streaks={imp_stats.streaks} "
                f"imputed={imp_stats.frames_imputed}f -> tracking "
                f"{fstats3.n_tracking}, longest {fstats3.longest_track_frames}f, "
                f"disc_frames {before}->{len(disc_frames)} (imputed-source {n_imp})"), stream)

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
        "disc_raw": disc_raw,
    }

    # 事件统计（需飞盘轨迹；无盘则不产生事件——见验证报告）。
    # 标定来源优先级（§13.16 B / R7 修复）：段内自动标定产物（时段匹配）> --calibration。
    if disc_frames:
        calib_for_events = getattr(args, "calibration", None)
        segcal = out_dir / "segment_calib.json"
        if segcal.exists():
            calib_for_events = str(segcal)
        if calib_for_events:
            try:
                from .events_runner import compute_events

                events_doc = compute_events(doc, Path(calib_for_events))
                doc["events"] = events_doc["events"]
                doc["score"] = events_doc["score"]
                doc["direction_flips"] = events_doc.get("direction_flips", [])
                doc["team_left"] = events_doc.get("team_left", 0)
                protocol.emit(protocol.log(
                    f"events: {len(events_doc['events'])} (score={events_doc['score']}, "
                    f"calib={Path(calib_for_events).name})"), stream)
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


def run_events_only(args, stream=None) -> int:
    """只重算事件统计（CPU 秒级，§13.16 B/R7）：复用已有 tracks.json + 标定。

    标定优先级：--events-calibration 显式给出 > tracks.json 同目录 segment_calib.json。
    保留既有 event_review / team_overrides（人工复核成果不被清空，R8 前置）。
    """
    doc_path = Path(args.events_only)
    if not doc_path.exists():
        protocol.emit(protocol.error(f"tracks.json not found: {doc_path}"), stream)
        return 1
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    if not doc.get("disc_frames"):
        protocol.emit(protocol.error("tracks.json has no disc_frames — 事件统计需盘轨迹"), stream)
        return 1
    calib = getattr(args, "events_calibration", None) or doc_path.parent / "segment_calib.json"
    if not Path(calib).exists():
        protocol.emit(protocol.error(f"calibration not found: {calib}"), stream)
        return 1
    protocol.emit(protocol.meta(len(doc.get("frames", {})), doc.get("fps") or 30.0,
                                doc.get("width") or 0, doc.get("height") or 0), stream)
    from .events_runner import compute_events

    events_doc = compute_events(doc, Path(calib))
    doc["events"] = events_doc["events"]
    doc["score"] = events_doc["score"]
    doc["direction_flips"] = events_doc.get("direction_flips", [])
    doc["team_left"] = events_doc.get("team_left", 0)
    protocol.emit(protocol.log(
        f"events: {len(events_doc['events'])} (score={events_doc['score']}, "
        f"flips={len(doc['direction_flips'])}, calib={Path(calib).name})"), stream)
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
    parser.add_argument("--disc-tile-grid", type=int, default=0,
                        help="盘切片补召回网格（2=2×2，Phase A §13.16；0=关闭）")
    parser.add_argument("--disc-tile-trigger", default="conf<0.5",
                        help="切片触发档位：off / only-no-detection / conf<X（默认 conf<0.5）")
    parser.add_argument("--disc-tile-topk", type=int, default=2,
                        help="每帧保留的盘候选上限（IOS-NMM 合并后按 conf 取前 K）")
    parser.add_argument("--disc-fusion", action="store_true",
                        help="F1 多维融合：盘检测帧间关联+马氏门控+速度拒绝（默认关）")
    parser.add_argument("--hand-roi", action="store_true",
                        help="F1 手部 ROI 复检：持盘贴身遮挡的盘二次检测（隐含 --disc-fusion，默认关）")
    parser.add_argument("--hand-roi-conf", type=float, default=None,
                        help="ROI 复检测出阈值（默认 0.30）")
    parser.add_argument("--hand-roi-imgsz", type=int, default=None,
                        help="ROI crop 送检分辨率（默认 640）")
    parser.add_argument("--hand-roi-max-gap", type=int, default=None,
                        help="距最后观测 ≤ 此帧数才复检（默认 120）")
    parser.add_argument("--hand-roi-k", type=int, default=None,
                        help="每帧最多复检的最近球员数（默认 2）")
    parser.add_argument("--possession-impute", action="store_true",
                        help="F1 持盘补全：融合 degraded 段锚点落在唯一球员框内时合成手部点盘观测"
                             "（需标定，默认关）")
    parser.add_argument("--impute-max-gap", type=int, default=None,
                        help="距最后观测 ≤ 此帧数才补全（默认 150）")
    parser.add_argument("--impute-max-streak", type=int, default=None,
                        help="单段最多连续补全帧数（默认 90）")
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="F1 段内自动标定：用本次 run 的球员脚点云现场标定（默认关；"
                             "给出时优先于 --calibration，过 0.85 内点率门才生效）")
    parser.add_argument("--calibration", default=None,
                        help="场地标定 json：场内过滤升级为多边形过滤；事件统计需要")
    parser.add_argument("--no-field-filter", action="store_true",
                        help="保存未过滤原始检出（自动标定等用途）")
    parser.add_argument("--team-only", metavar="TRACKS_JSON", default=None,
                        help="跳过跟踪，只对已有 tracks.json 重算分队")
    parser.add_argument("--events-only", metavar="TRACKS_JSON", default=None,
                        help="跳过跟踪，只重算事件统计（CPU 秒级；标定取同目录 "
                             "segment_calib.json 或 --events-calibration）")
    parser.add_argument("--events-calibration", default=None,
                        help="--events-only 的显式标定路径（默认同目录 segment_calib.json）")
    args = parser.parse_args(argv)

    if not args.team_only and not args.video and not args.events_only:
        parser.error("--video is required unless --team-only/--events-only is used")

    if args.hand_roi:
        args.disc_fusion = True  # 手部 ROI 定义在融合态之上，隐含开启
    if any(getattr(args, a) is None for a in
           ("hand_roi_conf", "hand_roi_imgsz", "hand_roi_max_gap", "hand_roi_k",
            "impute_max_gap", "impute_max_streak")):
        from . import hand_roi as _hr
        from . import possession_impute as _pi

        args.hand_roi_conf = args.hand_roi_conf if args.hand_roi_conf is not None else _hr.HAND_ROI_CONF
        args.hand_roi_imgsz = args.hand_roi_imgsz if args.hand_roi_imgsz is not None else _hr.HAND_ROI_IMGSZ
        args.hand_roi_max_gap = args.hand_roi_max_gap if args.hand_roi_max_gap is not None else _hr.HAND_ROI_MAX_GAP
        args.hand_roi_k = args.hand_roi_k if args.hand_roi_k is not None else _hr.HAND_ROI_K
        args.impute_max_gap = args.impute_max_gap if args.impute_max_gap is not None else _pi.IMPUTE_MAX_GAP
        args.impute_max_streak = args.impute_max_streak if args.impute_max_streak is not None else _pi.IMPUTE_STREAK

    try:
        if args.team_only:
            return run_team_only(args)
        if args.events_only:
            return run_events_only(args)
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
