"""验收抽检：team_id 与球衣颜色（HSV 红蓝）一致性核对。

原理：分队模块的 team_id 来自 SigLIP 嵌入聚类（0=半场 x 较小一侧），本脚本独立地
用 HSV 颜色规则给每个已分队裁剪判"红衣/蓝衣"，然后度量两件事：
  1) 簇内纯度 —— 同一 team_id 的裁剪主色占比（抽检准确率）；
  2) 簇间互斥 —— 两个 team_id 的主色必须不同（否则聚类把两队混了）。
在 WSL 运行：
    python3 tools/team_color_check.py \
        --tracks runs/gui_analysis/accept_10min/tracks.json \
        --video 'E:\\...\\accept_10min.mp4'
产出 JSON 报告 + 可读摘要；抽检准确率 ≥0.90 即达验收门（方案 §5 Phase 1 的机器可执行近似）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer import protocol  # noqa: E402
from frisbee_analyzer.team import crop_upper_body  # noqa: E402


def hsv_label(crop) -> tuple[str | None, float, float]:
    """上半身裁剪 → ('red'|'blue'|None, red_frac, blue_frac)。红色 H<8|H>172，蓝色 H∈[95,135]。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)
    n = int(h.size)
    red = ((h < 8) | (h > 172)) & (s >= 70) & (v >= 60)
    blue = (h >= 95) & (h <= 135) & (s >= 70) & (v >= 60)
    rf = float(red.sum()) / n
    bf = float(blue.sum()) / n
    if max(rf, bf) < 0.15 or abs(rf - bf) < 0.03:
        return None, rf, bf
    return ("red" if rf > bf else "blue"), rf, bf


def main() -> int:
    parser = argparse.ArgumentParser(description="team_id × 球衣颜色一致性抽检")
    parser.add_argument("--tracks", required=True, help="tracks.json 路径")
    parser.add_argument("--video", required=True, help="对应视频（Windows/WSL 路径均可）")
    parser.add_argument("--sample-interval", type=int, default=600, help="每 N 帧抽一帧（默认 600=20s）")
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径（默认与 tracks.json 同目录）")
    parser.add_argument("--all-dets", action="store_true",
                        help="不过滤观众（默认只统计场地球员：画面下部 + 框够大，见方案 §8.1）")
    parser.add_argument("--min-height", type=int, default=90)
    parser.add_argument("--min-y-frac", type=float, default=0.45, help="框底边最低 y ≥ H*该值 才算场内")
    args = parser.parse_args()

    doc = json.loads(Path(protocol.win_to_wsl(args.tracks)).read_text(encoding="utf-8"))
    frames = doc["frames"]
    video = protocol.win_to_wsl(args.video)
    out_path = protocol.win_to_wsl(args.output) if args.output \
        else str(Path(args.tracks).parent / "team_color_check.json")

    track_team: dict[int, int] = {}  # track 首次出现时的 team_id
    for dets in frames.values():
        for det in dets:
            if det.get("team_id") in (0, 1) and det["track_id"] not in track_team:
                track_team[det["track_id"]] = det["team_id"]

    keys = [k for k in sorted(frames, key=int) if int(k) % args.sample_interval == 0]
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"cannot open video: {video}")
        return 1

    team_crops: defaultdict[int, Counter] = defaultdict(Counter)   # team → 裁剪级颜色计数
    track_crops: defaultdict[int, Counter] = defaultdict(Counter)  # track → 裁剪级颜色计数
    ambiguous = tiny = 0

    for key in keys:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(key)):
            continue
        ok, frame = cap.read()
        if not ok:
            continue
        fh = frame.shape[0]
        for det in frames[key]:
            team = track_team.get(det["track_id"])
            if team is None:
                continue
            if not args.all_dets:  # 球员过滤：观众席在画面上部且框小（方案 §8.1 场地过滤的近似）
                x1, y1, x2, y2 = det["bbox"]
                if (y2 - y1) < args.min_height or y2 < fh * args.min_y_frac:
                    continue
            crop = crop_upper_body(frame, det["bbox"])
            if crop is None:
                tiny += 1
                continue
            label, _rf, _bf = hsv_label(crop)
            if label is None:
                ambiguous += 1
                continue
            team_crops[team][label] += 1
            track_crops[det["track_id"]][label] += 1
    cap.release()

    decisive = sum(sum(c.values()) for c in team_crops.values())
    team_dominant = {t: c.most_common(1)[0][0] for t, c in team_crops.items() if c}
    crop_agree = sum(c[team_dominant[t]] for t, c in team_crops.items() if t in team_dominant)

    track_total = track_agree = 0
    for tid, c in track_crops.items():
        team = track_team.get(tid)
        if team not in team_dominant:
            continue
        track_total += 1
        if c.most_common(1)[0][0] == team_dominant[team]:
            track_agree += 1

    report = {
        "tracks_json": args.tracks,
        "video": video,
        "sample_interval": args.sample_interval,
        "samples": len(keys),
        "crops_decisive": decisive,
        "crops_ambiguous": ambiguous,
        "crops_too_small": tiny,
        "team_crop_distribution": {str(t): dict(sorted(c.items())) for t, c in sorted(team_crops.items())},
        "team_dominant_color": {str(t): team_dominant.get(t) for t in (0, 1)},
        "cross_team_distinct": (len(set(team_dominant.values())) == 2) if len(team_dominant) == 2 else None,
        "crop_consistency": round(crop_agree / decisive, 4) if decisive else None,
        "track_consistency": round(track_agree / track_total, 4) if track_total else None,
        "gate": ">=0.90",
    }

    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "team_crop_distribution"},
                     ensure_ascii=False, indent=2))
    print("team crop distribution:", json.dumps(report["team_crop_distribution"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
