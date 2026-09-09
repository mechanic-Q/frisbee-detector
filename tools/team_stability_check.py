"""跨素材分队自洽性抽检（不依赖 HSV——用于深色/相似球衣素材）。

原理：对同一素材独立跑两次分队（不同 KMeans 随机状态 + 不同采样间隔），
统计 track 级队伍归属一致率。真两队在稳定特征下应高度自洽（≥0.85）；
伪两簇（按亮度/噪声分簇）在扰动下会翻转 → 一致率骤降 → 判"不可信"。
配合 team_color_check（HSV 可分素材）构成双素材验证矩阵。

用法：
    python3 tools/team_stability_check.py --tracks <tracks.json> --video <video>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer import protocol  # noqa: E402
from frisbee_analyzer.team import (TEAM_SAMPLE_INTERVAL, collect_crops,  # noqa: E402
                                   cluster_team_embeddings, label_from_crops)


def run_once(frames: dict, video: str, embedder, sample_interval: int, seed: int) -> dict[int, int]:
    """带扰动的一次分队：改采样间隔抽 crop（embedding 无随机性，扰动来自样本集变化）。"""
    track_ids, crops, xs = collect_crops(frames, video, sample_interval)
    if not crops:
        return {}
    embeddings = np.asarray(embedder.embed(crops), dtype=np.float32)
    from sklearn.cluster import KMeans
    from frisbee_analyzer.team import team_ids_by_x
    from collections import defaultdict as dd

    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(embeddings)
    labels = km.labels_
    mapping = team_ids_by_x(labels, np.asarray(xs, dtype=np.float64))
    per_track: defaultdict[int, list[int]] = dd(list)
    for tid, lab in zip(track_ids, labels):
        per_track[int(tid)].append(int(lab))
    return {tid: mapping.get(Counter(labs).most_common(1)[0][0])
            for tid, labs in per_track.items() if mapping.get(Counter(labs).most_common(1)[0][0]) is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description="分队自洽性抽检（HSV 无关）")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval-b", type=int, default=TEAM_SAMPLE_INTERVAL // 2)
    args = parser.parse_args()

    doc = json.loads(Path(protocol.win_to_wsl(args.tracks)).read_text(encoding="utf-8"))
    frames = doc["frames"]
    video = protocol.win_to_wsl(args.video)

    from frisbee_analyzer.team import SiglipEmbedder
    embedder = SiglipEmbedder()

    run_a = run_once(frames, video, embedder, TEAM_SAMPLE_INTERVAL, seed=42)
    run_b = run_once(frames, video, embedder, args.interval_b, seed=7)
    common = set(run_a) & set(run_b)
    if not common:
        print(json.dumps({"error": "no common tracks", "a": len(run_a), "b": len(run_b)}))
        return 1
    agree = sum(1 for t in common if run_a[t] == run_b[t])
    report = {
        "tracks_compared": len(common),
        "stability": round(agree / len(common), 4),
        "gate": ">=0.85",
        "pass": bool(agree / len(common) >= 0.85),
        "interval_a": TEAM_SAMPLE_INTERVAL,
        "interval_b": args.interval_b,
    }
    out = Path(protocol.win_to_wsl(args.tracks)).parent / "team_stability_check.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()
