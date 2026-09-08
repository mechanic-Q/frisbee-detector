"""把 E4' 自动角色标注转成 players_e4 三分类训练集（零人工）。

输入（主 checkout）：data/bili_final_test/{player_frames,role_labels}
类别映射：0=player-red 1=player-blue 2=referee 保留；3=spectator/ignore 丢弃（变背景）。
切分：按帧名时间序，后 10% 整块作 val（2s 间隔近重复帧，随机切分会泄漏）。
输出：data/datasets/players_e4/{images,labels}/{train,val} + configs/players_e4.yaml

在 WSL 运行：python3 tools/prep_players_dataset.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path("/mnt/e/frisbee-detector")
FRAMES = ROOT / "data/bili_final_test/player_frames"
LABELS = ROOT / "data/bili_final_test/role_labels"
OUT = ROOT / "data/datasets/players_e4"
KEEP_CLASSES = {"0", "1", "2"}  # 3=spectator/ignore 丢弃
VAL_FRACTION = 0.10


def main() -> int:
    if not FRAMES.is_dir() or not LABELS.is_dir():
        print(f"missing input: {FRAMES} / {LABELS}")
        return 1

    stems = sorted(p.stem for p in FRAMES.glob("*.jpg"))
    if not stems:
        print("no frames found")
        return 1
    n_val = max(1, int(len(stems) * VAL_FRACTION))
    val_stems = set(stems[-n_val:])  # 时间序最后 10%，防时间近邻泄漏

    stats = {"train": 0, "val": 0, "boxes_kept": 0, "boxes_dropped": 0}
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    for stem in stems:
        split = "val" if stem in val_stems else "train"
        src_img = FRAMES / f"{stem}.jpg"
        src_lbl = LABELS / f"{stem}.txt"
        shutil.copy2(src_img, OUT / "images" / split / src_img.name)
        kept = []
        if src_lbl.exists():
            for line in src_lbl.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                if parts[0] not in KEEP_CLASSES:
                    stats["boxes_dropped"] += 1
                    continue
                kept.append(" ".join(parts))
                stats["boxes_kept"] += 1
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        stats[split] += 1

    yaml_path = ROOT / "configs" / "players_e4.yaml"
    yaml_path.write_text(
        f"path: {OUT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: player-red\n"
        "  1: player-blue\n"
        "  2: referee\n",
        encoding="utf-8")

    print(json_dumps(stats, val_stems))
    print(f"dataset: {OUT}")
    print(f"yaml:    {yaml_path}")
    return 0


def json_dumps(stats, val_stems) -> str:
    import json
    return json.dumps({"stats": stats, "val_first": sorted(val_stems)[:1],
                       "val_last": sorted(val_stems)[-1:]}, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
