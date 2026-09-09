"""裁判类过采样数据集：把含 referee(2) 框的训练帧复制 2 份（共 3 倍权重），缓解裁判样本稀少。

从 players_e4_clean 复制到 players_e4_ref：含裁判框的训练帧额外复制 2 次（_dup1/_dup2），
val 不动。YOLO 自带的 mosaic/flip 增强让重复帧在每轮呈现不同观感。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path("/mnt/e/frisbee-detector")
SRC = ROOT / "data/datasets/players_e4_clean"
DST = ROOT / "data/datasets/players_e4_ref"
REF_DUP = 2  # 含裁判帧额外复制份数


def main() -> int:
    if not SRC.is_dir():
        print(f"missing source dataset: {SRC}")
        return 1
    stats = {"copied": 0, "ref_frames_duplicated": 0, "dups": 0}
    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in (SRC / "images" / split).glob("*.jpg"):
            lbl = SRC / "labels" / split / f"{img.stem}.txt"
            shutil.copy2(img, DST / "images" / split / img.name)
            shutil.copy2(lbl, DST / "labels" / split / lbl.name)
            stats["copied"] += 1
            has_ref = split == "train" and lbl.exists() and any(
                line.startswith("2 ") for line in lbl.read_text().splitlines())
            if has_ref:
                stats["ref_frames_duplicated"] += 1
                for k in range(1, REF_DUP + 1):
                    shutil.copy2(img, DST / "images" / split / f"{img.stem}_ref{k}.jpg")
                    shutil.copy2(lbl, DST / "labels" / split / f"{img.stem}_ref{k}.txt")
                    stats["dups"] += 1
    yaml_path = ROOT / "configs" / "players_e4_ref.yaml"
    yaml_path.write_text(
        f"path: {DST}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: player-red\n  1: player-blue\n  2: referee\n  3: spectator\n",
        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    print(f"dataset: {DST}\nyaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    main()
