"""迭代3：用已验收的 HSV 聚类队伍归属反向清洗 E4' 标签，生成 players_e4_clean 四类数据集。

帧对齐（已用感知哈希 5/5 点验证）：player_frames 的 f_N = accept_10min 视频第 (N-151)*60 帧
（2 秒间隔抽帧，f_151 ↔ 第 0 帧）。f_151..f_451 共 301 帧落在 accept_10min 内可清洗；
其余帧保持原 E4' 标签不动（无法交叉验证，但也不破坏——spectator 降级策略照旧）。

清洗逻辑（对可对齐帧的每个 E4' 标签框）：
  1. 与 tracks.json（HSV 聚类路径，一致性 97.8%）做 IoU 匹配（≥0.5）；
  2. 匹配到且检测 cls(0红/1蓝) 与标签 cls 一致 → 保留原类；
  3. 匹配到但类别冲突，或未匹配（观众/杂框）→ 降级 spectator(3)（显式类替代背景抑制）；
  4. 裁判(2) 仅在与 track 裁判一致时保留。
不可对齐帧：原 E4' 类 0/1/2 保留、3(spectator/ignore) 仍降级为显式 3。

输出：data/datasets/players_e4_clean/{images,labels}/{train,val} + configs/players_e4_clean.yaml
类别：0=player-red 1=player-blue 2=referee 3=spectator

在 WSL 运行：
    python3 tools/clean_players_labels.py --tracks runs/gui_analysis/accept_10min/tracks.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer import protocol  # noqa: E402

ROOT = Path("/mnt/e/frisbee-detector")
FRAMES = ROOT / "data/bili_final_test/player_frames"
LABELS = ROOT / "data/bili_final_test/role_labels"
OUT = ROOT / "data/datasets/players_e4_clean"
VAL_FRACTION = 0.10
IOU_MATCH = 0.5
FRAME_BASE = 151      # f_151 ↔ tracks.json 第 0 帧（哈希 5/5 验证）
FRAME_STRIDE = 60     # 2s × 30fps


def stem_to_video_frame(stem: str) -> int | None:
    """f_000301 → (301-151)*60 = 9000。返回 tracks.json 的 frame 键（字符串）。"""
    try:
        n = int(stem.split("_")[1])
    except (IndexError, ValueError):
        return None
    return (n - FRAME_BASE) * FRAME_STRIDE


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="HSV 聚类结果反向清洗 E4' 标签")
    parser.add_argument("--tracks", required=True, help="HSV 聚类路径的 tracks.json（97.8% 一致性产物）")
    parser.add_argument("--iou", type=float, default=IOU_MATCH)
    args = parser.parse_args()

    doc = json.loads(Path(protocol.win_to_wsl(args.tracks)).read_text(encoding="utf-8"))
    H = doc.get("height") or 1080
    tracks_by_frame: dict[int, list[tuple[float, float, float, float, int | None]]] = {}
    for key, dets in doc.get("frames", {}).items():
        tracks_by_frame[int(key)] = [
            (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d.get("team_id"))
            for d in dets
        ]

    stems = sorted(p.stem for p in FRAMES.glob("*.jpg"))
    n_val = max(1, int(len(stems) * VAL_FRACTION))
    val_stems = set(stems[-n_val:])
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"kept": 0, "demoted_spectator": 0, "dropped": 0,
             "aligned": 0, "unaligned": 0, "train": 0, "val": 0}
    for stem in stems:
        split = "val" if stem in val_stems else "train"
        src_img = FRAMES / f"{stem}.jpg"
        src_lbl = LABELS / f"{stem}.txt"
        if not src_img.exists():
            continue
        vf = stem_to_video_frame(stem)
        candidates = tracks_by_frame.get(vf, []) if vf is not None else []
        if candidates:
            stats["aligned"] += 1
        else:
            stats["unaligned"] += 1

        lines = []
        if src_lbl.exists():
            for line in src_lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                cx, cy, w, h = (float(v) for v in parts[1:5])
                if candidates:
                    x1, y1 = (cx - w / 2) * 1920, (cy - h / 2) * H
                    x2, y2 = (cx + w / 2) * 1920, (cy + h / 2) * H
                    best_iou, best_team = 0.0, None
                    for tx1, ty1, tx2, ty2, team in candidates:
                        v = iou((x1, y1, x2, y2), (tx1, ty1, tx2, ty2))
                        if v > best_iou:
                            best_iou, best_team = v, team
                    if best_iou < args.iou:
                        new_cls = "3"                  # 未匹配：观众/杂框
                        stats["demoted_spectator"] += 1
                    elif cls == "2":
                        if best_team == 2:
                            new_cls = "2"
                            stats["kept"] += 1
                        else:
                            new_cls = "3"
                            stats["demoted_spectator"] += 1
                    elif cls in ("0", "1") and best_team in (0, 1) and int(cls) == best_team:
                        new_cls = cls                  # 双通道一致 → 高置信
                        stats["kept"] += 1
                    elif cls in ("0", "1"):
                        new_cls = "3"                  # 队伍冲突 → 噪声
                        stats["demoted_spectator"] += 1
                    else:
                        stats["dropped"] += 1
                        continue
                else:
                    # 不可对齐帧：原 0/1/2 保留，3 → 显式 spectator
                    new_cls = cls if cls in ("0", "1", "2") else "3"
                    if new_cls == "3":
                        stats["demoted_spectator"] += 1
                    else:
                        stats["kept"] += 1
                lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        shutil.copy2(src_img, OUT / "images" / split / src_img.name)
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[split] += 1

    yaml_path = ROOT / "configs" / "players_e4_clean.yaml"
    yaml_path.write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: player-red\n  1: player-blue\n  2: referee\n  3: spectator\n",
        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    print(f"dataset: {OUT}\nyaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
