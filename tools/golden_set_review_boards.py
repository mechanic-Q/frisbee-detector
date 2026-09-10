"""Step 3b: 生成人工复核图板（替代不可用的 VLM 通道）。
策略：按"多模型共识度"分层，让复核者只在高信息量的帧上判断。
输出:
  data/golden_set_100/review/consensus_*.jpg   每帧所有候选框（按共识度着色）
  data/golden_set_100/review/INDEX.md          复核清单与判读说明
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("FRISBEE_ROOT", str(Path(__file__).resolve().parents[1])))
GOLD = ROOT / "data/golden_set_100"
FRAMES = GOLD / "frames"
REVIEW = GOLD / "review"
REVIEW.mkdir(parents=True, exist_ok=True)

# 无监督"疑似真盘"打分规则（不依赖外部 VLM）：
#  - 多模型共识度高（n_models）→ 更像真盘
#  - 框尺寸在飞盘合理范围（原图 1080P 下 8-60px）→ 更像真盘
#  - 位置在场地平面内（单应性投影）→ 更像真盘
MIN_PX, MAX_PX = 6, 80


def load_homography():
    """读项目标定（若有）用于场地内判定。"""
    cal_dir = ROOT / "configs/homography"
    for f in sorted(cal_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            raw = d.get("homography") or d.get("matrix")
            if raw:
                return np.array(raw, dtype=np.float64).reshape(3, 3), d
        except Exception:
            continue
    return None, None


def in_field(H, x, y, margin=5.0):
    if H is None:
        return None   # 无标定 → 未知
    p = H @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return False
    wx, wy = p[0] / p[2], p[1] / p[2]
    return (-margin <= wx <= 105) and (-margin <= wy <= 42)


def color_of(n_models, size_ok, field_ok):
    """BGR 着色：共识高+尺寸合理+场内 = 绿（疑似真）；单模型 = 灰（疑似误检）"""
    if n_models >= 5 and size_ok and field_ok is not False:
        return (0, 220, 0)
    if n_models >= 3 and size_ok:
        return (0, 200, 255)
    if n_models == 2:
        return (255, 160, 0)
    return (140, 140, 140)


def main():
    cands = json.loads((GOLD / "candidates.json").read_text())
    H, cal_meta = load_homography()
    print(f"homography: {'loaded' if H is not None else 'NONE (场地判定跳过)'}")

    index_lines = ["# 金标准复核清单（100 帧）", "",
                   "## 着色说明", "",
                   "- 🟢 绿：≥5 模型共识 + 尺寸合理 → 最可能是真飞盘",
                   "- 🟡 黄：≥3 模型共识 → 较可能",
                   "- 🟠 橙：2 模型共识 → 存疑",
                   "- ⚪ 灰：单模型 → 多为误检", "",
                   "## 判读要点", "",
                   "飞盘 = 白色小圆盘（1080P 下通常 10-50px），多出现在球员手边/空中/草地上。",
                   "草纹、观众白衣、场地标记、水瓶等常被误检。", "",
                   "---", ""]

    tier_stats = {"green": 0, "yellow": 0, "orange": 0, "gray": 0}
    frames_summary = []

    for fname in sorted(cands.keys()):
        img_path = FRAMES / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = cands[fname]
        n_green = n_yellow = n_orange = n_gray = 0
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            size = max(x2 - x1, y2 - y1)
            size_ok = MIN_PX <= size <= MAX_PX
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            fok = in_field(H, cx, cy)
            col = color_of(b["n_models"], size_ok, fok)
            if col == (0, 220, 0):
                n_green += 1
            elif col == (0, 200, 255):
                n_yellow += 1
            elif col == (255, 160, 0):
                n_orange += 1
            else:
                n_gray += 1
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            label = f"{b['n_models']}m"
            cv2.putText(img, label, (int(x1), max(int(y1) - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        tier_stats["green"] += n_green
        tier_stats["yellow"] += n_yellow
        tier_stats["orange"] += n_orange
        tier_stats["gray"] += n_gray
        # 缩略图便于快速浏览（原图太大）
        h, w = img.shape[:2]
        small = cv2.resize(img, (1280, int(1280 * h / w)))
        cv2.imwrite(str(REVIEW / fname), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        frames_summary.append({
            "frame": fname, "total": len(boxes),
            "green": n_green, "yellow": n_yellow, "orange": n_orange, "gray": n_gray,
        })

    # 拼接总览图（20 帧一板）
    for i in range(0, len(frames_summary), 20):
        group = frames_summary[i:i + 20]
        rows = []
        for j in range(0, len(group), 4):
            row_imgs = []
            for item in group[j:j + 4]:
                im = cv2.imread(str(REVIEW / item["frame"]))
                if im is None:
                    continue
                im = cv2.resize(im, (480, 270))
                cv2.putText(im, f"{item['frame']} G{item['green']}Y{item['yellow']}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                row_imgs.append(im)
            while len(row_imgs) < 4:
                row_imgs.append(np.zeros((270, 480, 3), np.uint8))
            rows.append(np.hstack(row_imgs))
        grid = np.vstack(rows)
        cv2.imwrite(str(REVIEW / f"board_{i//20:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        index_lines.append(f"## board_{i//20:02d}.jpg  （帧 {group[0]['frame']} ~ {group[-1]['frame']}）")
        for item in group:
            index_lines.append(f"- {item['frame']}: 总 {item['total']} 框 "
                               f"(绿{item['green']} 黄{item['yellow']} 橙{item['orange']} 灰{item['gray']})")
        index_lines.append("")

    (REVIEW / "INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")
    summary = {"frames": len(frames_summary), "tiers": tier_stats,
               "homography_used": H is not None,
               "note": "VLM 通道不可用（API key 已按安全要求清理）；改用共识度分层 + 人工目检"}
    (REVIEW / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({"frames": len(frames_summary), "tiers": tier_stats}, ensure_ascii=False))
    print(f"boards: {(len(frames_summary)+19)//20} -> {REVIEW}")


if __name__ == "__main__":
    main()
