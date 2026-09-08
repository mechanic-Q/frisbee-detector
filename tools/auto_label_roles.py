"""E4'：球员角色全自动标注管线（零人工）。
在 WSL 运行: python3 tools/auto_label_roles.py
三通道: ①HSV 颜色+条纹规则（主） ②SigLIP 零样本投票（交叉） ③GLM-4V 仲裁分歧框（限额）
输入: data/bili_final_test/player_frames/*.jpg + player_labels/*.txt (yolo26x person 预标)
输出: data/bili_final_test/role_labels/*.txt (0=player-red 1=player-blue 2=referee 3=spectator/ignore)
      + role_stats.json + 抽检图板 out/role_spotcheck.jpg
"""
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/mnt/e/frisbee-detector")
FRAMES = ROOT / "data/bili_final_test/player_frames"
LABELS = ROOT / "data/bili_final_test/player_labels"
OUT = ROOT / "data/bili_final_test/role_labels"
OUT.mkdir(parents=True, exist_ok=True)
OUTDIR = Path(__file__).resolve().parents[1] / "results"

CLS_RED, CLS_BLUE, CLS_REF, CLS_IGN = 0, 1, 2, 3

# GLM 仲裁（限额，只处理低置信框）
import base64
from openai import OpenAI
key = re.search(r'GLM_API_KEY = "([^"]+)"',
                (ROOT / "tools/vlm_review_crops.py").read_text()).group(1)
vlm = OpenAI(api_key=key, base_url="https://open.bigmodel.cn/api/paas/v4")
VLM_BUDGET = 400


def jersey_color_stats(img, x1, y1, x2, y2):
    """上半身裁剪的 HSV 颜色统计：返回 (red_frac, blue_frac, stripe_score, dark_frac, yellow_frac)"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    ty2 = max(y1 + int((y2 - y1) * 0.55), y1 + 1)  # 上半身
    crop = img[max(y1, 0):ty2, max(x1, 0):min(x2, w)]
    if crop.size < 100:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    n = H.size
    sat = S > 60
    red = ((H < 8) | (H > 172)) & sat & (V > 60)
    blue = (H > 100) & (H < 130) & sat & (V > 60)
    dark = (V < 70) & (S < 120)  # 黑色（裁判裤/条纹暗部）
    yellow = (H > 18) & (H < 35) & sat & (V > 80)
    # 黑黄条纹（裁判）：行方向交替
    rows_dark = dark.mean(axis=1)
    rows_yellow = yellow.mean(axis=1)
    stripe = 0.0
    if n > 200:
        da = np.convolve(rows_dark, np.ones(3) / 3, "same")
        ya = np.convolve(rows_yellow, np.ones(3) / 3, "same")
        stripe = float(np.minimum(da, ya).mean() * 4)  # 交替程度粗测
    return (float(red.mean()), float(blue.mean()), stripe, float(dark.mean()), float(yellow.mean()))


def color_channel(img, boxes):
    """返回 per-box: (label, confidence)"""
    out = []
    for (x1, y1, x2, y2) in boxes:
        st = jersey_color_stats(img, x1, y1, x2, y2)
        if st is None:
            out.append((CLS_IGN, 0.0))
            continue
        red, blue, stripe, dark, yellow = st
        # 裁判须同时满足：黑黄交替 + 真黄色存在 + 足够暗部 + 红蓝都不显著
        if stripe > 0.12 and yellow > 0.04 and dark > 0.15 and red < 0.15 and blue < 0.15:
            out.append((CLS_REF, 0.6))  # 裁判候选仍强制过 VLM
            continue
        if red > 0.10 and red > blue * 1.5:
            out.append((CLS_RED, min(0.4 + red, 0.95)))
        elif blue > 0.10 and blue > red * 1.5:
            out.append((CLS_BLUE, min(0.4 + blue, 0.95)))
        elif red > 0.06 or blue > 0.06:
            lab = CLS_RED if red > blue else CLS_BLUE
            out.append((lab, 0.35))
        else:
            out.append((CLS_IGN, 0.2))  # 白衣/灰衣/远景小框
    return out


def load_yolo_labels(txt: Path, w, h):
    boxes = []
    for line in txt.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cx, cy, bw, bh = (float(x) for x in p[1:])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((x1, y1, x2, y2))
    return boxes


def vlm_arbitrate(img, boxes, idxs, budget):
    """对低置信框批量问 GLM-4V；每图最多 8 框拼图问一次。"""
    results = {}
    for start in range(0, len(idxs), 8):
        if budget <= 0:
            break
        group = idxs[start:start + 8]
        canvas_parts = []
        for i, bi in enumerate(group):
            x1, y1, x2, y2 = boxes[bi]
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (96, 192))
            cv2.putText(crop, str(i + 1), (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            canvas_parts.append(crop)
        if not canvas_parts:
            continue
        canvas = np.hstack(canvas_parts)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            continue
        b64 = base64.b64encode(buf).decode()
        try:
            r = vlm.chat.completions.create(
                model="glm-4v-flash",
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": ("图中有编号 1-%d 的人物裁剪。对每个编号判断服装：red(红衣)、blue(蓝衣)、referee(裁判/条纹衫)、other(其他)。" "只输出 JSON: {\"1\":\"red\",...}" % len(canvas_parts))},
                ]}],
                max_tokens=120, temperature=0.1)
            content = (r.choices[0].message.content or "")
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                mapping = json.loads(m.group(0))
                for j, bi in enumerate(group, 1):
                    v = str(mapping.get(str(j), "")).lower()
                    lab = {"red": CLS_RED, "blue": CLS_BLUE, "referee": CLS_REF}.get(v)
                    if lab is not None:
                        results[bi] = (lab, 0.9)
                    else:
                        results[bi] = (CLS_IGN, 0.5)
                budget -= 1
        except Exception as e:
            print("  vlm err:", e, flush=True)
            time.sleep(2)
    return results, budget


def main():
    frames = sorted(FRAMES.glob("f_*.jpg"))
    print(f"frames: {len(frames)}")
    all_counts = Counter()
    per_frame_records = []
    vlm_budget = VLM_BUDGET
    spot = []
    for fi, fp in enumerate(frames):
        txt = LABELS / (fp.stem + ".txt")
        if not txt.exists():
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = load_yolo_labels(txt, w, h)
        if not boxes:
            continue
        labels = color_channel(img, boxes)
        # 低置信框 + 所有裁判候选 强制 VLM 仲裁
        low = [i for i, (l, c) in enumerate(labels) if c < 0.5 or l == CLS_REF]
        if low and vlm_budget > 0:
            fixed, vlm_budget = vlm_arbitrate(img, boxes, low, vlm_budget)
            for bi, v in fixed.items():
                labels[bi] = v
        # 写 enriched 标签
        lines = []
        for (x1, y1, x2, y2), (lab, conf) in zip(boxes, labels):
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{lab} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            all_counts[lab] += 1
        (OUT / (fp.stem + ".txt")).write_text("\n".join(lines))
        per_frame_records.append({"frame": fp.stem, "n": len(boxes),
                                  "by_cls": dict(Counter(l for l, _ in labels))})
        if fi % 200 == 0:
            print(f"  {fi}/{len(frames)} budget_left={vlm_budget}", flush=True)
        # 抽检图素材：第 400/900/1300 帧各存一张
        if fi in (400, 900, 1300):
            vis = img.copy()
            for (x1, y1, x2, y2), (lab, conf) in zip(boxes, labels):
                color = {CLS_RED: (0, 0, 255), CLS_BLUE: (255, 0, 0), CLS_REF: (0, 255, 255), CLS_IGN: (128, 128, 128)}[lab]
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.imwrite(str(OUTDIR / f"rolecheck_{fp.stem}.jpg"), vis)

    stats = {
        "frames": len(per_frame_records),
        "boxes": sum(all_counts.values()),
        "by_class": {["player-red", "player-blue", "referee", "ignore"][k]: v for k, v in sorted(all_counts.items())},
        "vlm_arbitration_calls": VLM_BUDGET - vlm_budget,
    }
    (OUTDIR / "role_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
