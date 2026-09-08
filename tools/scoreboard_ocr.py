"""E6 路B：记分牌 OCR 旁证 — 顶部固定 ROI + 帧差触发 + GLM-4V 读比分。
在 WSL 运行: python3 tools/scoreboard_ocr.py
输出: data/bili_final_test/out/score_timeline.json（每次比分变化时间点）
"""
import base64
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

ROOT = Path("/mnt/e/frisbee-detector")
VID = ROOT / "movie/2024城市飞盘俱乐部锦标赛『决赛』【北京大哥 VS 上海沪蛙】高清版3-1.mp4"
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)

key = os.environ.get("GLM_API_KEY", "")
client = OpenAI(api_key=key, base_url="https://open.bigmodel.cn/api/paas/v4")

# 顶部记分牌 ROI（1080P，从探针帧看在顶部中央偏左，先取宽条再由帧差定位）
ROI_Y1, ROI_Y2 = 0, 120


def ask_score(img_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return "ERR"
    b64 = base64.b64encode(buf).decode()
    try:
        r = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "读取画面记分牌上的比分。只输出格式 'A:B' 的两个数字（如 3:2）。如果看不到记分牌或读不清，输出 NONE"},
            ]}],
            max_tokens=10, temperature=0.1)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print("  vlm err:", e)
        return "ERR"


def main():
    cap = cv2.VideoCapture(str(VID))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * 2)  # 0.5fps 采样
    print(f"fps={fps}, total={total}, sampling every {step} frames (~0.5fps)")

    timeline = []          # (sec, 'A:B')
    prev_roi = None
    idx = 0
    reads = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or idx > total:
            break
        sec = idx / fps
        roi = frame[ROI_Y1:ROI_Y2]
        small = cv2.resize(roi, (480, 54))
        if prev_roi is not None:
            diff = float(np.abs(small.astype(np.int16) - prev_roi.astype(np.int16)).mean())
            # 帧差触发：记分牌区域变化（得分/镜头切换）才调 VLM
            if diff > 3.0 and (not timeline or sec - timeline[-1][0] > 3):
                s = ask_score(roi)
                reads += 1
                if re.fullmatch(r"\d+\s*[:：]\s*\d+", s):
                    a, b = re.split(r"[:：]", s)
                    cur = f"{int(a)}:{int(b)}"
                    if not timeline or timeline[-1][1] != cur:
                        timeline.append((sec, cur))
                        print(f"  [{sec:7.1f}s] score {cur} (diff={diff:.1f})", flush=True)
                elif s == "NONE" and timeline and sec - timeline[-1][0] > 60:
                    pass  # 镜头离开记分牌，保持最后比分
        prev_roi = small
        idx += step
    cap.release()

    data = [{"t_sec": round(s, 1), "score": sc} for s, sc in timeline]
    (OUT / "score_timeline.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"vlm reads: {reads}, score changes: {len(data)}")
    print("saved -> out/score_timeline.json")


if __name__ == "__main__":
    main()
