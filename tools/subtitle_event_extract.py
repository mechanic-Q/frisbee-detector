"""E6 路A：AI字幕 → LLM 事件抽取（得分手/攻守转换候选）。
运行: python tools/subtitle_event_extract.py
输出: data/bili_final_test/out/sub_events.json + sub_events_readable.txt
"""
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SRT = ROOT / "data/bili_final_test/subs/BV1hTtpeyEzb.ai-zh.srt"
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)

key = os.environ.get("GLM_API_KEY", "")
client = OpenAI(api_key=key, base_url="https://open.bigmodel.cn/api/paas/v4")

PROMPT = """你是极限飞盘比赛解说文本的事件抽取器。下面是一段带时间戳的比赛字幕（可能是观众闲聊+解说的混合，ASR 有错字）。
请抽取以下类型的事件，输出 JSON 数组（没有则输出 []）：
[{{"t_start": 秒, "t_end": 秒, "type": "score|turnover|pull", "team": "北京大哥|上海沪蛙|未知", "evidence": "原句"}}]
- score: 明确提到得分/进了一分/拿下第X分（注意"第X分"计数）
- turnover: 明确提到掉盘/失误/被断/攻守转换
- pull: 开盘/发盘
不要过度推断，只抽有明确语句支撑的。team 判断不了就填"未知"。

字幕:
{chunk}"""


def parse_srt(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    items = []
    for b in blocks:
        lines = b.strip().splitlines()
        if len(lines) < 2:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+) --> (\d+):(\d+):(\d+)[,.](\d+)", lines[1])
        if not m:
            continue
        g = list(map(int, m.groups()))
        t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        content = " ".join(lines[2:]).strip()
        if content:
            items.append((t0, t1, content))
    return items


def fmt_ts(sec: float) -> str:
    return f"{int(sec//60):02d}:{int(sec%60):02d}"


def main():
    items = parse_srt(SRT)
    print(f"subtitle cues: {len(items)}")
    # 60s 窗口，含 10s 重叠
    events = []
    WIN, STEP = 60.0, 50.0
    t = 0.0
    dur = items[-1][1]
    n_req = 0
    while t < dur:
        chunk = "\n".join(f"[{fmt_ts(a)}-{fmt_ts(b)}] {c}" for a, b, c in items if a >= t - 1 and a < t + WIN)
        if chunk.strip():
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model="glm-4-flash",
                        messages=[{"role": "user", "content": PROMPT.format(chunk=chunk)}],
                        temperature=0.1)
                    content = (r.choices[0].message.content or "").strip()
                    content = re.sub(r"^```(json)?|```$", "", content, flags=re.M).strip()
                    m = re.search(r"\[.*\]", content, re.S)
                    if m:
                        arr = json.loads(m.group(0))
                        events.extend(arr)
                    break
                except Exception as e:
                    print(f"  retry {attempt}: {e}", flush=True)
                    time.sleep(3 * (attempt + 1))
            n_req += 1
            if n_req % 5 == 0:
                print(f"  window {fmt_ts(t)}/{fmt_ts(dur)}, events so far {len(events)}", flush=True)
        t += STEP

    # 去重（同类型同 5s 内）；容错处理 LLM 返回嵌套数组的情况
    flat = []
    for e in events:
        if isinstance(e, list):
            flat.extend(x for x in e if isinstance(x, dict))
        elif isinstance(e, dict):
            flat.append(e)
    dedup = []
    for e in sorted(flat, key=lambda x: x.get("t_start", 0) or 0):
        if dedup and e["type"] == dedup[-1]["type"] and abs(e["t_start"] - dedup[-1]["t_start"]) < 5:
            continue
        dedup.append(e)

    (OUT / "sub_events.json").write_text(json.dumps(dedup, ensure_ascii=False, indent=2))
    lines = [f'{fmt_ts(e["t_start"])} {e["type"]:9s} {e.get("team","?"):6s} | {e.get("evidence","")[:50]}' for e in dedup]
    (OUT / "sub_events_readable.txt").write_text("\n".join(lines))
    scores = [e for e in dedup if e["type"] == "score"]
    print(f"total events: {len(dedup)} (score={len(scores)}, turnover={len([e for e in dedup if e['type']=='turnover'])})")
    print("saved -> out/sub_events.json / sub_events_readable.txt")


if __name__ == "__main__":
    main()
