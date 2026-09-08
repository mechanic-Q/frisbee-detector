"""E6 融合：字幕路 + ASR路(LLM抽取) + 记分牌OCR路 → 三路投票自动 GT 事件表。
在 WSL 运行: python3 tools/fuse_auto_gt.py
输出: results/auto_gt_events.json
"""
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

key = os.environ.get("GLM_API_KEY", "")
client = OpenAI(api_key=key, base_url="https://open.bigmodel.cn/api/paas/v4")

PROMPT = """你是极限飞盘解说转写的事件抽取器。从下列带时间戳的转写句子中抽取事件，输出 JSON 数组：
[{{"t_start": 秒, "type": "score|turnover", "team": "北京大哥|上海沪蛙|未知", "evidence": "原句"}}]
score=得分/拿一分/拿下第X分；turnover=掉盘/失误/被断/攻守易势。没有就输出 []。
转写:
{chunk}"""


def llm_extract(sentences):
    events = []
    WIN, STEP = 60.0, 50.0
    dur = max((s["t_end"] for s in sentences), default=0)
    t = 0.0
    while t < dur:
        chunk = "\n".join(f"[{int(s['t_start']//60):02d}:{int(s['t_start']%60):02d}] {s['text']}"
                          for s in sentences if t - 1 <= s["t_start"] < t + WIN)
        if chunk.strip():
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model="glm-4-flash",
                        messages=[{"role": "user", "content": PROMPT.format(chunk=chunk)}],
                        temperature=0.1)
                    content = re.sub(r"^```(json)?|```$", "", (r.choices[0].message.content or "").strip(), flags=re.M)
                    m = re.search(r"\[.*\]", content, re.S)
                    if m:
                        for e in json.loads(m.group(0)):
                            if isinstance(e, dict) and e.get("type") in ("score", "turnover"):
                                events.append(e)
                    break
                except Exception as e:
                    print("  retry:", e, flush=True)
                    time.sleep(3)
        t += STEP
    return events


def main():
    # 记分牌路（准绳）：比分跳变 → score 事件
    ocr = json.loads((RES / "score_timeline.json").read_text())
    ocr_events = []
    prev_a = prev_b = None
    for i, p in enumerate(ocr[1:], 1):  # 跳过初始 0:0
        a, b = (int(x) for x in re.split(r"[:：]", p["score"]))
        pa, pb = (int(x) for x in re.split(r"[:：]", ocr[i - 1]["score"])) if i >= 1 else (0, 0)
        if a > pa or b > pb:
            team = "北京大哥" if a > pa else "上海沪蛙"
            ocr_events.append({"t_start": p["t_sec"], "type": "score", "team": team,
                               "evidence": f"记分牌 {ocr[i-1]['score']}->{p['score']}"})

    sub = json.loads((RES / "sub_events.json").read_text())
    asr_path = RES / "asr_transcript.json"
    asr_events = []
    if asr_path.exists():
        d = json.loads(asr_path.read_text())
        sents = d.get("sentences") if isinstance(d, dict) else d
        if sents:
            asr_events = llm_extract(sents)

    def cluster(events_all, tol=90.0):
        """按时间聚类同类型事件，team 多数投票"""
        out = []
        for typ in ("score", "turnover"):
            evs = sorted([e for e in events_all if e["type"] == typ], key=lambda e: e.get("t_start", e.get("t_sec", 0)))
            used = [False] * len(evs)
            for i, e in enumerate(evs):
                if used[i]:
                    continue
                group = [e]
                used[i] = True
                te = e.get("t_start", e.get("t_sec", 0))
                for j in range(i + 1, len(evs)):
                    if used[j]:
                        continue
                    tj = evs[j].get("t_start", evs[j].get("t_sec", 0))
                    if abs(tj - te) <= tol:
                        group.append(evs[j])
                        used[j] = True
                sources = {g.get("source", "?") for g in group}
                teams = [g.get("team", "未知") for g in group if g.get("team", "未知") != "未知"]
                team = max(set(teams), key=teams.count) if teams else "未知"
                ev0 = min(group, key=lambda g: g.get("t_start", g.get("t_sec", 0)))
                out.append({"type": typ, "t_sec": round(min(g.get("t_start", g.get("t_sec", 0)) for g in group), 1),
                            "team": team, "votes": len(sources), "sources": sorted(sources),
                            "evidence": (group[0].get("evidence") or "")[:60]})
        return sorted(out, key=lambda x: x["t_sec"])

    pool = [{"source": "subtitle", **e} for e in sub]
    pool += [{"source": "ocr", **e} for e in ocr_events]
    pool += [{"source": "asr", **e} for e in asr_events]
    # 视频时长钳位（去掉 ASR/LLM 幻觉时间戳）
    DUR = 2832.7
    pool = [e for e in pool if 0 <= e.get("t_start", e.get("t_sec", -1)) <= DUR]
    fused = cluster(pool)
    # OCR 连续重复跳变去重（同比分对只保留首次）
    seen_pair = set()
    dedup = []
    for e in fused:
        if e["type"] == "score" and "记分牌" in e["evidence"]:
            pair = e["evidence"]
            if pair in seen_pair:
                continue
            seen_pair.add(pair)
        dedup.append(e)
    fused = dedup

    (RES / "auto_gt_events.json").write_text(json.dumps({
        "note": "votes=支持信源数(字幕/ocr/asr); score 以 OCR 记分牌为准绳",
        "final_score": ocr[-1]["score"] if ocr else None,
        "events": fused,
    }, ensure_ascii=False, indent=1))
    from collections import Counter
    print("fused:", dict(Counter(e["type"] for e in fused)))
    for e in fused:
        print(f"  {int(e['t_sec']//60):02d}:{int(e['t_sec']%60):02d} {e['type']:9s} {e['team']:6s} votes={e['votes']} | {e['evidence'][:40]}")


if __name__ == "__main__":
    main()
