"""E6 路C：FunASR 音频转写 — paraformer-zh 句级时间戳，为 LLM 事件抽取提供 ASR 对照文本。
Windows 原生运行（GPU 排队）: python tools/gpu_run.py funasr-asr python tools/asr_transcribe.py
输出: results/asr_transcript.json = {"audio": ..., "duration_sec": ..., "sentences": [{t_start,t_end,text}]}
依赖: ffmpeg 预先提取 results/audio.mp3（16kHz 单声道）；模型首次运行自动从 modelscope 下载（~1GB）。
"""
import json
import sys
import time
from pathlib import Path

AUDIO = Path(__file__).resolve().parents[1] / "results/audio.mp3"
OUT = Path(__file__).resolve().parents[1] / "results/asr_transcript.json"


def main():
    if not AUDIO.exists():
        print(f"missing audio: {AUDIO}", file=sys.stderr)
        sys.exit(2)

    from funasr import AutoModel

    t0 = time.time()
    # 长音频（47min）必须配 VAD 分段 + 标点切句；sentence_timestamp 依赖标点对齐句边界
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 60000},
        punc_model="ct-punc",
    )
    print(f"models ready in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    res = model.generate(
        input=str(AUDIO),
        batch_size_s=300,
        hotword="北京大哥 上海沪蛙 掉盘 攻守转换 开盘",
        sentence_timestamp=True,
    )
    print(f"inference done in {time.time()-t1:.0f}s", flush=True)

    item = res[0] if res else {}
    sents = []
    for s in item.get("sentence_info", []):
        text = (s.get("text") or "").strip()
        if text:
            sents.append({
                "t_start": round(s["start"] / 1000.0, 2),
                "t_end": round(s["end"] / 1000.0, 2),
                "text": text,
            })

    data = {
        "audio": str(AUDIO),
        "model": "paraformer-zh+fsmn-vad+ct-punc",
        "duration_sec": sents[-1]["t_end"] if sents else None,
        "num_sentences": len(sents),
        "sentences": sents,
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"sentences: {len(sents)}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
