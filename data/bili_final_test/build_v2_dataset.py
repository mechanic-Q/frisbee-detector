"""VLM 过滤硬负样本候选 → 构建 frisbee_merged_v2 数据集。
在 WSL 运行: python3 data/bili_final_test/build_v2_dataset.py
流程: 480P 误检裁剪 → GLM-4V-Flash 逐个确认"非飞盘" → 复制 merged → v2/images/train 加入空标签负样本。
"""
import json
import shutil
import time
from pathlib import Path

import cv2
from openai import OpenAI

ROOT = Path("/mnt/e/frisbee-detector")
CAND = ROOT / "data/bili_final_test/hardneg_candidates"
MERGED = ROOT / "data/datasets/frisbee_merged"
V2 = ROOT / "data/datasets/frisbee_merged_v2"
CONF_MIN = 0.45

# 复用项目内已配置的 GLM 客户端（与 tools/vlm_review_crops.py 一致）
import re
key_match = re.search(r'GLM_API_KEY = "([^"]+)"',
                      (ROOT / "tools/vlm_review_crops.py").read_text())
client = OpenAI(api_key=key_match.group(1),
                base_url="https://open.bigmodel.cn/api/paas/v4")

PROMPT = ("这张图片里有没有飞盘(ultimate frisbee / flying disc)？"
          "只回答 YES 或 NO。")


def is_frisbee(img_path: Path) -> bool:
    """True=含飞盘, False=不含(即负样本)"""
    img = cv2.imread(str(img_path))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return True
    import base64
    b64 = base64.b64encode(buf).decode()
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="glm-4v-flash",
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ]}],
                max_tokens=4, temperature=0.1)
            ans = (r.choices[0].message.content or "").strip().upper()
            if "YES" in ans:
                return True
            if "NO" in ans:
                return False
        except Exception as e:
            print(f"  retry {attempt}: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return True  # 不确定时保守：不当作负样本


def main():
    rows = [l.split(",") for l in (CAND / "crops.csv").read_text().splitlines()[1:] if l.strip()]
    cand = [(r[0], int(r[1]), float(r[6])) for r in rows if float(r[6]) >= CONF_MIN]
    print(f"candidates (conf>={CONF_MIN}): {len(cand)} / {len(rows)}")

    # 1) VLM 过滤
    negatives, uncertain = [], 0
    for i, (name, frame, conf) in enumerate(cand):
        p = CAND / name
        if not p.exists():
            continue
        if is_frisbee(p):
            pass
        else:
            negatives.append(name)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(cand)} checked, negatives so far {len(negatives)}", flush=True)
    print(f"VLM 确认负样本: {len(negatives)}")

    # 2) 构建 v2
    if V2.exists():
        shutil.rmtree(V2)
    print("copying merged -> v2 ...", flush=True)
    shutil.copytree(MERGED, V2, ignore=shutil.ignore_patterns("*.cache"))
    for name in negatives:
        stem = "bili480neg_" + Path(name).stem
        shutil.copy2(CAND / name, V2 / "images/train" / f"{stem}.jpg")
        (V2 / "labels/train" / f"{stem}.txt").write_text("")
    n_train = len(list((V2 / "images/train").glob("*.*")))
    (V2 / "_v2_meta.json").write_text(json.dumps({
        "based_on": "frisbee_merged", "added_negatives": len(negatives),
        "source": "bili_final_480p_fp_crops_vlm_confirmed",
        "conf_min": CONF_MIN, "train_imgs": n_train,
    }, indent=2))
    print(f"v2 train images: {n_train} (negatives +{len(negatives)})")

    # 3) yaml
    yaml_text = (ROOT / "configs/frisbee_merged.yaml").read_text().replace(
        "frisbee_merged\n", "frisbee_merged_v2\n").replace(
        "path: /mnt/e/frisbee-detector/data/datasets/frisbee_merged",
        "path: /mnt/e/frisbee-detector/data/datasets/frisbee_merged_v2")
    (ROOT / "configs/frisbee_merged_v2.yaml").write_text(yaml_text)
    print("wrote configs/frisbee_merged_v2.yaml")


if __name__ == "__main__":
    main()
