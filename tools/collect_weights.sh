#!/bin/bash
# 安全权重搬运：从 ComfyUI 输出目录原子复制到项目 runs/detect。
# 用法: bash tools/collect_weights.sh <run_name> [src_base]
# 逻辑：先 cp 到 .tmp-<name>，验证 best.pt 存在且非空 → mv 原子替换；绝不先删原目录。
set -u
NAME="$1"
SRC_BASE="${2:-/home/lmr/comfy/ComfyUI/runs/detect/runs}"
SRC="$SRC_BASE/$NAME"
DST_BASE="${3:-/mnt/e/frisbee-detector/runs/detect}"
DST="$DST_BASE/$NAME"
TMP="$DST_BASE/.tmp-$NAME-$$"

if [ ! -d "$SRC" ]; then
  echo "SKIP: source missing: $SRC" >&2
  exit 1
fi
if [ ! -s "$SRC/weights/best.pt" ] && [ ! -s "$SRC/checkpoint_best_ema.pth" ]; then
  echo "SKIP: no usable weights in $SRC" >&2
  exit 1
fi

rm -rf "$TMP"
cp -r "$SRC" "$TMP" || { echo "FAIL: copy error, original untouched" >&2; rm -rf "$TMP"; exit 1; }

# 校验拷贝完整性
if [ ! -s "$TMP/weights/best.pt" ] && [ ! -s "$TMP/checkpoint_best_ema.pth" ]; then
  echo "FAIL: copy incomplete (no weights in tmp), original untouched" >&2
  rm -rf "$TMP"
  exit 1
fi

# 原子替换：旧目录改名备份 → 新目录就位 → 删除备份
if [ -d "$DST" ]; then
  mv "$DST" "$DST.bak-$$" || { echo "FAIL: cannot rename existing dst" >&2; rm -rf "$TMP"; exit 1; }
fi
mv "$TMP" "$DST" || { echo "FAIL: cannot move tmp into place" >&2; exit 1; }
rm -rf "$DST".bak-* 2>/dev/null
echo "OK: $NAME collected -> $DST"
