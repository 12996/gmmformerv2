#!/bin/bash
# E2: extract projected CLIP token text to a NEW h5, then backup+replace L14 name.
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
PY="$ROOT/conda/bin/python"
TEXTDIR="$ROOT/data/prvr/charades/TextData"
DST="$TEXTDIR/charades_clip_B32_proj.h5"
LOG="$ROOT/logs/e2_extract_proj.log"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=
export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HOME="$ROOT/.cache/hf"
export TORCH_HOME="$ROOT/.cache/torch"
export CLIP_PT="$ROOT/.cache/clip/ViT-B-32.pt"
mkdir -p "$ROOT/logs"

{
  echo "START $(date -Is)"
  echo "HOST $(hostname)"
  echo "DST $DST"
  echo "CLIP_PT $CLIP_PT"
  ls -lh "$CLIP_PT"
  "$PY" -u "$ROOT/experiments/extract_clip_text_proj.py" \
    --captions "$TEXTDIR/charadestrain.caption.txt" "$TEXTDIR/charadesval.caption.txt" \
    --dst "$DST" \
    --model ViT-B-32 \
    --pretrained openai \
    --batch-size 64 \
    --device cpu
  "$PY" - <<'PY'
import os
import h5py

root = "/data/zhaopu/wang-2024-gmmformer-v2"
textdir = os.path.join(root, "data/prvr/charades/TextData")
dst = os.path.join(textdir, "charades_clip_B32_proj.h5")

def ids(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line.split(" ", 1)[0])
    return out

train = ids(os.path.join(textdir, "charadestrain.caption.txt"))
val = ids(os.path.join(textdir, "charadesval.caption.txt"))
need = list(dict.fromkeys(train + val))
with h5py.File(dst, "r") as h:
    keys = set(h.keys())
    missing = [k for k in need if k not in keys]
    shapes = []
    for k in need[:8] + val[:8]:
        shapes.append((k, h[k].shape, str(h[k].dtype)))
    ndims = {h[k].shape[-1] for k in list(h.keys())[:200]}
print("N_TRAIN_CAP", len(train), "N_VAL_CAP", len(val), "UNIQUE", len(need))
print("H5_KEYS", len(keys), "MISSING", len(missing), missing[:10])
print("SAMPLE_SHAPES", shapes)
print("DIMS_SAMPLE", ndims)
if missing:
    raise SystemExit("missing cap ids")
if ndims != {512}:
    raise SystemExit("unexpected dim %s" % ndims)
print("EXTRACT_OK")
PY
  echo "END $(date -Is)"
} >"$LOG" 2>&1
echo "LOG $LOG"
