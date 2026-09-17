#!/bin/bash
# ActivityNet E4 CLIP projected token sequences. Tar has no CLIP text.
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
PY="$ROOT/conda/bin/python"
TEXTDIR="$ROOT/data/prvr/activitynet/TextData"
DST="$TEXTDIR/activitynet_clip_B32_proj.h5"
LOG="$ROOT/logs/act_e4_extract_proj.log"
GPU="${1:-}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HOME="$ROOT/.cache/hf"
export TORCH_HOME="$ROOT/.cache/torch"
export CLIP_PT="$ROOT/.cache/clip/ViT-B-32.pt"
if [ -n "$GPU" ]; then
  if [ "$GPU" = "2" ] || [ "$GPU" = "6" ] || [ "$GPU" = "7" ]; then
    echo REFUSE_GPU"$GPU"
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$GPU"
  DEV=cuda
else
  export CUDA_VISIBLE_DEVICES=
  DEV=cpu
fi
mkdir -p "$ROOT/logs"
{
  echo "START $(date -Is) gpu=${GPU:-cpu} dest=$DST"
  ls -lh "$CLIP_PT"
  ls -lh "$TEXTDIR"/activitynet{train,val,test}.caption.txt
  if [ -f "$DST" ] && [ -s "$DST" ]; then
    echo CLIP_ALREADY
  else
    "$PY" -u "$ROOT/experiments/extract_clip_text_proj.py" \
      --captions "$TEXTDIR/activitynettrain.caption.txt" "$TEXTDIR/activitynetval.caption.txt" "$TEXTDIR/activitynettest.caption.txt" \
      --dst "$DST" \
      --model ViT-B-32 \
      --pretrained openai \
      --batch-size 64 \
      --device "$DEV"
  fi
  "$PY" - <<'PY'
import os
import h5py

textdir = "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/activitynet/TextData"
dst = os.path.join(textdir, "activitynet_clip_B32_proj.h5")

def ids(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line.split(" ", 1)[0])
    return out

need = []
for sp in ("train", "val", "test"):
    need.extend(ids(os.path.join(textdir, "activitynet%s.caption.txt" % sp)))
need = list(dict.fromkeys(need))
with h5py.File(dst, "r") as h:
    keys = set(h.keys())
    missing = [k for k in need if k not in keys]
    sample = [(k, h[k].shape, str(h[k].dtype)) for k in need[:6]]
    ranks = {len(h[k].shape) for k in list(h.keys())[:200]}
    dims = {h[k].shape[-1] for k in list(h.keys())[:200]}
print("UNIQUE", len(need), "H5", len(keys), "MISSING", len(missing), missing[:8])
print("SAMPLE", sample)
print("RANKS", ranks, "DIMS", dims)
if missing:
    raise SystemExit("missing cap ids")
if ranks != {2}:
    raise SystemExit("not token sequence %s" % ranks)
if dims != {512}:
    raise SystemExit("bad dim %s" % dims)
print("EXTRACT_OK")
PY
  echo "END $(date -Is)"
} >"$LOG" 2>&1
echo "LOG $LOG"
tail -n 40 "$LOG"
