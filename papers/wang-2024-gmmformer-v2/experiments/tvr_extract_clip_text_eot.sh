#!/bin/bash
# TVR EOT CLIP: ln_final @ text_projection at the real EOT index. Never zip hdf5. Never GPU2.
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
PY="$ROOT/conda/bin/python"
TEXTDIR="$ROOT/data/prvr/tvr/TextData"
DST="$TEXTDIR/tvr_clip_B32_eot.h5"
PARTIAL="$DST.partial"
ZIP_EOT="$TEXTDIR/clip_ViT_B_32_tvr_query_feat.hdf5"
PROJ="$TEXTDIR/tvr_clip_B32_proj.h5"
LOG="$ROOT/logs/tvr_e4_extract_eot.log"
GPU="${1:-}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HOME="$ROOT/.cache/hf"
export TORCH_HOME="$ROOT/.cache/torch"
export CLIP_PT="$ROOT/.cache/clip/ViT-B-32.pt"
if [ -n "$GPU" ]; then
  if [ "$GPU" = "2" ]; then
    echo REFUSE_GPU2
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
  ls -lh "$TEXTDIR"/tvr{train,val,test}.caption.txt
  echo "ZIP_EOT_UNUSED $ZIP_EOT"
  if [ -f "$DST" ] && [ -s "$DST" ]; then
    echo CLIP_EOT_ALREADY
  else
    rm -f "$PARTIAL"
    "$PY" -u "$ROOT/experiments/extract_clip_text_eot.py" \
      --captions "$TEXTDIR/tvrtrain.caption.txt" "$TEXTDIR/tvrval.caption.txt" "$TEXTDIR/tvrtest.caption.txt" \
      --dst "$PARTIAL" \
      --model ViT-B-32 \
      --pretrained openai \
      --batch-size 64 \
      --device "$DEV"
    mv -f "$PARTIAL" "$DST"
  fi
  "$PY" - <<'PY'
import os
import h5py
import numpy as np

textdir = "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/tvr/TextData"
dst = os.path.join(textdir, "tvr_clip_B32_eot.h5")
proj = os.path.join(textdir, "tvr_clip_B32_proj.h5")
zip_eot = os.path.join(textdir, "clip_ViT_B_32_tvr_query_feat.hdf5")

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
    need.extend(ids(os.path.join(textdir, "tvr%s.caption.txt" % sp)))
need = list(dict.fromkeys(need))
with h5py.File(dst, "r") as h:
    keys = set(h.keys())
    missing = [k for k in need if k not in keys]
    k0 = need[0]
    arr = np.asarray(h[k0][...])
    print("ONE_KEY", k0, "ONE_SHAPE", arr.shape, "DTYPE", arr.dtype)
    if arr.shape == (512,):
        arr = arr.reshape(1, 512)
        print("EXPANDED", arr.shape)
    if tuple(arr.shape) != (1, 512):
        raise SystemExit("bad sample shape %s" % (arr.shape,))
    sample = [(k, h[k].shape, str(h[k].dtype)) for k in need[:6]]
    ranks = {len(h[k].shape) for k in list(h.keys())[:200]}
    dims = {h[k].shape[-1] for k in list(h.keys())[:200]}
    seq_lens = {int(np.asarray(h[k]).reshape(-1, h[k].shape[-1]).shape[0]) for k in need[:200]}
print("UNIQUE", len(need), "H5", len(keys), "MISSING", len(missing), missing[:8])
print("SAMPLE", sample)
print("RANKS", ranks, "DIMS", dims, "SEQ_LENS_HEAD", seq_lens)
print("ZIP_EOT_EXISTS", os.path.isfile(zip_eot), "NOT_USED")
if missing:
    raise SystemExit("missing cap ids")
if dims != {512}:
    raise SystemExit("bad dim %s" % dims)
if 0 in seq_lens:
    raise SystemExit("empty eot")
if os.path.isfile(proj):
    with h5py.File(dst, "r") as he, h5py.File(proj, "r") as hp:
        k = need[0]
        e = np.asarray(he[k][...], dtype=np.float32).reshape(-1)
        p = np.asarray(hp[k][...], dtype=np.float32)
        last = p[-1]
        cos = float(np.dot(e, last) / (np.linalg.norm(e) * np.linalg.norm(last) + 1e-8))
        print("MATCH_PROJ_LAST", k, "proj_shape", p.shape, "cos", cos)
        if cos < 0.999:
            raise SystemExit("eot not matching proj last token cos=%s" % cos)
print("EXTRACT_EOT_OK")
PY
  echo "END $(date -Is)"
} >"$LOG" 2>&1
echo "LOG $LOG"
tail -n 40 "$LOG"
