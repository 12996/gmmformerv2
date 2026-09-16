#!/bin/bash
set -euo pipefail
ROOT="/data/zhaopu/wang-2024-gmmformer-v2"
PY="$ROOT/conda/bin/python"
LOG="$ROOT/logs/pip_rest.log"
export PYTHONUNBUFFERED=1
export http_proxy=http://127.0.0.1:17897
export https_proxy=http://127.0.0.1:17897
export all_proxy=http://127.0.0.1:17897
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export ALL_PROXY="$all_proxy"
export no_proxy=localhost,127.0.0.1
export NO_PROXY="$no_proxy"
{
  echo "START $(date -Is)"
  echo "PROXY $https_proxy"
  "$PY" -m pip install tqdm pyyaml ipdb numpy scikit-learn scipy easydict einops h5py matplotlib seaborn
  "$PY" - <<'PY'
import torch, h5py, yaml, einops, numpy, matplotlib
print("TORCH", torch.__version__, "CUDA", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("DEV0", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
print("H5PY", h5py.__version__, "NUMPY", numpy.__version__)
print("IMPORT_OK")
PY
  echo "DONE $(date -Is)"
} >"$LOG" 2>&1

