#!/bin/bash
# Install torch+deps into the 5090_1 prefix conda env.
set -euo pipefail
ROOT="/data/zhaopu/wang-2024-gmmformer-v2"
PY="$ROOT/conda/bin/python"
LOG="$ROOT/logs/pip_torch.log"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"
{
  echo "START $(date -Is)"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  "$PY" -m pip install tqdm pyyaml ipdb numpy scikit-learn scipy easydict einops h5py matplotlib seaborn \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
  "$PY" - <<'PY'
import torch, h5py, yaml, einops, numpy
print("TORCH", torch.__version__, "CUDA", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("DEV0", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
print("H5PY", h5py.__version__, "NUMPY", numpy.__version__)
PY
  echo "DONE $(date -Is)"
} >"$LOG" 2>&1
