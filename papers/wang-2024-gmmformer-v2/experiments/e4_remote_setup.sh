#!/bin/bash
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
SRC=$ROOT/vendor/GMMFormer_v2/src
cp -a $ROOT/experiments/cha_e4_5090_1.py $SRC/Configs/cha.py
python3 $ROOT/experiments/patch_e4_i3d_cliptext_loader.py
cp -a $ROOT/experiments/run_cha_e4_gpu.sh $ROOT/run_cha_e4_gpu.sh
chmod +x $ROOT/run_cha_e4_gpu.sh
grep -n 'visual_feature\|q_feat_size\|visual_feat_dim' $SRC/Configs/cha.py
python3 - <<'PY'
from pathlib import Path
t=Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Datasets/builder.py").read_text()
i=t.find("if cfg['visual_feature'] in ('i3d'")
print(t[i:i+650])
mc=Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py").read_text()
print("TC_SUM", "out = torch.sum(oo * weight.unsqueeze(2).repeat(1, 1, oo.shape[2], 1), dim=-1)" in mc)
print("TC_MEAN", "out = torch.mean(oo, dim = -1).squeeze()" in mc or "out = torch.mean(oo, dim=-1).squeeze()" in mc)
PY
