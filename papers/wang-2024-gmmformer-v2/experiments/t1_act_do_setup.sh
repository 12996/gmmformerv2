#!/bin/bash
# T1-act setup on 5090_1. Do not touch cha.py or Charades tmp/h5.
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
SRC="$ROOT/vendor/GMMFormer_v2/src"
CFG="$SRC/Configs"
echo "START_SETUP $(date -Is)"

echo "==== backup act.py / builder / main / validations ===="
ts=t1act_20260917
if [ ! -f "$CFG/act.py.clip.bak_$ts" ]; then
  cp -a "$CFG/act.py" "$CFG/act.py.clip.bak_$ts"
  echo "WROTE $CFG/act.py.clip.bak_$ts"
else
  echo "KEEP $CFG/act.py.clip.bak_$ts"
fi
for f in "$SRC/Datasets/builder.py" "$SRC/main.py" "$SRC/Validations/validations.py"; do
  bak="$f.bak_$ts"
  if [ ! -f "$bak" ]; then
    cp -a "$f" "$bak"
    echo "WROTE $bak"
  else
    echo "KEEP $bak"
  fi
done

echo "==== install act.cfg + patches ===="
cp -a "$ROOT/experiments/act_5090_1.py" "$CFG/act.py"
cp -a "$ROOT/experiments/run_act_gpu.sh" "$ROOT/run_act_gpu.sh"
cp -a "$ROOT/experiments/run_act_e4_gpu.sh" "$ROOT/run_act_e4_gpu.sh"
chmod +x "$ROOT/run_act_gpu.sh" "$ROOT/run_act_e4_gpu.sh" \
  "$ROOT/experiments/run_act_gpu.sh" "$ROOT/experiments/run_act_e4_gpu.sh" \
  "$ROOT/experiments/t1_act_do_setup.sh" "$ROOT/experiments/extract_act_clip_text.sh"
"$ROOT/conda/bin/python" "$ROOT/experiments/patch_act_i3d_loader.py"

echo "==== act.py keys ===="
grep -n 'visual_feature\|q_feat_size\|visual_feat_dim\|hidden_size\|batchsize\|seed\|max_ctx_l\|hard_negative\|max_es_cnt\|lr\|data_root\|root\|collection' "$CFG/act.py"
echo "==== cha.py must stay ===="
grep -n 'visual_feature\|q_feat_size\|visual_feat_dim' "$CFG/cha.py"
echo "==== TC mean ===="
COMP="$SRC/Models/gmmformerV2/model_components.py"
if grep -n 'out = torch.mean(oo, dim = -1).squeeze()' "$COMP"; then
  echo MEAN_STILL_PRESENT >&2
  exit 1
fi
grep -n 'out = torch.sum(oo \* weight' "$COMP"
echo "NO_MEAN_OK"

echo "==== forbidden tmp/h5 exist ===="
ls -ld "$ROOT/tmp/clip_e2_seed9527" "$ROOT/tmp/i3d_seed9527" "$ROOT/tmp/i3d_tc_seed9527" "$ROOT/tmp/clip_roberta_e3_seed9527" "$ROOT/tmp/i3d_cliptext_e4_seed9527"
ls -ld "$ROOT/tmp/tvr_i3d_seed9527" "$ROOT/tmp/tvr_i3d_cliptext_e4_seed9527"
ls -lh "$ROOT/data/prvr/charades/FeatureData/charades_i3d.h5" "$ROOT/data/prvr/charades/TextData/roberta_charades_query_feat.hdf5"
ps -p 564493,613385 -o pid,etime,cmd || echo TVR_PIDS_CHANGED
if [ -e "$ROOT/data/prvr/charades/official/Charades_v1_480.zip" ]; then
  echo "CHARADES_ZIP_PRESENT_DO_NOT_UNZIP"
  ls -lh "$ROOT/data/prvr/charades/official/Charades_v1_480.zip" || true
fi

echo "==== occupancy ===="
python3 "$ROOT/experiments/e2_gpu_occupancy.py" || python3 - <<'PY'
import subprocess
q = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"], text=True)
print(q)
PY

echo "==== pick gpu leftover>=10GiB, never 2/6/7 ===="
python3 "$ROOT/experiments/pick_act_gpu.py"
python3 - <<'PY'
import subprocess
q = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"], text=True)
for line in q.splitlines():
    idx, used, total, util = [x.strip() for x in line.split(",")]
    leftover = float(total) - float(used)
    print("GPU%s leftover=%.0f used=%s util=%s" % (idx, leftover, used, util))
PY

echo "==== ntrain ===="
"$ROOT/conda/bin/python" "$ROOT/experiments/t1_act_ntrain_check.py"
echo "SETUP_OK"
