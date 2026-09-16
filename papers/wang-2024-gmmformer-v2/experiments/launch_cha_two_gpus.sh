#!/bin/bash
# Independent Charades CLIP trains on GPUs with no compute process.
set -euo pipefail
ROOT="/data/zhaopu/wang-2024-gmmformer-v2"
cd "$ROOT"
mkdir -p "$ROOT/logs"
mapfile -t FREE < <(python3 - <<'PY'
import subprocess
q = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
busy = set()
try:
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
except subprocess.CalledProcessError:
    apps = ""
uuids = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
    text=True,
)
u2i = {}
for line in uuids.splitlines():
    idx, uid = [x.strip() for x in line.split(",", 1)]
    u2i[uid] = idx
for line in apps.splitlines():
    if not line.strip():
        continue
    uid, _pid = [x.strip() for x in line.split(",", 1)]
    if uid in u2i:
        busy.add(u2i[uid])
free = []
for line in q.splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    if idx not in busy and float(used) < 2048:
        free.append(idx)
print("\n".join(free))
PY
)
echo "FREE_GPUS ${FREE[*]:-none}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
if [ "${#FREE[@]}" -lt 1 ]; then
  echo "NO_FREE_GPU"
  exit 2
fi
SEEDS=(9527 9528 9529 9530 9531 9532 9533 9534)
i=0
for gpu in "${FREE[@]}"; do
  seed="${SEEDS[$i]}"
  nohup "$ROOT/run_cha_gpu.sh" "$gpu" "$seed" >/dev/null 2>&1 &
  echo "LAUNCH gpu${gpu} seed${seed} pid=$!"
  i=$((i + 1))
  if [ "$i" -ge "${#SEEDS[@]}" ]; then
    break
  fi
done
sleep 3
ps -u zhaopu -o pid,cmd | grep -E 'main.py -d cha' | grep -v grep || true

