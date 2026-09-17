#!/bin/bash
# Launch Act E4 on leftover GPU4. Never 2/5/6/7. Do not touch T1/TVR.
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
GPU=4
SEED=9527
if [ "$GPU" = "2" ] || [ "$GPU" = "5" ] || [ "$GPU" = "6" ] || [ "$GPU" = "7" ]; then
  echo REFUSE_GPU"$GPU"
  exit 2
fi
SH="$ROOT/experiments/run_act_e4_gpu.sh"
RUN="$ROOT/logs/runs/act_e4_seed${SEED}_gpu${GPU}"
mkdir -p "$RUN" "$ROOT/logs"
chmod +x "$SH" "$ROOT/run_act_e4_gpu.sh"
if pgrep -u zhaopu -f 'PRVR_ROOT=.*/tmp/act_i3d_cliptext_e4' >/dev/null; then
  echo E4_ALREADY
  pgrep -u zhaopu -af 'main.py -d act'
  exit 0
fi
nohup "$SH" "$GPU" "$SEED" > "$RUN/launcher.out" 2>&1 &
echo LAUNCHER=$!
sleep 15
echo PIDFILE_BEGIN
cat "$RUN/pid"
echo PIDFILE_END
echo RUNJSON_BEGIN
cat "$RUN/run.json"
echo RUNJSON_END
PID=$(cat "$RUN/pid")
ps -p "$PID" -o pid,etime,cmd
echo KEEP_T1_TVR
ps -p 2913382,564493,613385 -o pid,etime,cmd
echo LOGTXT
ls -l "$ROOT/tmp/act_i3d_cliptext_e4_seed${SEED}/results/activitynet/gmmformer_v2/log.txt" || true
echo E4_LAUNCH_DONE
