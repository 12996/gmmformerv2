# Blocked must be reported in the same turn

Date: 2026-09-17  
Trigger: user asked 「你昨天没跑起来为什么不跟我说？」

## Rule

If a job **did not start**, **cannot start**, or **is not running**:

- Say **`blocked`** in that same reply.
- Give the reason (missing features, no `main.py`, no tmp dir, occupancy full, download failed).
- Do not describe “subagent still working” as progress when the train host has no process and no log.

Do this without waiting for 「现在怎么样了」.

## What went wrong

Spawned ActivityNet / TVR I3D and E4-structure jobs. `5090_1` `data/prvr/` only had `charades`. No `tmp/act_*` or `tmp/tvr_*`, no `main.py -d act|tvr`. Agents later idle-timed out. Parent did not tell the user until the next day.

## Check

On the train host: `ps` for `main.py`, `ls` the promised `PRVR_ROOT`, occupancy script. If those are empty, the result is `blocked`, not in-progress.
