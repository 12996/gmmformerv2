# Parent plans; subagent does the work

Date: 2026-09-17  
Trigger: user asked 「你现在在做什么怎么没有调用子agent开始工作而是自己干起来了」

## Rule

After the next experiment is locked:

1. Parent writes `方向探索日志.md` / `工作状态.md` and gives the 抓手.
2. Parent **spawns a subagent** for download, extract, train, watch.
3. Parent does **not** run the long job itself (wget, gdown, multi-connection download, unzip, `main.py`).

Exception: a short read-only check (`ps`, occupancy, `ls`, log tail) to report `blocked`/`pass` in the same turn (`blocked-must-report.md`).

## What went wrong

User said use laptop port 7897. Parent killed the remote downloader and started `tvr_parallel_download.py` + `tvr_pipe_upload.py` in this session instead of spawning a subagent.

## Check

If the next step is more than a status probe, there must be a live subagent id in the reply.
