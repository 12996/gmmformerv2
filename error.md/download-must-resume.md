# Downloads must resume, not exit

Date: 2026-09-17  
Trigger: user asked 「停了什么意思…希望下载的时候能监控下载状态是否健康并且能断点继续下」

## Rule

A dataset download process must:

- Persist progress (`.parts.json` or equivalent).
- On SSL / timeout / proxy drop: retry that chunk, **do not `sys.exit`**.
- Print a health line (done/total, stall seconds, error count) on a timer.
- If the process still dies, a supervisor restarts it from the same state file.

“停了” is only allowed when the zip is complete and `ZipFile` opens, or the user cancels.

## What went wrong

`tvr_parallel_download.py` treated a burst of SSL EOF as fatal (`INCOMPLETE` exit 1) at 254/2121. Progress was on disk; the process left. Parent reported “stopped” instead of keeping the resume loop alive.
