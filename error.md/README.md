# error.md

This folder is **lab law from real mistakes**, not a diary.

- Each file is one binding constraint. Later sessions must follow it without the user repeating it.
- Write a file when the user calls out a failure (missed report, wrong handle, overwrote status, etc.).
- Do not dump training curves or paper notes here. Those stay in `papers/<slug>/experiments/`.
- `AGENTS.md` points here; if a rule here conflicts with a habit, **this folder wins over habit**.

Index:

| file | constraint |
|---|---|
| `blocked-must-report.md` | No data / no process / GPUs full → say `blocked` in the same reply; do not wait for the user to ask |
| `parent-spawns-subagent.md` | Parent plans and reports; long jobs (download/train) go to a subagent, not the parent |
| `download-must-resume.md` | Dataset download retries and resumes; SSL flicker is not a reason to exit |
| `prefer-baidu-or-browser.md` | MS-SL zips: Baidu first, then logged-in browser; not Drive chunked gdown |
