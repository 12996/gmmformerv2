---
name: paper-verify
description: >
  Reproduce the paper’s key tables or metrics and record paper vs ours in
  papers/<slug>/VERIFY.md. Use when the user says 对表格, 对一下数字, verify
  reproduction, match the paper, or /paper-verify.
  Do not swap datasets or start a writeup.
---

# paper-verify

Compare official-run numbers to the paper. Adaptation comes later.

## Inputs

Require `papers/<slug>/PAPER.md` and `SETUP.md`. If setup failed, only verify what still can run; say what you cannot measure.

Use the **Tables to match** rows in `PAPER.md` as the checklist. Do not pick random metrics.

## Steps

1. Copy `templates/VERIFY.md` to `papers/<slug>/VERIFY.md`.
2. For each checklist row, run the official eval (or the closest README command). Save raw logs under `papers/<slug>/experiments/`.
3. Fill paper / ours / delta / match (`yes` / `no` / `close`). `close` means same setup, small numerical drift. `no` needs a reason.
4. Set `PAPER.md` **Status** to `verify` after every checklist row is either filled or marked skipped with a reason.

## Do not

- Change data, task, or model to improve the score
- Skip writing a row because the number looks bad
- Write notes or articles

## Done

- `VERIFY.md` exists
- Every **Tables to match** row is accounted for
- Tell the user what matched and that the next skill is `paper-adapt` if they want it on our problem
