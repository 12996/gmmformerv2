---
name: paper-adapt
description: >
  Apply a verified official repo to our problem with the smallest possible change.
  Write papers/<slug>/ADAPT.md. Use when the user says 换到我的数据, 接到我的问题上,
  swap dataset/task, adapt the paper, or /paper-adapt.
  Do not rewrite the official project or start from a clean reimplementation.
---

# paper-adapt

Connect the official code to one file in `problems/`. Prefer a data/task swap over a method rewrite.

## Inputs

- `papers/<slug>/VERIFY.md` must exist (mismatch is fine).
- A target `problems/<name>.md`. If missing, copy `templates/PROBLEM.md`, fill it with the user, then continue.

## Steps

1. Copy `templates/ADAPT.md` to `papers/<slug>/ADAPT.md`.
2. Map paper I/O (dataset class, config, eval script) onto the problem file. Keep the method code unless it cannot run on our I/O.
3. Make the smallest edit that loads our data and reports our metric. List every touched vendor file in **Changes**.
4. Run on our problem. Put numbers in **Results** and logs in `papers/<slug>/experiments/`.
5. Write **Not comparable** whenever splits, metrics, or compute differ from the paper.
6. Set `PAPER.md` **Status** to `adapt`.

## Do not

- Reimplement the paper from scratch
- Refactor vendor code you did not need to touch
- Quietly change the method and still call it a reproduction

## Done

- `ADAPT.md` has mapping, file list, and our metric
- User can see what is not comparable to the paper
- Next skill is `research-line`
