---
name: repro-setup
description: >
  Clone the official paper repo, install the env, and run the first official command.
  Write papers/<slug>/SETUP.md. Use when the user says 跑通官方代码, 复现环境,
  clone the official repo, or /repro-setup.
  Do not change the method or compare tables yet.
---

# repro-setup

Get the official project to run once. Do not adapt it.

## Inputs

Require `papers/<slug>/PAPER.md`. If missing, run `paper-intake` first. Confirm the slug with the user if several papers exist.

## Steps

1. Clone the official repo into `papers/<slug>/vendor/` (gitignored). Checkout the commit pinned in `PAPER.md` when one exists.
2. Follow that repo’s README: env, CUDA, data. Put large data outside git (see `.gitignore`). Record exact commands in `SETUP.md`.
3. Copy `templates/SETUP.md` to `papers/<slug>/SETUP.md`.
4. Run **one** official command. Prefer the README quickstart or the claimed eval command. Do not start a multi-day official training unless the user asks.
5. Mark `result: pass` only if that command finished without error. On failure, fill **blockers** and stop.
6. Set `PAPER.md` **Status** to `setup` only on pass.

## Do not

- Edit vendor files to “make it ours”
- Invent a different training script
- Fill `VERIFY.md` here

## Done

- `papers/<slug>/vendor/` exists (or blocker says clone/env failed)
- `SETUP.md` has the command and pass/fail
- Next skill is `paper-verify` if pass
