---
name: paper-intake
description: >
  Ingest a new paper into this lab. Build papers/<slug>/PAPER.md with the problem,
  method bullets, official repo, tables to reproduce, hardware, and license.
  Use when the user adds a paper, drops a PDF/arXiv/GitHub link, or says 收一篇,
  读这篇, 加一篇, 新论文, or /paper-intake.
  Do not clone the official repo, write a full reading note, or start a writeup.
---

# paper-intake

Create the paper card only. Stop when `PAPER.md` is filled and status is `intake`.

## Inputs

Need at least one of: PDF path, arXiv id/url, title, official repo url. If more than one paper is implied, do them one at a time.

## Steps

1. Choose `slug` = `firstauthor-year-shorttitle` (lowercase kebab-case). If the folder exists, reuse it; do not create a second slug for the same paper.
2. Copy `templates/PAPER.md` to `papers/<slug>/PAPER.md`. Create `papers/<slug>/experiments/`.
3. Fill the card from the paper and repo README. Keep **Method** to 3–6 bullets. Extract every table/figure number you are expected to match later. Pin a commit if the README names one.
4. If `problems/` has an obvious target, set **Adaptation target**. Otherwise leave `TBD` and ask once.
5. Set **Status** to `intake`.

## Do not

- Clone into `vendor/`
- Write `notes.md`, articles, or a survey
- Implement or “improve” the method

## Done

- `papers/<slug>/PAPER.md` exists
- Official code url (or explicit “none”) is filled
- **Tables to match** has at least one row, or an explicit reason there is nothing to match
- Tell the user the slug and that the next skill is `repro-setup`
