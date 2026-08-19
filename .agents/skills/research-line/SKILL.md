---
name: research-line
description: >
  Fold a paper’s setup/verify/adapt result back into a research line.
  Update lines/<theme>/LINE.md with role, gaps, and the next experiment.
  Use when the user says 更新研究线, 下一步做什么, 这条线, research line,
  or /research-line.
---

# research-line

Update one research line. Do not open a second theme unless the user names it.

## Inputs

Need a theme name and a paper slug. Infer the theme from existing `lines/*/LINE.md` when there is only one; otherwise ask. Read that paper’s `PAPER.md` plus `VERIFY.md` / `ADAPT.md` if they exist.

## Steps

1. If `lines/<theme>/LINE.md` is missing, copy `templates/LINE.md` and fill **Question** with the user.
2. Add or update the paper row: slug, role (`baseline` / `related` / `ours`), status from `PAPER.md`.
3. Rewrite **Gaps** from what this paper failed to cover on our problem — not a generic “future work” list from the PDF.
4. Write **Next experiment** as one concrete change, with data, metric, and done-when. One next step, not a roadmap.
5. Set `PAPER.md` **Status** to `lined` if verify or adapt already happened.

## Do not

- Start a new line for every paper
- Turn this into a survey or blog post
- Invent the next experiment if the user already stated it — use theirs

## Done

- `lines/<theme>/LINE.md` lists this paper
- **Next experiment** is specific enough to start `paper-adapt` or a new run without another planning session
