# Paper Lab

This repository is a reproduction-to-adaptation workbench, not a note vault.

Main path: **intake → setup → verify → adapt → line**.
Do not start structured notes or writeups until `VERIFY.md` exists for that paper.

## Skills

Canonical files live in `.agents/skills/<name>/SKILL.md` (plural `.agents`).
Codex and Grok load them automatically. Claude Code does not — when a task matches the table below, **read that SKILL.md first and follow it**. Do not improvise the procedure from memory.

| When | Read |
|---|---|
| New paper, PDF, arXiv, official GitHub, 收一篇 / 读这篇 / 加一篇 | `.agents/skills/paper-intake/SKILL.md` |
| Clone, env, first official run, 跑通官方代码 | `.agents/skills/repro-setup/SKILL.md` |
| Match paper tables/metrics, 对表格 | `.agents/skills/paper-verify/SKILL.md` |
| Swap in our data/task, 换到我的数据 / 接到我的问题上 | `.agents/skills/paper-adapt/SKILL.md` |
| Update the research line, 下一步实验 | `.agents/skills/research-line/SKILL.md` |

New skills are created only under `.agents/skills/<name>/`. Do not write skill bodies under `.claude/`, `.grok/`, `.agent/`, or `.codex/`.

## Layout

```
.agents/skills/          # the only skill source
templates/               # copy these, then fill
papers/<slug>/           # one directory per paper
  PAPER.md SETUP.md VERIFY.md ADAPT.md
  experiments/           # commands, logs, numbers
  vendor/                # official clone (gitignored)
problems/<name>.md       # our task/data that adapt targets
lines/<theme>/LINE.md    # research line
outputs/                 # later writeups; not an entry point
```

Slug: `firstauthor-year-shorttitle`, lowercase kebab-case.

`vendor/` is the official project. Do not treat it as our research code. Record adaptation edits in `ADAPT.md`.

## Status on a paper

`PAPER.md` has a `Status` field. Move it only forward:
`intake` → `setup` → `verify` → `adapt` → `lined`.
A later skill may run only if the previous artifact exists (partial setup is allowed if `SETUP.md` says what failed).
