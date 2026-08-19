# paper-lab

研究生读论文工作台：把官方代码跑通、对上论文表格，再换到自己的数据/任务上。

技能正文只在 `.agents/skills/`。用法和目录约定见 [AGENTS.md](AGENTS.md)。

## 主路径

1. 收论文 → `papers/<slug>/PAPER.md`
2. 跑通官方仓库 → `SETUP.md`
3. 对照表格 → `VERIFY.md`
4. 接到 `problems/` 里的问题上 → `ADAPT.md`
5. 折回研究线 → `lines/<theme>/LINE.md`

用 Codex、Grok 或 Claude Code 在本仓库打开即可。前两个会自动发现 skills；Claude 会读 `AGENTS.md` / `CLAUDE.md` 再打开对应 `SKILL.md`。
