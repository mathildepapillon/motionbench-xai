# TASK 6B — Leaderboard generation

**Phase:** 6 | **Tag:** [mechanical] | **Depends on:** 6A | **PR title:** `[6B] Leaderboard`

## Worktree setup

```bash
git worktree add ../mbxai-task-6B-leaderboard -b task/6B-leaderboard
```

## Files to create

```
scripts/generate_leaderboard.py
docs/leaderboard.md
```

## Spec

```bash
python scripts/generate_leaderboard.py --results_dir results/ --output docs/leaderboard.md
```

- One Markdown table per (dataset, metric) pair.
- Rows = methods, sorted by score.
- Columns: method, score (mean ± std across test sequences), rank.
- Optionally generate a Hugging Face Space (common for D&B 2025–2026 papers).

## Definition of done

- [ ] Script runs on dummy results fixture
- [ ] ruff pass
- [ ] TASKS.md row 6B: done, notes
- [ ] PR: `[6B] Leaderboard`
