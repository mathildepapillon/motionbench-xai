# AGENTS.md — Operating discipline for all MotionBench-XAI agents

> This file is the **system prompt** for every agent working in this repository.
> Read it **in full** before writing any code.

---

## 0. Before starting any task

1. **Read `TASKS.md`** — find your task row, set `status: in-progress` and
   `agent: <your-worktree-name>`. Do this **before** writing a single line of code.
2. **Read `tasks/TASK_{ID}.md`** — your specific task spec. Read it entirely.
3. **Read `docs/architecture.md`** — the four base abstractions.
4. **Read `docs/SOURCE_MAP.md`** — old-code-to-new-code map (required if porting from CARE-PD).
5. **Read the `base.py`** for every module you will modify or import from.
   **Do not modify any `base.py` file.** These are locked for Phase 0.
6. If anything in the task spec is **ambiguous or contradicts a base interface**:
   STOP, write the issue into your `TASKS.md` row under `notes`, and ask
   the human before proceeding.

---

## 1. The non-negotiable rules

### 1.1 Never touch base interfaces
The following files are **FROZEN** after Phase 0. They may not be modified by any agent:

```
motionbench/players/base.py
motionbench/data/base.py
motionbench/oracles/base.py
motionbench/imputers/base.py
motionbench/attribution/base.py
motionbench/metrics/base.py
```

If you believe an interface is wrong, write your concern into `TASKS.md`
under your task row. Do not silently change.

### 1.2 Use the libraries — never re-derive algorithms
The following libraries are pre-approved. If a task asks you to implement
something already in these libraries, **wrap, don't rewrite**:

| Library | Use for |
|---|---|
| `captum` | IG, DeepLift, GradientShap, Saliency, SmoothGrad, GradCAM |
| `shap` | KernelExplainer, custom masker pattern |
| `quantus` | Fidelity, stability, sanity-check metrics |
| `zennit` | LRP (ε, γ, α-β variants) |
| `timeshap` | TimeSHAP wrapper |
| `windowshap` | WindowSHAP wrapper |
| `shats` | ShaTS wrapper |
| `pyvinecopulib` | Vine copula imputer |

### 1.3 Stay in your scope
Do **not** modify files that belong to another agent's task. If you discover
a bug or improvement outside your scope, log it in `BACKLOG.md` and move on.

### 1.4 Never import from in-progress files
When your task imports from another module, only import from:
- `<module>/base.py` (the locked interface), or
- A public `__init__.py` that is already committed to `main`.

Never reach into another agent's in-progress files.

### 1.5 Prefer the simplest correct implementation
If a task spec is ambiguous, prefer the simplest correct implementation and
note the ambiguity in `TASKS.md`. Do not over-engineer.

---

## 2. Code quality requirements

Every file you write must have:

- [ ] **Module docstring** — one-line summary at the top, plus a longer
  description of the module's role, inputs, outputs, and relevant references.
- [ ] **Full type hints** — all public functions and methods (arguments, return
  types, class variables). Use `from __future__ import annotations` for
  forward references.
- [ ] **Google-style docstrings** on all public functions. Include `Args:`,
  `Returns:`, and `Raises:` sections.
- [ ] **At least one unit test** in `tests/` for every public function or class.
  Tests live in `tests/test_<module>.py`.
- [ ] `ruff check .` passes with zero errors.
- [ ] `mypy motionbench/` passes (strict mode).

---

## 3. Test requirements

- Run `pytest tests/<your-module>/ -v` before marking your task done.
- Mark slow tests with `@pytest.mark.slow`.
- Mark manual/ablation tests with `@pytest.mark.manual`.
- Tests that require a GPU: `@pytest.mark.gpu`.
- CI only runs tests **not** marked with `slow`, `gpu`, or `manual`.
- Do **not** commit model checkpoints. Store checkpoint download URLs in `README.md`.

---

## 4. Finishing a task

When your task is complete:

1. Run: `pytest tests/<your-module>/ -m "not slow and not gpu and not manual"`
2. Run: `ruff check . && ruff format --check .`
3. Run: `mypy motionbench/<your-module>/`
4. Fix any errors. Do not skip.
5. Update `TASKS.md`: set `status: done` and write a **3-line summary** in the
   `notes` field covering: (1) what was implemented, (2) any design decisions
   or trade-offs, (3) anything deferred to `BACKLOG.md`.
6. Open a PR titled **exactly** as specified in `tasks/TASK_{ID}.md`.
7. The PR description must include: test results, ruff/mypy status, and links
   to any deferred items in `BACKLOG.md`.

---

## 5. Worktree conventions

- Your worktree is at `../mbxai-task-{ID}-<short-name>`.
- Create it with: `git worktree add ../mbxai-task-{ID}-<name> -b task/{ID}-<name>`
- Base your branch off `main`.
- Never merge to `main` yourself. Open a PR; the human merges.

---

## 6. Shape conventions (reference)

All of motionbench uses these conventions. Deviating causes silent bugs.

| Symbol | Meaning | Typical value |
|---|---|---|
| J | Number of skeletal joints | 17 (H36M) |
| F | Coordinates per joint | 3 (xyz) |
| T | Frames per clip | 81 (3 s at 27 fps) |
| M | Number of SHAP players | 4–17 |
| B | Batch size | varies |

- **Single sample:** `(J, F, T)` — no batch dimension in `attribute()` / `impute()`.
- **Batch:** `(B, J, F, T)` — classifier always receives batches.
- **Mask:** `(J, F, T)` bool — `True` = observed.
- **Attribution:** `(M,)` — one score per player (already aggregated).
- **Coalition:** `(M,)` int/bool — `1` = player included.

---

## 7. Determinism

Use `seed` arguments throughout. CI tests must pass with the seeds in
`tests/conftest.py`. Never use global random state.
