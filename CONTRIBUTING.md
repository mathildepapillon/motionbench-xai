# Contributing to MotionBench-XAI

Thanks for your interest! Contributions that extend the benchmark are
especially welcome: new imputers, new player sets, new attribution methods,
and new datasets with tractable oracles.

## Setup

```bash
python3 -m pip install -e ".[dev]"
pre-commit install            # optional: ruff + hygiene hooks
```

## Extending the benchmark

| To add a … | Implement | Register |
|---|---|---|
| completion model | `motionbench.imputers.base.BaseImputer` (`fit`, `impute`; observed entries preserved bit-for-bit) | a `configs/methods/*.yaml` with your imputer `_target_` and a `game:` key |
| player set | `motionbench.players.base.PlayerSet` (`n_players`, `coalition_mask`, `aggregate`) | a `configs/players/*.yaml` |
| attribution method | `motionbench.attribution.base.BaseAttributor` (`attribute(x, players, target) -> (M,)`) | a `configs/methods/*.yaml` |
| dataset | `motionbench.data.base.BaseDataset`; expose `.oracle` if a closed-form conditional exists | a `configs/data/*.yaml` |

Shape conventions everywhere: per-sample layout `(J, F, T)`; boolean masks
`(J, F, T)` with `True` = observed; per-player attributions `(M,)`.

## Before you open a PR

```bash
ruff check . && ruff format --check .   # lint / format
pytest tests/ -m "not slow and not gpu and not manual"
```

- Every public class/function needs a docstring stating shapes and
  conventions (see the existing modules for the house style).
- New numerics need tests — deterministic where possible, MC-vs-closed-form
  with justified tolerances otherwise (see `tests/test_oracle_gate.py`).
- No absolute machine-specific paths in package code or configs; use config
  keys or environment variables (`scripts/configure_paths.sh`).
- Conventions the paper's prose leaves open belong in
  `docs/CONVENTIONS.md`.

## Reporting issues

Please include the exact config (or CLI overrides), the package version /
commit, and — for numerical discrepancies — the seed and a minimal script.
