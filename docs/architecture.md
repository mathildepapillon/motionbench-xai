# Architecture — The Four Base Abstractions

> This document describes the four locked base abstractions in motionbench-xai.
> **These interfaces are frozen after Phase 0.** All agents must read this
> before writing any module-level code.

---

## Overview

MotionBench-XAI evaluates XAI attribution methods on time-series motion data.
Every evaluation pipeline is built from four composable abstractions:

```
Dataset ──► PlayerSet ──► Attributor ──► Metric
                │
                ▼
          Oracle / Imputer
```

1. A **Dataset** provides (x, y) pairs and an optional **Oracle**.
2. A **PlayerSet** defines how a sequence is partitioned into players.
3. An **Attributor** produces per-player attribution scores φ.
4. A **Metric** evaluates φ against oracle, imputer, or classifier.
5. An **Oracle** / **Imputer** estimates the conditional expectation v(S).

---

## 1. PlayerSet (`motionbench/players/base.py`)

Defines the SHAP game's player structure. A player is the atomic unit of
explanation — all attributors output one number per player.

```python
class PlayerSet(ABC):
    n_players: int          # M — number of players
    shape: tuple[int,int,int]  # (J, F, T) coordinate space

    def coalition_mask(z: Tensor) -> Tensor:
        # (M,) binary → (J, F, T) bool element mask
        # Indivisible masking: hiding player k hides ALL its coordinates

    def aggregate(phi_coords: Tensor) -> Tensor:
        # (J, F, T) attribution map → (M,) per-player scores
        # Uses Shapley additivity: φ_k = Σ_{i ∈ group_k} φ_i
```

**Canonical implementations:**

| Class | Players | Mask layout |
|-------|---------|-------------|
| `TemporalWindows(K, T)` | K equal time windows | (J, F, T) temporal stripes |
| `SpatialJoints(J)` | J skeletal joints | (J, F, T) joint blocks |
| `AnatomicalGroups(groups)` | predefined joint groups | (J, F, T) arbitrary |
| `GaitPhase(n_phases)` | stride-aligned phases | (J, F, T) temporal phases |
| `JointWindowCells(J, K)` | J × K spatiotemporal cells | (J, F, T) grid |

---

## 2. Dataset (`motionbench/data/base.py`)

Structural protocol (no inheritance needed).

```python
class BaseDataset(Protocol):
    def __getitem__(idx: int) -> tuple[Tensor, Tensor]: ...  # (J,F,T), scalar
    def __len__() -> int: ...
    shape: tuple[int,int,int]  # (J, F, T)
    metadata: dict             # {"skeleton": ..., "frame_rate": ...}
    oracle: Optional[Oracle]   # None for real data

class GroundTruthDataset(BaseDataset, Protocol):
    oracle: Oracle             # required, non-Optional
```

**Synthetic datasets** implement `GroundTruthDataset` and expose a closed-form
oracle. **Real datasets** implement `BaseDataset` with `oracle=None`.

---

## 3. Oracle / Imputer (`motionbench/oracles/base.py`, `motionbench/imputers/base.py`)

Both share the same core operation: given an observed part of a sequence,
complete the hidden part. The oracle does this *exactly*; imputers do it
*approximately*.

```python
# Oracle (exact, synthetic only)
class Oracle(ABC):
    def conditional_sample(x_obs, mask, n, seed) -> Tensor:
        # mask: (J,F,T) bool, True=observed
        # return: (n, J, F, T) exact draws from p(x_hid | x_obs)

    def true_shapley(x, classifier, players, n_mc, seed) -> Tensor:
        # return: (M,) exact Shapley values (enumerate all 2^M coalitions if M≤12)

# Imputer (approximate, trainable)
class BaseImputer(ABC):
    def fit(train_data) -> BaseImputer: ...
    def impute(x_obs, mask, n_samples, seed) -> Tensor:
        # Same signature as Oracle.conditional_sample
        # Observed entries MUST be preserved bit-for-bit
```

**Key design decision:** The Oracle satisfies the BaseImputer interface.
This means you can pass an Oracle as an imputer to KernelSHAP to compute
"oracle SHAP" — the exact Shapley values under the true data distribution.
The EC1/EC2/EC3 metrics measure how far any other imputer deviates from this.

**The imputation-attribution boundary:** EC metrics only make sense when
the imputer and the oracle have the same player structure (same mask layout).
Mixing temporal oracle with spatial imputer is a bug.

---

## 4. Attributor (`motionbench/attribution/base.py`)

```python
class BaseAttributor(ABC):
    def __init__(classifier: Callable, **kwargs): ...
    def attribute(x: Tensor, players: PlayerSet, target: int) -> Tensor:
        # x: (J, F, T) — single sample, no batch dim
        # return: (M,) per-player attribution scores
```

**Subclass conventions:**
- `KernelShapAttributor(classifier, imputer, players, **kwargs)` — SHAP, pluggable imputer.
- Gradient-based methods do not take an imputer.
- All methods aggregate to player level via `players.aggregate(phi_coords)`.

---

## 5. Metric (`motionbench/metrics/base.py`)

```python
class BaseMetric(ABC):
    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = False

    def evaluate(phi, x, classifier, players, target, oracle, imputer) -> dict[str, float]:
        # phi: (M,) attribution vector
        # return: {"metric_name": float, ...}
```

**Metric taxonomy:**

| Category | requires_oracle | requires_imputer | Examples |
|----------|----------------|-----------------|---------|
| Ground-truth | True | False | EC1, EC2, EC3, TopK, Spearman |
| Fidelity | False | True | PixelFlipping, FaithfulnessCorr |
| Stability | False | False | MaxSensitivity, Continuity |
| Sanity | False | False | ModelParamRand, RandomLogit |
| Meta | False | False | RankingAgreement |

---

## Shape conventions (canonical)

All coordinates use `(J, F, T)` layout. No exceptions.

| Symbol | Meaning | Example |
|--------|---------|---------|
| J | Skeletal joints | 17 (H36M-17) |
| F | Features per joint | 3 (xyz) |
| T | Time-steps per clip | 81 (3 s at 27 fps) |
| M | SHAP players | 4 (temporal K=4), 17 (joints) |
| B | Batch size | 32 |

- Single sample: `(J, F, T)`
- Batch: `(B, J, F, T)`
- Coalition indicator: `(M,)` binary int/bool
- Element mask: `(J, F, T)` bool, `True = observed`
- Attribution: `(M,)` float

---

## Data flow through a complete evaluation

```python
# 1. Dataset provides data and oracle
dataset = GaussianMotionDataset(K=4, J=17, F=3, T=81)
x, y = dataset[0]          # (J, F, T), scalar

# 2. PlayerSet defines the game
players = TemporalWindows(K=4, T=81, J=17, F=3)

# 3. Imputer fills masked coordinates
imputer = VAEACImputer.load("checkpoints/vaeac.pt")
imputer.fit(dataset)

# 4. Attributor produces per-player scores
attributor = KernelShapAttributor(classifier, imputer=imputer)
phi = attributor.attribute(x, players, target=0)  # (M=4,)

# 5. Metric evaluates quality
metric = EC1Metric()
scores = metric.evaluate(phi, x, classifier, players, oracle=dataset.oracle)
# → {"ec1": 0.034}
```
