"""Minimal end-to-end MotionBench-XAI evaluation — CPU-only, ~2 minutes.

One complete benchmark cell, self-contained (no checkpoints, no cluster,
no external data):

1. Generate the gauss_k4 synthetic dataset (J=5 joints, F=3 features,
   T=16 frames; equicorrelated joints, AR(1) time; tertile labels).
2. Train a small MLP classifier on it (plain torch loop).
3. Attribute test sequences with two KernelSHAP variants that differ only
   in their imputer: KS-Zero (off-manifold zero fill -> marginal game) and
   KS-Oracle (exact conditional sampler -> conditional game).
4. Grade both against the exact deterministic oracle of the game each
   method plays: EC1 (mean absolute error) and EC3 (1 - Pearson).

Run:  python examples/minimal_evaluation.py
"""

from __future__ import annotations

import numpy as np
import torch

from motionbench.attribution.kernel_shap import KernelShapAttributor
from motionbench.attribution.sampled_coalitions import (
    phi_from_values,
    sampled_coalition_set,
)
from motionbench.classifiers.synthetic_mlp import SyntheticMLPClassifier
from motionbench.data.synthetic.gaussian_motion import GaussianMotionDataset
from motionbench.imputers.off_manifold import ZeroImputer
from motionbench.oracles.deterministic import (
    DeterministicConditionalOracle,
    marginal_fill,
)
from motionbench.players.spatial_joints import SpatialJoints

J, F, T, K = 5, 3, 16, 4
N_TRAIN, N_EVAL, N_EXPLAIN = 600, 50, 10

# ---------------------------------------------------------------------------
# 1. Data — gauss_k4: equicorrelated joints (rho=0.5), AR(1) time (alpha=0.8)
# ---------------------------------------------------------------------------
torch.manual_seed(0)
train_ds = GaussianMotionDataset(J=J, F=F, T=T, N=N_TRAIN, rho=0.5, alpha=0.8, K=K, seed=1042)
eval_ds = GaussianMotionDataset(J=J, F=F, T=T, N=N_EVAL, rho=0.5, alpha=0.8, K=K, seed=42)

# ---------------------------------------------------------------------------
# 2. Classifier — small MLP on window-mean features
# ---------------------------------------------------------------------------
clf = SyntheticMLPClassifier(J=J, F=F, T=T, K=K, n_classes=3, hidden=64)
opt = torch.optim.Adam(clf.parameters(), lr=5e-3, weight_decay=1e-3)
x_tr = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
y_tr = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
clf.train()
for _epoch in range(60):
    perm = torch.randperm(len(x_tr))
    for s in range(0, len(x_tr), 64):
        idx = perm[s : s + 64]
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(x_tr[idx]), y_tr[idx])
        loss.backward()
        opt.step()
clf.eval()
with torch.no_grad():
    acc = float((clf(x_tr).argmax(-1) == y_tr).float().mean())
print(f"train accuracy: {acc:.2f}")

# ---------------------------------------------------------------------------
# 3. Attribution — spatial players (one player per joint), two imputers
# ---------------------------------------------------------------------------
players = SpatialJoints(J=J, F=F, T=T)
M = players.n_players
oracle = eval_ds.oracle  # exact GaussianOracle for this generative model

methods = {
    "KS-Zero  (marginal game)": (ZeroImputer().fit(eval_ds), "marg"),
    "KS-Oracle (conditional game)": (oracle, "cond"),
}

# Shared exact coalition design (M=5 -> all 32 coalitions) and the
# deterministic conditional-mean operators, precomputed once.
Z, w = sampled_coalition_set(M, budget=1024)
det = DeterministicConditionalOracle.from_oracle(oracle, players, Z)
masks_np = [
    players.coalition_mask(torch.as_tensor(z != 0)).numpy() for z in np.asarray(Z, dtype=np.int64)
]

print(f"\n{'method':30s} {'EC1':>8s} {'EC3':>8s}   (n={N_EXPLAIN} sequences)")
for name, (imputer, game) in methods.items():
    ec1s, ec3s = [], []
    for i in range(N_EXPLAIN):
        x, _y = eval_ds[i]
        with torch.no_grad():
            target = int(clf(x[None]).argmax(-1).item())

        def prob_fn(batch: torch.Tensor, _t: int = target) -> torch.Tensor:
            with torch.no_grad():
                return torch.softmax(clf(batch), dim=-1)[:, _t]

        # Method attribution via the package KernelSHAP attributor.
        attributor = KernelShapAttributor(
            prob_fn, imputer, n_samples=64, n_completion_samples=5, seed=42
        )
        phi = attributor.attribute(x, players, target=target).double().numpy()

        # Exact deterministic target of the game this method plays.
        x64 = x.numpy().astype(np.float64)
        if game == "cond":
            fills = det.fill_all(x64)  # E[x_hid | x_obs] per coalition
        else:
            fills = np.stack([marginal_fill(x64, m) for m in masks_np])  # E[x] = 0
        v_star = prob_fn(torch.from_numpy(fills)).double().numpy()
        phi_star = phi_from_values(Z, w, v_star)

        ec1s.append(float(np.mean(np.abs(phi - phi_star))))
        pear = (
            float(np.clip(np.corrcoef(phi, phi_star)[0, 1], -1, 1))
            if (np.std(phi) > 1e-10 and np.std(phi_star) > 1e-10)
            else 0.0
        )
        ec3s.append(1.0 - pear)
    print(f"{name:30s} {np.mean(ec1s):8.4f} {np.mean(ec3s):8.4f}")

print(
    "\nKS-Zero matches its (marginal) target almost exactly by construction;\n"
    "KS-Oracle carries finite-sample completion noise against its exact\n"
    "conditional target. Swap in your own imputer (any BaseImputer) or your\n"
    "own player set (any PlayerSet) to evaluate them the same way."
)
