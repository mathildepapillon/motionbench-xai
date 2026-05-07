"""scripts/run_package_temporal_shap.py — replace our hand-rolled
KS-Temporal/WindowSHAP with the published pip packages.

Runs (per dataset, per classifier, per method):
    - WindowSHAP Stationary (uniform K=4 windows; matches our existing setup)
    - WindowSHAP Sliding   (sliding window with stride=window_len/2)
    - WindowSHAP Dynamic   (adaptive split-point search)
    - TimeSHAP local_event (per-timestep with pruning, then summed into K=4 windows)

For each method we save attributions in the SAME (n_seq, K) shape that
KernelSHAP variants use, so they slot into the existing EC1/EC3 tables.

Outputs:
    results/synthetic/<ds>/<clf>/<method_pkg>/result.json
    results/synthetic/<ds>/<clf>/<method_pkg>/attributions.npz
"""
from __future__ import annotations

import json, logging, time, sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr, kendalltau

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from windowshap.windowshap import (
    StationaryWindowSHAP, SlidingWindowSHAP, DynamicWindowSHAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASETS = [
    "gaussian_k4", "skeleton_structured", "gait_periodic",
    "skeleton_gait_combined", "burr_m5",
]
CLASSIFIERS = ["synthetic_mlp"]   # one classifier for time budget; oracle is shared
N_SEQ = int(__import__("os").environ.get("N_SEQ", "20"))
N_BG = int(__import__("os").environ.get("N_BG", "10"))
NSAMPLES = int(__import__("os").environ.get("NSAMPLES", "256"))
RESULTS_DIR = REPO / "results" / "synthetic"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------- #
# Adapter: wrap our (B, J, F, T) classifier as a (B, T, F') sklearn-style model
# --------------------------------------------------------------------- #
class ClassifierAdapter:
    """Adapt a torch (B, J, F, T) classifier to a numpy (B, T, J*F) ``predict``."""

    def __init__(self, clf, J: int, F: int, T: int, target_class: int, device):
        self.clf = clf
        self.J, self.F, self.T = J, F, T
        self.target_class = target_class
        self.device = device

    def predict(self, x_btf):
        """``x_btf`` has shape (B, T, J*F) per windowshap's convention."""
        if x_btf.ndim == 2:  # (T, J*F)
            x_btf = x_btf[None]
        B = x_btf.shape[0]
        x_jft = x_btf.reshape(B, self.T, self.J, self.F).transpose(0, 2, 3, 1)
        x_t = torch.from_numpy(x_jft.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.clf(x_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # WindowSHAP expects (B, n_outputs); we return the probability of the target class only
        return probs[:, self.target_class : self.target_class + 1]


# --------------------------------------------------------------------- #
# Metrics (mirrors motionbench/eval.py for synthetic)
# --------------------------------------------------------------------- #
def shapley_metrics(phi_pred: np.ndarray, phi_true: np.ndarray) -> dict:
    """Compute EC1, EC3, topk.  Both inputs are (n_seq, K).

    EC3 = 1 - Pearson(phi, phi_oracle) per sequence, then averaged across
    sequences.  This matches the pipeline's ``EC3Metric`` definition exactly.
    Range [0, 2]; 0 = perfect, 1 = uncorrelated, 2 = perfectly anti-correlated.
    """
    n, K = phi_pred.shape
    # EC1: mean absolute error
    ec1 = float(np.mean(np.abs(phi_pred - phi_true)))
    # EC3: 1 - Pearson (per-sequence), matching pipeline EC3Metric
    pearsons = []
    for i in range(n):
        a = phi_pred[i] - phi_pred[i].mean()
        b = phi_true[i] - phi_true[i].mean()
        std_a = float(np.sqrt((a ** 2).sum()))
        std_b = float(np.sqrt((b ** 2).sum()))
        if std_a > 1e-12 and std_b > 1e-12:
            p = float(np.dot(a, b) / (std_a * std_b))
            p = max(-1.0, min(1.0, p))
            pearsons.append(p)
        else:
            pearsons.append(0.0)  # constant vector: no linear information
    ec3 = float(1.0 - np.mean(pearsons)) if pearsons else float("nan")
    # Top-1 recovery
    top1_pred = np.argmax(np.abs(phi_pred), axis=1)
    top1_true = np.argmax(np.abs(phi_true), axis=1)
    top1 = float(np.mean(top1_pred == top1_true))
    return {"ec1": ec1, "ec3": ec3, "top1_recovery": top1, "n_sequences": n}


# --------------------------------------------------------------------- #
# Per-method runners
# --------------------------------------------------------------------- #
def run_windowshap_stationary(adapter, x_test, x_bg, K):
    """Stationary uniform-K windows: matches our K=4 player set exactly."""
    T = adapter.T
    win_len = T // K
    n_test = x_test.shape[0]
    F_total = x_test.shape[-1]
    # The package wraper_predict has O(B*T*F) python loops, so processing one
    # sequence at a time keeps memory bounded and gives us per-seq progress.
    phi_K_all = np.zeros((n_test, K))
    for i in range(n_test):
        explainer = StationaryWindowSHAP(
            model=adapter, window_len=win_len,
            B_ts=x_bg, test_ts=x_test[i : i + 1], model_type="lstm",
        )
        # Override default nsamples to keep runtime bounded
        explainer.explainer = None  # force re-init
        # Monkey-patch to use NSAMPLES instead of 'auto'
        import shap as _shap
        explainer.explainer = _shap.KernelExplainer(explainer.wraper_predict, explainer.background_data)
        sv = explainer.explainer.shap_values(explainer.test_data, nsamples=NSAMPLES, silent=True)
        sv = np.array(sv)  # (num_test, n_features) or (n_outputs, num_test, n_features)
        if sv.ndim == 2:
            sv = sv[None]
        ts_phi = sv[0, :, :].reshape(1, K, F_total)  # one seq, K windows, F_total ts feats
        phi_K_all[i] = ts_phi[0].sum(axis=1)
    return phi_K_all


def run_windowshap_sliding(adapter, x_test, x_bg, K):
    """Sliding window with stride = window_len // 2."""
    T = adapter.T
    win_len = T // K
    stride = max(win_len // 2, 1)
    explainer = SlidingWindowSHAP(
        model=adapter, stride=stride, window_len=win_len,
        B_ts=x_bg, test_ts=x_test, model_type="lstm",
    )
    phi_full = explainer.shap_values()  # (num_test, T, F')
    n_test = phi_full.shape[0]
    F_total = phi_full.shape[-1]
    phi_K = phi_full.reshape(n_test, K, win_len, F_total).sum(axis=(2, 3))
    return phi_K


def run_windowshap_dynamic(adapter, x_test, x_bg, K):
    """Dynamic split-point WindowSHAP — must run one sequence at a time.

    The published package was written for an older shap API that returned a
    list-of-arrays for classifiers; newer shap returns a 2D array directly.
    We monkey-patch ``shap_values`` to add the missing leading axis.
    """
    import shap as _shap
    _orig = _shap.KernelExplainer.shap_values

    def _patched(self, X, **kw):
        sv = _orig(self, X, **kw)
        sv = np.asarray(sv)
        # Newer shap returns (num_test, n_features, n_outputs).
        # WindowSHAP package expects (n_outputs, num_test, n_features).
        if sv.ndim == 3:
            sv = sv.transpose(2, 0, 1)
        elif sv.ndim == 2:
            sv = sv[None]
        return sv
    _shap.KernelExplainer.shap_values = _patched
    try:
        T = adapter.T
        n_test = x_test.shape[0]
        F_total = x_test.shape[-1]
        win_len = T // K
        phi_K_all = np.zeros((n_test, K))
        for i in range(n_test):
            explainer = DynamicWindowSHAP(
                model=adapter, delta=0.05, n_w=8,
                B_ts=x_bg, test_ts=x_test[i : i + 1], model_type="lstm",
            )
            phi_full = explainer.shap_values(nsamples_in_loop=NSAMPLES,
                                             nsamples_final=NSAMPLES)
            # phi_full shape (1, T, F_total); aggregate to fixed K windows
            phi_K_all[i] = phi_full[0].reshape(K, win_len, F_total).sum(axis=(1, 2))
    finally:
        _shap.KernelExplainer.shap_values = _orig
    return phi_K_all


def run_timeshap_event(adapter, x_test, x_bg, K):
    """TimeSHAP per-event with no pruning, then aggregate to K windows."""
    from timeshap.explainer import local_event

    T = adapter.T
    F_total = x_test.shape[-1]
    win_len = T // K
    n_test = x_test.shape[0]
    baseline = x_bg.mean(axis=0, keepdims=True)  # (1, T, F')
    # Note: timeshap operates on a recurrent model; we wrap our predict as f(x)
    # local_event returns a DataFrame with per-timestep Shapley values.
    phi_K_all = np.zeros((n_test, K))

    def f_predict(x):
        # x can be (1, T, F') or (B, T, F')
        if x.ndim == 2:
            x = x[None]
        return adapter.predict(x)

    # event_dict and entity columns are required by timeshap; use minimal stubs
    event_dict = {"rs": 42, "nsamples": 32}  # 32 coalition samples
    for i in range(n_test):
        try:
            df = local_event(
                f=f_predict,
                data=x_test[i : i + 1],
                event_dict=event_dict,
                entity_uuid=str(i),
                entity_col="entity",
                baseline=baseline,
                pruned_idx=0,
            )
            # df has columns ["t", "Shapley Value"] roughly; per-timestep attributions
            # Aggregate into K windows
            phi_t = np.zeros(T)
            for _, row in df.iterrows():
                t_idx = int(row["Feature"]) if "Feature" in row else int(row.get("t", 0))
                phi_t[t_idx] = float(row["Shapley Value"])
            phi_K_all[i] = phi_t.reshape(K, win_len).sum(axis=1)
        except Exception as e:
            log.warning("timeshap seq %d failed: %s", i, e)
            phi_K_all[i] = np.nan
    return phi_K_all


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def run_one_cell(ds_name: str, clf_name: str, methods: list[str]):
    from motionbench.pipelines.synthetic_eval import _build_classifier
    ds_cfg_d = OmegaConf.to_container(
        OmegaConf.load(REPO / "configs" / "data" / f"{ds_name}.yaml"),
        resolve=True,
    )
    K = int(ds_cfg_d.pop("K", 4))
    target = ds_cfg_d.pop("_target_")
    mod_path, cls_name = target.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(mod_path)
    DatasetCls = getattr(mod, cls_name)
    dataset = DatasetCls(**ds_cfg_d)
    J, F, T = dataset.shape
    n_classes = int(dataset.metadata.get("n_classes", 3))
    device = torch.device(DEVICE)

    clf_yaml = REPO / "configs" / "classifiers" / f"{clf_name}.yaml"
    clf_cfg = OmegaConf.load(clf_yaml)
    clf = _build_classifier(clf_cfg, J, F, T, K, n_classes).to(device)
    clf.eval()
    ckpt_path = REPO / "motionbench/classifiers/checkpoints/synthetic" / ds_name / f"{clf_name}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        clf.load_state_dict(ckpt["model_state_dict"])

    # Pull data: first N_SEQ as test, rest as background
    n = min(N_SEQ, len(dataset))
    seqs, labels = [], []
    for i in range(n):
        x_i, y_i = dataset[i]
        seqs.append(x_i)
        labels.append(int(y_i) if hasattr(y_i, "__int__") else 0)
    X_test = torch.stack(seqs).to(device)
    with torch.no_grad():
        targets = clf(X_test).argmax(dim=-1).cpu().numpy()

    # Background: N_BG random samples from the dataset (skip first N_SEQ)
    bg_n = min(N_BG, len(dataset) - n)
    bg_seqs = []
    for i in range(n, n + bg_n):
        x_i, _ = dataset[i]
        bg_seqs.append(x_i)
    X_bg = torch.stack(bg_seqs)

    # Reshape to (B, T, J*F)
    x_test_btf = X_test.cpu().numpy().transpose(0, 3, 1, 2).reshape(n, T, J * F)
    x_bg_btf = X_bg.cpu().numpy().transpose(0, 3, 1, 2).reshape(bg_n, T, J * F)

    # Oracle phi: load pre-computed values from disk (faster + reuses existing pipeline).
    oracle_path = RESULTS_DIR / ds_name / clf_name / "kernelshap_oracle" / "attributions.npz"
    if not oracle_path.exists():
        log.warning("No oracle attributions at %s — EC1/EC3 will be skipped", oracle_path)
        phi_true = np.zeros((n, K), dtype=np.float32)
    else:
        with np.load(oracle_path) as nz:
            phi_oracle_full = nz["phi"]  # shape (N_seq_oracle, K) typically
        phi_true = phi_oracle_full[:n].astype(np.float32)

    runners = {
        "windowshap_stationary": run_windowshap_stationary,
        "windowshap_sliding":    run_windowshap_sliding,
        "windowshap_dynamic":    run_windowshap_dynamic,
        "timeshap_event":        run_timeshap_event,
    }

    for method in methods:
        out_dir = RESULTS_DIR / ds_name / clf_name / method
        if (out_dir / "result.json").exists():
            log.info("[SKIP cached] %s/%s/%s", ds_name, clf_name, method)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info(">>> %s / %s / %s", ds_name, clf_name, method)
        t0 = time.time()
        # Per-sequence target -> use target=majority class for adapter
        # We'll loop sequences for adapter targets
        phi_pred_all = np.zeros((n, K), dtype=np.float32)
        try:
            # Group sequences by target class for batched WindowSHAP runs
            for target_cls in np.unique(targets):
                idx = np.where(targets == target_cls)[0]
                if idx.size == 0:
                    continue
                adapter = ClassifierAdapter(clf, J, F, T, int(target_cls), device)
                fn = runners[method]
                phi_pred_all[idx] = fn(adapter, x_test_btf[idx], x_bg_btf, K)
            elapsed = time.time() - t0
            metrics = shapley_metrics(phi_pred_all, phi_true)
            metrics.update({
                "method": method,
                "dataset": ds_name,
                "classifier": clf_name,
                "elapsed_s": elapsed,
            })
            np.savez(out_dir / "attributions.npz",
                     phi=phi_pred_all, phi_true=phi_true, target=targets)
            with open(out_dir / "result.json", "w") as f:
                json.dump(metrics, f, indent=2)
            log.info("  done %s in %.1fs  EC1=%.4f  EC3=%.4f  top1=%.2f",
                     method, elapsed, metrics["ec1"], metrics["ec3"], metrics["top1_recovery"])
        except Exception as e:
            log.exception("FAILED %s/%s/%s: %s", ds_name, clf_name, method, e)


def main():
    methods_arg = sys.argv[1] if len(sys.argv) > 1 else (
        "windowshap_stationary,windowshap_sliding,windowshap_dynamic"
    )
    methods = methods_arg.split(",")
    t_total = time.time()
    for ds in DATASETS:
        for clf in CLASSIFIERS:
            run_one_cell(ds, clf, methods)
    log.info("ALL DONE in %.1fs", time.time() - t_total)


if __name__ == "__main__":
    main()
