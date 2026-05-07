"""One-shot script: run the overnight sweep for windowshap only."""
import warnings

warnings.filterwarnings("ignore")

from omegaconf import OmegaConf  # noqa: E402

from motionbench.pipelines.synthetic_eval import run_synthetic_eval  # noqa: E402

cfg = OmegaConf.load("configs/experiments/overnight_synthetic_sweep.yaml")
OmegaConf.update(cfg, "methods", ["windowshap"])
df = run_synthetic_eval(cfg)
print(f"WindowSHAP sweep complete. {len(df)} cells.")
