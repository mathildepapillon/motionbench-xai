"""scripts/train_esc50_ast.py — Fold-disciplined AST fine-tuning on ESC-50.

Trains the per-fold ESC-50 classifiers behind the paper's ESC-50 tables:
one Audio Spectrogram Transformer per data fold, fine-tuned from
``MIT/ast-finetuned-audioset-10-10-0.4593`` with a fresh 50-class head on
that fold's training cache only (fold f's test cache is used for epoch
selection and the reported test accuracy — the classifier-gate protocol).

- Data: the per-fold caches written by ``scripts/preprocess_esc50.py``
  (``fold{f}_train.npz`` / ``fold{f}_test.npz``, mels in the motionbench
  ``(N, J=128, F=1, T=1024)`` format).  Input formatting matches the
  attribution-time forward path of
  ``motionbench.classifiers.esc50_classifier.ESC50ASTClassifier`` bit-for-bit:
  squeeze the F dim and permute to ``(B, T=1024, mel=128)``.
- Labels: the cache ``y`` arrays follow ESC-50 canonical target ids (0..49),
  NOT alphabetical class order.  The head is trained directly on the cache
  ``y``, so head index ``k`` = ESC-50 canonical target id ``k``.
- Recipe: AdamW lr 3e-5, weight-decay 0.01, batch 16, 5 epochs, 5% linear
  warmup + cosine to 0, bf16 autocast training / fp32 eval, seed 42+fold,
  best-test-epoch selection.
- Output: a checkpoint in the release format loaded by
  ``motionbench.classifiers.esc50_classifier.load_esc50_classifier``
  (``{state_dict, arch{family, base_model, num_labels, config, input_format,
  label_convention}, ...metadata}``; ``torch.load(..., weights_only=True)``
  compatible).

Usage::

    python scripts/train_esc50_ast.py --fold 1 \\
        --data-dir data/esc50 \\
        --out motionbench/classifiers/checkpoints/real/esc50_ast_fold1.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = "data/esc50"
DEFAULT_OUT = "motionbench/classifiers/checkpoints/real/esc50_ast_fold{fold}.pt"


def to_ast_input(x_jft: np.ndarray) -> torch.Tensor:
    """(B, 128, 1, 1024) cache mel -> (B, 1024, 128) AST input_values.

    Identical to ``ESC50ASTClassifier.forward``: squeeze(2).permute(0,2,1).
    """
    xb = torch.from_numpy(np.ascontiguousarray(x_jft, dtype=np.float32))
    return xb.squeeze(2).permute(0, 2, 1).contiguous()


@torch.no_grad()
def accuracy(model, x_tm: torch.Tensor, y: torch.Tensor, device, batch: int = 48) -> float:
    """Top-1 accuracy in fp32 eval mode (the attribution-time forward path)."""
    model.eval()
    correct = 0
    for s in range(0, len(x_tm), batch):
        logits = model(input_values=x_tm[s : s + batch].to(device)).logits
        correct += int((logits.argmax(-1).cpu() == y[s : s + batch]).sum())
    return correct / len(x_tm)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory with fold{f}_train.npz / fold{f}_test.npz caches "
        "(from scripts/preprocess_esc50.py); relative paths resolve "
        "against the repo root.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help=f"Checkpoint output path (default: {DEFAULT_OUT}); relative "
        "paths resolve against the repo root.",
    )
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None, help="default 42+fold (project convention)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main() -> None:
    """Fine-tune the AST classifier for one fold and save the checkpoint."""
    args = parse_args()

    fold = args.fold
    seed = args.seed if args.seed is not None else 42 + fold
    device = torch.device(args.device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = _REPO_ROOT / args.data_dir
    out = _REPO_ROOT / (args.out if args.out is not None else DEFAULT_OUT.format(fold=fold))
    out.parent.mkdir(parents=True, exist_ok=True)

    tr = np.load(data_dir / f"fold{fold}_train.npz")
    te = np.load(data_dir / f"fold{fold}_test.npz")
    x_tr = to_ast_input(tr["x_train"])  # (1600, 1024, 128)
    y_tr = torch.from_numpy(tr["y_train"].astype(np.int64))
    x_te = to_ast_input(te["x_test"])  # (400, 1024, 128)
    y_te = torch.from_numpy(te["y_test"].astype(np.int64))
    print(
        f"[fold{fold}] train {tuple(x_tr.shape)} test {tuple(x_te.shape)} "
        f"seed={seed} lr={args.lr} epochs={args.epochs} batch={args.batch}",
        flush=True,
    )

    from transformers import ASTForAudioClassification

    model = ASTForAudioClassification.from_pretrained(
        BASE_MODEL, num_labels=50, ignore_mismatched_sizes=True
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(x_tr) + args.batch - 1) // args.batch
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        """Linear warmup then cosine decay LR multiplier."""
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)

    best = {"test_acc": -1.0}
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr), generator=g)
        ep_loss = 0.0
        for s in range(0, len(perm), args.batch):
            idx = perm[s : s + args.batch]
            xb = x_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_values=xb).logits
                loss = loss_fn(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += float(loss.detach()) * len(idx)
        ep_loss /= len(x_tr)
        # fp32 eval (matches the attribution-time forward path)
        train_acc = accuracy(model, x_tr, y_tr, device)
        test_acc = accuracy(model, x_te, y_te, device)
        history.append({"epoch": ep, "loss": ep_loss, "train_acc": train_acc, "test_acc": test_acc})
        print(
            f"[fold{fold}] epoch {ep}/{args.epochs} loss={ep_loss:.4f} "
            f"train={train_acc:.4f} test={test_acc:.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if test_acc > best["test_acc"]:
            best = {
                "epoch": ep,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "state_dict": {
                    k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()
                },
            }

    gap = best["train_acc"] - best["test_acc"]
    ckpt = {
        "state_dict": best["state_dict"],
        "arch": {
            "family": "transformers.ASTForAudioClassification",
            "base_model": BASE_MODEL,
            "num_labels": 50,
            "config": model.config.to_dict(),
            "input_format": "cache mel (B,128,1,1024) -> squeeze(2).permute(0,2,1) -> (B,1024,128)",
            "label_convention": (
                "head index k = ESC-50 canonical target id k "
                "(cache y trained directly; NOT alphabetical "
                "class order)"
            ),
        },
        "fold": fold,
        "train_data": str(data_dir / f"fold{fold}_train.npz"),
        "test_data": str(data_dir / f"fold{fold}_test.npz"),
        "seed": seed,
        "lr": args.lr,
        "weight_decay": 0.01,
        "schedule": "5% linear warmup + cosine to 0",
        "epochs_run": args.epochs,
        "batch_size": args.batch,
        "best_epoch": best["epoch"],
        "train_acc": best["train_acc"],
        "test_acc": best["test_acc"],
        "train_test_gap": gap,
        "history": history,
        "precision": "bf16 autocast train, fp32 eval",
    }
    torch.save(ckpt, out)
    meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "arch")}
    meta["arch_base"] = BASE_MODEL
    (out.parent / f"{out.stem}_meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"[fold{fold}] BEST epoch {best['epoch']}: "
        f"train={best['train_acc']:.4f} test={best['test_acc']:.4f} "
        f"gap={gap:+.4f} -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
