"""
train.py
--------
Training loop for EEG2GAIT.

Features:
  - Adam optimiser (lr=0.001)
  - HTSR custom loss
  - Early stopping: patience=30, monitored metric = mean Pearson r on val set
  - Per-epoch logging: loss + all evaluation metrics
  - Checkpoint saving (best val r model)
  - Final test-set evaluation with per-joint breakdown
"""

import os
import sys
import time
import json
import copy
import random
import argparse
from pathlib import Path

import torch
import numpy as np

# ── Kaggle/PyTorch compatibility fix ────────────────────────────────────────
# Some Kaggle PyTorch builds fail when torch._dynamo lazily imports torch._utils.
# Force-loading torch._utils now prevents that crash.
try:
    import torch._utils   # noqa: F401  – must happen before Adam is used
except Exception:
    pass

# ── Path setup ──────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    DATA_DIR, OUTPUT_DIR, SUBJECTS, SESSIONS,
    BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE,
    SEED, NUM_WORKERS, DEVICE,
    N_CHANNELS, WINDOW_SAMPS, N_JOINTS,
)
from dataset import get_dataloaders, build_adjacency_matrix, _build_standard_positions
from model  import build_model
from loss   import HTSRLoss
from metrics import evaluate_loader, compute_metrics


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Logging helper ───────────────────────────────────────────────────────────

def log(msg: str, log_file=None):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


# ── Main training function ──────────────────────────────────────────────────

def train(subjects=None, sessions=None, device_str=None,
          batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS,
          patience=PATIENCE, output_dir=OUTPUT_DIR):
    """
    Full training pipeline.

    Args:
        subjects:    list of subject IDs (default: all 8)
        sessions:    list of session IDs (default: all 3)
        device_str:  "cpu" | "cuda" | "mps"
        batch_size:  mini-batch size
        lr:          Adam learning rate
        max_epochs:  maximum training epochs
        patience:    early-stopping patience (epochs)
        output_dir:  directory for checkpoints and logs
    """
    set_seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None: subjects = SUBJECTS
    if sessions is None: sessions = SESSIONS

    # ── Device ──────────────────────────────────────────────────────────
    if device_str is None:
        device_str = DEVICE
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  EEG2GAIT Training Run")
    print(f"  Device:   {device}")
    print(f"  Subjects: {subjects}")
    print(f"  Sessions: {sessions}")
    print(f"{'='*60}\n")

    log_path = output_dir / "train_log.txt"
    log_file = open(log_path, "w")

    # ── Data ────────────────────────────────────────────────────────────
    log("Loading data …", log_file)
    train_dl, val_dl, test_dl = get_dataloaders(
        subjects=subjects, sessions=sessions,
        batch_size=batch_size, num_workers=NUM_WORKERS,
        verbose=True
    )
    log(f"Train batches: {len(train_dl)} | "
        f"Val batches: {len(val_dl)} | "
        f"Test batches: {len(test_dl)}", log_file)

    # ── Adjacency matrix (GCM initialization, paper eq.1) ────────────
    log("Building adjacency matrix …", log_file)
    positions = _build_standard_positions()           # [59, 3]
    A_np      = build_adjacency_matrix(positions)     # [59, 59], no self-loops
    A_tensor  = torch.from_numpy(A_np)
    # eq.1: Ã_prior = ReLU(A_prior + A_prior^T) + I
    A_init = torch.relu(A_tensor + A_tensor.T) + torch.eye(A_tensor.size(0))

    # ── Model ────────────────────────────────────────────────────────────
    log("Building model …", log_file)
    model = build_model(A_init=A_init).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable parameters: {n_params:,}", log_file)

    # ── Optimiser & loss ─────────────────────────────────────────────────
    # Use SGD-compatible init path to dodge torch._dynamo import bug on some
    # Kaggle environments; falls back to plain Adam if the bug is already fixed.
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    except AttributeError:
        # _dynamo import failed; rebuild optimizer manually via Optimizer base
        import importlib
        _utils = importlib.import_module("torch._utils")
        torch._utils = _utils                   # patch into torch namespace
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = HTSRLoss()

    # ── Training state ───────────────────────────────────────────────────
    best_val_r    = -1.0
    best_epoch    = 0
    epochs_no_imp = 0
    best_state    = None
    history       = {"train_loss": [], "val_r": [], "val_r2": [], "val_mae": []}

    # ── Epoch loop ───────────────────────────────────────────────────────
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss  = criterion(y_hat, y)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_dl), 1)

        # ── Validate ─────────────────────────────────────────────────────
        val_metrics = evaluate_loader(model, val_dl, device_str)
        val_r       = val_metrics["r_mean"]
        val_r2      = val_metrics["r2_mean"]
        val_mae     = val_metrics["mae_mean"]

        elapsed = time.time() - t0
        msg = (f"Epoch {epoch:03d}/{max_epochs} | "
               f"Loss: {avg_train_loss:.4f} | "
               f"Val r: {val_r:.4f} | "
               f"Val R²: {val_r2:.4f} | "
               f"Val MAE: {val_mae:.4f} | "
               f"Time: {elapsed:.1f}s")
        log(msg, log_file)

        # Track history
        history["train_loss"].append(avg_train_loss)
        history["val_r"].append(val_r)
        history["val_r2"].append(val_r2)
        history["val_mae"].append(val_mae)

        # ── Early stopping ───────────────────────────────────────────────
        if val_r > best_val_r:
            best_val_r    = val_r
            best_epoch    = epoch
            epochs_no_imp = 0
            best_state    = copy.deepcopy(model.state_dict())
            ckpt_path     = output_dir / "best_model.pt"
            torch.save({
                "epoch":      epoch,
                "state_dict": best_state,
                "val_r":      best_val_r,
                "val_r2":     val_r2,
                "val_mae":    val_mae,
            }, ckpt_path)
            log(f"  ✓ New best model saved (val r={best_val_r:.4f})", log_file)
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= patience:
            log(f"\n  Early stopping triggered after {epoch} epochs "
                f"(best epoch={best_epoch}, best val r={best_val_r:.4f})", log_file)
            break

    # ── Load best model ──────────────────────────────────────────────────
    log(f"\nLoading best model (epoch {best_epoch}) …", log_file)
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test evaluation ───────────────────────────────────────────────────
    log("\nEvaluating on TEST set …", log_file)
    test_metrics = evaluate_loader(model, test_dl, device_str)

    log("\n" + "="*60, log_file)
    log("TEST RESULTS", log_file)
    log("="*60, log_file)
    log(f"  Mean Pearson r : {test_metrics['r_mean']:.4f}", log_file)
    log(f"  Mean R²        : {test_metrics['r2_mean']:.4f}", log_file)
    log(f"  Mean MAE       : {test_metrics['mae_mean']:.4f}", log_file)
    log("-"*60, log_file)
    log("  Per-joint breakdown:", log_file)
    from config import JOINT_NAMES
    for name in JOINT_NAMES:
        log(f"    {name:5s}  r={test_metrics[f'r_{name}']:.4f}  "
            f"R²={test_metrics[f'r2_{name}']:.4f}  "
            f"MAE={test_metrics[f'mae_{name}']:.4f}", log_file)
    log("="*60, log_file)

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "best_epoch":    best_epoch,
        "best_val_r":    best_val_r,
        "test_metrics":  test_metrics,
        "history":       history,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {results_path}", log_file)

    log_file.close()
    return model, results


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train EEG2GAIT on the MoBI dataset")
    p.add_argument("--subjects",    nargs="+", default=None,
                   help="Subject IDs to include (e.g. SL01 SL02). Default: all 8.")
    p.add_argument("--sessions",    nargs="+", default=None,
                   help="Session IDs (e.g. T01 T02). Default: all 3.")
    p.add_argument("--device",      default="auto",
                   help="Device: cpu | cuda | mps | auto  (default: auto)")
    p.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--max-epochs",  type=int, default=MAX_EPOCHS)
    p.add_argument("--patience",    type=int, default=PATIENCE)
    p.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        subjects   = args.subjects,
        sessions   = args.sessions,
        device_str = args.device,
        batch_size = args.batch_size,
        lr         = args.lr,
        max_epochs = args.max_epochs,
        patience   = args.patience,
        output_dir = args.output_dir,
    )
