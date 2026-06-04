"""
metrics.py
----------
Evaluation metrics for EEG2GAIT gait angle prediction.

Metrics (per-joint and averaged):
  - Pearson correlation coefficient (r)
  - R² score (coefficient of determination)
  - Mean Absolute Error (MAE)
"""

import torch
import numpy as np
from typing import Dict, Tuple

try:
    from .config import JOINT_NAMES, N_JOINTS
except ImportError:
    from config import JOINT_NAMES, N_JOINTS


def pearson_r(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Pearson correlation coefficient per joint column.

    Args:
        y_pred: [N, J]
        y_true: [N, J]
    Returns:
        r: [J]
    """
    r_vals = []
    for j in range(y_pred.shape[1]):
        p = y_pred[:, j]
        t = y_true[:, j]
        # Avoid NaN for constant predictions
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            r_vals.append(0.0)
        else:
            corr = np.corrcoef(p, t)[0, 1]
            r_vals.append(float(corr) if not np.isnan(corr) else 0.0)
    return np.array(r_vals)


def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    R² (coefficient of determination) per joint.

    Args:
        y_pred: [N, J]
        y_true: [N, J]
    Returns:
        r2: [J]
    """
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    # Avoid division by zero
    r2 = np.where(ss_tot < 1e-12, 0.0, 1.0 - ss_res / ss_tot)
    return r2


def mae_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Error per joint.

    Args:
        y_pred: [N, J]
        y_true: [N, J]
    Returns:
        mae: [J]
    """
    return np.abs(y_pred - y_true).mean(axis=0)


def compute_metrics(y_pred: np.ndarray,
                    y_true: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics and return as a flat dict.

    Keys: per-joint (e.g. "r_GHR") and averaged ("r_mean", "r2_mean", "mae_mean").
    """
    r   = pearson_r(y_pred, y_true)
    r2  = r2_score (y_pred, y_true)
    mae = mae_score(y_pred, y_true)

    results: Dict[str, float] = {}
    for j, name in enumerate(JOINT_NAMES):
        results[f"r_{name}"]   = float(r[j])
        results[f"r2_{name}"]  = float(r2[j])
        results[f"mae_{name}"] = float(mae[j])

    results["r_mean"]   = float(r.mean())
    results["r2_mean"]  = float(r2.mean())
    results["mae_mean"] = float(mae.mean())
    return results


def evaluate_loader(model, loader, device: str) -> Dict[str, float]:
    """
    Run inference over the entire DataLoader and compute metrics.

    Returns:
        Dict of metric names → values.
    """
    import torch
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y_hat = model(X)
            preds.append(y_hat.cpu().numpy())
            targets.append(y.numpy())

    y_pred = np.concatenate(preds,   axis=0)  # [N, J]
    y_true = np.concatenate(targets, axis=0)  # [N, J]
    return compute_metrics(y_pred, y_true)
