"""
loss.py
-------
Hybrid Temporal-Spectral Reward (HTSR) Loss (paper Sec. III-I).

Paper formulation:
  L_time        = MSE(ŷ, y)                                   (eq.8)
  L_time_reward = L_time + β·log(1 − e^{−L_time} + ε)         (eq.9)

  L_freq        = L1(|DFT(ŷ)|, |DFT(y)|)                     (eq.10)
  L_freq_reward = L_freq + β·log(1 − e^{−L_freq} + ε)         (eq.11)

  L_total = α·L_freq_reward + (1−α)·L_time_reward              (eq.12)

  Default: α=0.5, β=0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import ALPHA, BETA, EPSILON, JOINT_LOSS_WEIGHTS
except ImportError:
    from config import ALPHA, BETA, EPSILON, JOINT_LOSS_WEIGHTS


class HTSRLoss(nn.Module):
    """
    Hybrid Temporal-Spectral Reward Loss.

    Args:
        alpha (float): Weight for frequency-domain loss  (default 0.5)
        beta  (float): Reward strength                   (default 0.1)
        eps   (float): Numerical stability term          (default 1e-8)
        joint_weights (list[float] | None): per-joint weights applied to the
            time-domain MSE. None → uniform (paper default). v3 default uses
            JOINT_LOSS_WEIGHTS from config to up-weight the knees, which had
            the worst per-joint MAE in the master baseline.
    """

    def __init__(self,
                 alpha: float = ALPHA,
                 beta:  float = BETA,
                 eps:   float = EPSILON,
                 joint_weights = JOINT_LOSS_WEIGHTS):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        if joint_weights is None:
            self.register_buffer("joint_w", None)
        else:
            w = torch.tensor(joint_weights, dtype=torch.float32)
            w = w * (len(w) / w.sum())   # normalise to mean = 1 (same scale as uniform)
            self.register_buffer("joint_w", w)

    def _reward(self, loss_val: torch.Tensor) -> torch.Tensor:
        """
        Reward term: L + β·log(1 − e^{−L} + ε)
        For large L: log(1 − e^{-L}) → 0  (no additional penalty)
        For small L: log(1 − e^{-L}) → −∞ (tempered by β, encourages well-predicted samples)
        """
        L     = loss_val.clamp(min=1e-12)
        inner = (1.0 - torch.exp(-L) + self.eps).clamp(min=self.eps)
        return L + self.beta * torch.log(inner)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            y_pred: [B, dj]
            y_true: [B, dj]
        Returns:
            Scalar total loss.
        """
        # ── Time-domain (eq.8-9) ─────────────────────────────────────────
        # v3 change: per-joint-weighted MSE (knees up-weighted). Normalised
        # weights mean overall scale matches the uniform paper version.
        if self.joint_w is not None:
            sq_err = (y_pred - y_true) ** 2                    # [B, dj]
            # Defensive device match: HTSRLoss is sometimes instantiated without
            # ever being `.to(device)`'d (the original training script does that),
            # so move the weight buffer to the input device lazily.
            w = self.joint_w
            if w.device != y_pred.device:
                w = w.to(y_pred.device)
                self.joint_w = w
            L_time = (sq_err * w).mean()
        else:
            L_time = F.mse_loss(y_pred, y_true)
        L_time_r = self._reward(L_time)

        # ── Frequency-domain (eq.10-11) ───────────────────────────────────
        # DFT applied over the joint dimension (dim=1, length dJ=6).
        # rfft returns dJ//2 + 1 = 4 unique complex frequency bins.
        # v3-speed fix: cuFFT requires power-of-2 sizes in fp16, and dJ=6 isn't
        # one. Cast to fp32 for the FFT — cost is negligible (6 elements).
        Y_hat_freq = torch.fft.rfft(y_pred.float(), dim=1)
        Y_freq     = torch.fft.rfft(y_true.float(), dim=1)
        L_freq     = F.l1_loss(Y_hat_freq.abs(), Y_freq.abs())
        L_freq_r   = self._reward(L_freq)

        # ── Total (eq.12) ─────────────────────────────────────────────────
        return self.alpha * L_freq_r + (1.0 - self.alpha) * L_time_r
