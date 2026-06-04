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
    from .config import ALPHA, BETA, EPSILON
except ImportError:
    from config import ALPHA, BETA, EPSILON


class HTSRLoss(nn.Module):
    """
    Hybrid Temporal-Spectral Reward Loss.

    Args:
        alpha (float): Weight for frequency-domain loss  (default 0.5)
        beta  (float): Reward strength                   (default 0.1)
        eps   (float): Numerical stability term          (default 1e-8)
    """

    def __init__(self,
                 alpha: float = ALPHA,
                 beta:  float = BETA,
                 eps:   float = EPSILON):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

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
        L_time   = F.mse_loss(y_pred, y_true)
        L_time_r = self._reward(L_time)

        # ── Frequency-domain (eq.10-11) ───────────────────────────────────
        # DFT applied over the joint dimension (dim=1, length dJ=6).
        # rfft returns dJ//2 + 1 = 4 unique complex frequency bins.
        Y_hat_freq = torch.fft.rfft(y_pred, dim=1)
        Y_freq     = torch.fft.rfft(y_true, dim=1)
        L_freq     = F.l1_loss(Y_hat_freq.abs(), Y_freq.abs())
        L_freq_r   = self._reward(L_freq)

        # ── Total (eq.12) ─────────────────────────────────────────────────
        return self.alpha * L_freq_r + (1.0 - self.alpha) * L_time_r
