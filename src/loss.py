"""
loss.py
-------
Hybrid Temporal-Spectral Reward (HTSR) Loss.

Paper formulation:
  L_time        = MSE(ŷ, y)
  L_time_reward = L_time + β·log(1 − e^{−L_time} + ε)

  L_freq        = L1(DFT(ŷ), DFT(y))   [over complex magnitudes]
  L_freq_reward = L_freq + β·log(1 − e^{−L_freq} + ε)

  L_total = α·L_freq_reward + (1−α)·L_time_reward

  Default: α=0.5, β=0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
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
        For large L: log(1 − e^{-L}) → 0  (reward ≈ loss, no penalty)
        For small L: log(1 − e^{-L}) → −∞ (but tempered by β)
        """
        # Clamp to avoid exp underflow/overflow
        L     = loss_val.clamp(min=1e-12)
        inner = 1.0 - torch.exp(-L) + self.eps
        # Guard log argument
        inner = inner.clamp(min=self.eps)
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
        # ── Time-domain ───────────────────────────────────────────────────
        L_time = F.mse_loss(y_pred, y_true)
        L_time_r = self._reward(L_time)

        # ── Frequency-domain ──────────────────────────────────────────────
        # Compute DFT along joint dimension (or sample-dim if we had sequences)
        # Since predictions are [B, dj] scalars, we apply rfft over the batch
        # dimension (treating batch as a "time" axis) to capture spectral structure
        # across the batch.  Alternatively, if shape is [B, dj], we treat dj as
        # the signal axis.
        # The paper applies DFT to the predicted and true waveforms. Here each
        # sample is already a scalar (mean of window). We use rfft over the dj axis.
        Y_pred_fft = torch.fft.rfft(y_pred, dim=1)   # [B, dj//2+1] complex
        Y_true_fft = torch.fft.rfft(y_true, dim=1)

        # L1 over magnitudes
        L_freq = F.l1_loss(Y_pred_fft.abs(), Y_true_fft.abs())
        L_freq_r = self._reward(L_freq)

        # ── Total ──────────────────────────────────────────────────────────
        L_total = self.alpha * L_freq_r + (1.0 - self.alpha) * L_time_r
        return L_total
