"""
model.py
--------
EEG2GAIT: Hierarchical Graph Convolutional Network for EEG-Based Gait Decoding.

Architecture (following paper Figure 2 and Section III):

  Input X ∈ R^{B × C × T}
    │
    ├─[LTL] Local Temporal Learner       Conv1D(F=25, k=10) → X_ltl ∈ R^{B×F×C×T}
    │       → reshape to  R^{B×F×C×T}   (actually treated as B×F×C×T)
    │
    ├─[GCM] Graph Construction Module    learnable A ∈ R^{C×C}, normalised Â
    │
    ├─[HGP] Hierarchical GCN Pyramid     3 branches (depth 1,2,3) → concat → H
    │
    ├─[GSL] Global Spatial Learner       residual + depth-wise conv(C,1) + BN + ELU
    │                                    + Dropout + AvgPool(1,3) → T//3
    │
    ├─[FFN] Feature Fusion Layers        3 × (Conv(1,10)+BN+ELU+MaxPool) with 50,100,200
    │                                    filters → T//81
    │
    ├─[GTL] Global Temporal Learner      Multi-head self-attention (residual)
    │
    └─[OUT] Output Layer                 Cat(GTL, input) → Conv(dj, 1, T//81×2) → (B, dj)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    N_CHANNELS, WINDOW_SAMPS,
    F_FILTERS, LTL_KERNEL,
    HGP_DEPTHS,
    GSL_DROPOUT, FFN_DROPOUTS,
    GTL_HEADS, GTL_DIM,
    N_JOINTS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Graph Convolution Layer
# ──────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """
    One layer of GCN:  H' = σ( Â H W )
    where  Â = D^{-1/2} A D^{-1/2}  (symmetric normalisation)
           H ∈ R^{B × F × C × T}   treated as a graph signal with C nodes
           W ∈ R^{F × F}            learnable weight matrix per feature-map

    For efficiency we apply W as a 1×1 conv over the feature dimension.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc      = nn.Linear(in_features, out_features, bias=False)
        self.bn      = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        H     : [B, C, F]   node features (batch, nodes, feat)
        A_hat : [C, C]      normalised adjacency
        Returns [B, C, F']
        """
        # Graph diffusion: H_out[b,i] = Σ_j A_hat[i,j] * H[b,j]
        # bmm: [B, C, C] × [B, C, F] → but A_hat is shared → use einsum
        AH = torch.einsum("ij,bjf->bif", A_hat, H)  # [B, C, F]
        out = self.fc(AH)                             # [B, C, F']
        # BN over the feature dim (need to permute)
        B, C, Fp = out.shape
        out = self.bn(out.view(B * C, Fp)).view(B, C, Fp)
        out = F.elu(out)
        return self.dropout(out)


# ──────────────────────────────────────────────────────────────────────────────
# Graph Construction Module (GCM)
# ──────────────────────────────────────────────────────────────────────────────

class GraphConstructionModule(nn.Module):
    """
    Maintains one *learnable* adjacency matrix A ∈ R^{C×C}.
    Initialised from the distance-based binary matrix.
    Computes  Â = D^{-1/2} A D^{-1/2}  dynamically.
    """

    def __init__(self, A_init: torch.Tensor):
        """
        A_init: [C, C]  binary adjacency matrix (from distance threshold)
        """
        super().__init__()
        self.A = nn.Parameter(A_init.clone().float())

    def forward(self) -> torch.Tensor:
        """Returns normalised  Â [C, C]."""
        A = F.relu(self.A)          # keep non-negative
        # Degree vector
        D = A.sum(dim=1)            # [C]
        # Avoid div-by-zero for isolated nodes
        D_inv_sqrt = torch.where(D > 0,
                                 torch.pow(D + 1e-8, -0.5),
                                 torch.zeros_like(D))
        D_mat = torch.diag(D_inv_sqrt)  # [C, C]
        A_hat = D_mat @ A @ D_mat       # [C, C]
        return A_hat


# ──────────────────────────────────────────────────────────────────────────────
# Single GCN Branch (one depth)
# ──────────────────────────────────────────────────────────────────────────────

class GCNBranch(nn.Module):
    """
    A stack of `depth` GCN layers.
    Each branch has its OWN independent learnable adjacency matrix.
    """

    def __init__(self, in_feat: int, hidden_feat: int,
                 depth: int, A_init: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.gcm    = GraphConstructionModule(A_init)
        layers = []
        for d in range(depth):
            in_f  = in_feat    if d == 0 else hidden_feat
            out_f = hidden_feat
            layers.append(GraphConvLayer(in_f, out_f, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: [B, C, F] → [B, C, F_hidden]"""
        A_hat = self.gcm()
        for layer in self.layers:
            H = layer(H, A_hat)
        return H


# ──────────────────────────────────────────────────────────────────────────────
# Local Temporal Learner (LTL)
# ──────────────────────────────────────────────────────────────────────────────

class LocalTemporalLearner(nn.Module):
    """
    1D temporal convolution per-channel using groups.
    Input  X : [B, C, T]
    Output   : [B, F, C, T]  (F independent temporal filters per channel)

    Implementation note:
    We reshape to [B*C, 1, T] and apply Conv1d with F filters and kernel_size=k.
    Then reshape back to [B, C, F, T] and permute to [B, F, C, T].
    """

    def __init__(self, n_channels: int = N_CHANNELS,
                 n_filters: int = F_FILTERS,
                 kernel_size: int = LTL_KERNEL):
        super().__init__()
        self.n_channels = n_channels
        self.n_filters  = n_filters
        self.conv = nn.Conv1d(
            in_channels  = 1,
            out_channels = n_filters,
            kernel_size  = kernel_size,
            padding      = kernel_size // 2,
            bias         = False
        )
        self.bn  = nn.BatchNorm2d(n_filters)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] → [B, F, C, T]"""
        B, C, T = x.shape
        # process each channel independently
        x_flat  = x.reshape(B * C, 1, T)           # [B*C, 1, T]
        out     = self.conv(x_flat)                 # [B*C, F, T']
        T_out   = out.shape[-1]
        out     = out.reshape(B, C, self.n_filters, T_out)  # [B,C,F,T']
        out     = out.permute(0, 2, 1, 3)           # [B, F, C, T']
        out     = self.bn(out)
        out     = self.act(out)
        return out  # [B, F, C, T']  where T' ≈ T (zero-padded)


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical GCN Pyramid (HGP)
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalGCNPyramid(nn.Module):
    """
    Multiple GCN branches with different depths (1, 2, 3).
    Input:  [B, F, C, T]   (from LTL)
    Applies GCN independently at each time step → outputs [B, depth_branches×F, C, T]

    For efficiency: treat the temporal dimension as batch items.
    Reshape to [B*T, C, F], pass through each branch, then reshape back.
    """

    def __init__(self, in_feat: int, hidden_feat: int,
                 depths: List[int], A_init: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            GCNBranch(in_feat, hidden_feat, d, A_init, dropout)
            for d in depths
        ])
        self.n_branches = len(depths)
        self.hidden_feat = hidden_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F, C, T]
        Returns: [B, n_branches*F, C, T]   (concat across branches)
        """
        B, F, C, T = x.shape
        # reshape: treat each (b, t) as a graph-batch item
        x_perm = x.permute(0, 3, 2, 1)     # [B, T, C, F]
        x_flat = x_perm.reshape(B * T, C, F)  # [B*T, C, F]

        branch_outs = []
        for branch in self.branches:
            out = branch(x_flat)             # [B*T, C, F_hidden]
            out = out.reshape(B, T, C, self.hidden_feat)   # [B,T,C,F_h]
            out = out.permute(0, 3, 2, 1)   # [B, F_h, C, T]
            branch_outs.append(out)

        return torch.cat(branch_outs, dim=1)  # [B, n_branches*F_h, C, T]


# ──────────────────────────────────────────────────────────────────────────────
# Global Spatial Learner (GSL)
# ──────────────────────────────────────────────────────────────────────────────

class GlobalSpatialLearner(nn.Module):
    """
    Combines HGP output with original features via residual connection.

    Pipeline per the paper:
      1. Concat [HGP_out, LTL_out] along filter dim  → [B, F_merged, C, T]
      2. Depth-wise spatial collapse: pool across C   → [B, F_merged, 1, T]
      3. Point-wise projection to out_filters         → [B, out_filters, 1, T]
      4. BatchNorm → ELU → Dropout(0.5)
      5. AvgPool(1,3)                                 → [B, out_filters, 1, T//3]
    """

    def __init__(self, n_channels: int,
                 in_filters: int,    # F_hgp (HGP output filters)
                 orig_filters: int,  # F     (LTL output filters)
                 out_filters: int,
                 dropout: float = 0.5):
        super().__init__()
        merged = in_filters + orig_filters

        # Collapse C spatial positions → 1 via mean, then project
        # (equivalent to paper's grouped spatial conv with uniform weights)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, None))  # [B,F,C,T]→[B,F,1,T]
        self.pointwise    = nn.Conv2d(merged, out_filters, kernel_size=1, bias=False)
        self.bn           = nn.BatchNorm2d(out_filters)
        self.act          = nn.ELU()
        self.drop         = nn.Dropout(dropout)
        self.pool         = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))

    def forward(self, h_hgp: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """
        h_hgp : [B, F_hgp, C, T]
        x_orig: [B, F,     C, T]
        Returns [B, out_filters, 1, T//3]
        """
        h = torch.cat([h_hgp, x_orig], dim=1)   # [B, F_merged, C, T]
        h = self.spatial_pool(h)                 # [B, F_merged, 1, T]
        h = self.pointwise(h)                    # [B, out_filters, 1, T]
        h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.pool(h)                         # [B, out_filters, 1, T//3]
        return h


# ──────────────────────────────────────────────────────────────────────────────
# Feature Fusion Network (FFN)  — three sequential blocks
# ──────────────────────────────────────────────────────────────────────────────

class FeatureFusionNetwork(nn.Module):
    """
    Three sequential blocks: Conv(1,10) → BN → ELU → MaxPool(1,3).
    Filter progression: in → 50 → 100 → 200.
    Dropout(0.5) before each block.
    Input/Output shapes:
        in:   [B, in_filters, 1, T_in]
        out:  [B, 200, 1, T_in//27]  (3 poolings of stride 3 → T//3^3 = T//27)
    """

    def __init__(self, in_filters: int, dropout: float = 0.5):
        super().__init__()
        cfg = [(in_filters, 50), (50, 100), (100, 200)]
        self.blocks = nn.ModuleList()
        self.drops  = nn.ModuleList()
        for in_f, out_f in cfg:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=(1, 10), padding=(0, 5), bias=False),
                nn.BatchNorm2d(out_f),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            ))
            self.drops.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_filters, 1, T] → [B, 200, 1, T//27]"""
        for drop, block in zip(self.drops, self.blocks):
            x = drop(x)
            x = block(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Global Temporal Learner (GTL) — Multi-head Self-Attention
# ──────────────────────────────────────────────────────────────────────────────

class GlobalTemporalLearner(nn.Module):
    """
    Multi-head self-attention over the temporal dimension with residual connection.

    Input:  [B, F, 1, T_small]
    Treats the T_small time-steps as the sequence length and F as feature dim.
    Output: [B, F, 1, T_small]
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(embed_dim, n_heads,
                                             dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, F, 1, T] → [B, F, 1, T]"""
        B, F, _, T = x.shape
        # reshape to sequence: [B, T, F]
        x_seq = x.squeeze(2).permute(0, 2, 1)           # [B, T, F]
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)    # [B, T, F]
        attn_out = self.dropout(attn_out)
        x_res    = self.norm(x_seq + attn_out)           # residual + layer-norm
        # reshape back
        out = x_res.permute(0, 2, 1).unsqueeze(2)       # [B, F, 1, T]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# EEG2GAIT — Full Model
# ──────────────────────────────────────────────────────────────────────────────

class EEG2GAIT(nn.Module):
    """
    Full EEG2GAIT model.

    Input:  X ∈ R^{B × C × T}
                C = 59 channels
                T = 100 (1 sec @ 100 Hz)

    Output: ŷ ∈ R^{B × dj}
                dj = 6 joint angles
    """

    def __init__(self,
                 n_channels:   int   = N_CHANNELS,
                 window_samps: int   = WINDOW_SAMPS,
                 n_joints:     int   = N_JOINTS,
                 n_filters:    int   = F_FILTERS,       # F = 25
                 ltl_kernel:   int   = LTL_KERNEL,
                 hgp_depths:   List  = None,
                 hgp_hidden:   int   = 25,               # hidden per GCN branch
                 gtl_heads:    int   = GTL_HEADS,
                 gsl_out:      int   = 50,
                 A_init:       torch.Tensor = None):
        super().__init__()

        if hgp_depths is None:
            hgp_depths = HGP_DEPTHS   # [1, 2, 3]

        self.n_channels   = n_channels
        self.window_samps = window_samps
        self.n_joints     = n_joints

        # ── Compute temporal sizes ────────────────────────────────────────
        # LTL zero-pads to maintain T: T_ltl = T (with even padding)
        T_ltl = window_samps          # ≈ T (see LTL padding logic)
        T_gsl = T_ltl // 3           # after AvgPool(1,3)
        T_ffn = T_gsl // 27          # after 3× MaxPool(1,3)  (3^3=27)
        self.T_ffn = T_ffn

        # ── Modules ──────────────────────────────────────────────────────
        # 1. LTL
        self.ltl = LocalTemporalLearner(n_channels, n_filters, ltl_kernel)

        # 2. HGP
        if A_init is None:
            A_init = torch.ones(n_channels, n_channels)  # fallback
        F_hgp_out = len(hgp_depths) * hgp_hidden
        self.hgp = HierarchicalGCNPyramid(
            in_feat=n_filters, hidden_feat=hgp_hidden,
            depths=hgp_depths, A_init=A_init, dropout=0.1
        )

        # 3. GSL
        self.gsl = GlobalSpatialLearner(
            n_channels  = n_channels,
            in_filters  = F_hgp_out,
            orig_filters = n_filters,
            out_filters  = gsl_out,
            dropout      = GSL_DROPOUT
        )

        # 4. FFN
        self.ffn = FeatureFusionNetwork(in_filters=gsl_out, dropout=FFN_DROPOUTS)
        # Output of FFN: [B, 200, 1, T_ffn]

        # 5. GTL
        self.gtl = GlobalTemporalLearner(embed_dim=200, n_heads=gtl_heads, dropout=0.1)

        # 6. Output layer
        # Concatenate GTL(out) + GTL(in) along T → [B, 200, 1, T_ffn*2]
        # Apply Conv2d(dj, (1, T_ffn*2)) → [B, dj, 1, 1]
        self.out_conv = nn.Conv2d(
            in_channels  = 200,
            out_channels = n_joints,
            kernel_size  = (1, T_ffn * 2),
            bias         = True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, C, T]
        Returns ŷ: [B, dj]
        """
        B, C, T = x.shape

        # ── 1. LTL ────────────────────────────────────────────────────────
        x_ltl = self.ltl(x)        # [B, F, C, T]
        # Ensure T dimension is maintained
        _, F, _, T_ltl = x_ltl.shape

        # ── 2. HGP ────────────────────────────────────────────────────────
        x_hgp = self.hgp(x_ltl)   # [B, n_branches*F_h, C, T_ltl]

        # ── 3. GSL ────────────────────────────────────────────────────────
        x_gsl = self.gsl(x_hgp, x_ltl)   # [B, gsl_out, 1, T_ltl//3]

        # ── 4. FFN ────────────────────────────────────────────────────────
        x_ffn = self.ffn(x_gsl)           # [B, 200, 1, T//81]

        # ── 5. GTL ────────────────────────────────────────────────────────
        x_gtl = self.gtl(x_ffn)           # [B, 200, 1, T//81]

        # ── 6. Output ─────────────────────────────────────────────────────
        # Concatenate along T dimension
        x_cat = torch.cat([x_gtl, x_ffn], dim=3)   # [B, 200, 1, T//81 * 2]

        # Dynamic out_conv kernel if T_ffn changed from init (safety check)
        T_cat = x_cat.shape[3]
        if self.out_conv.kernel_size[1] != T_cat:
            # Re-init output conv with correct size (handles variable-length inputs)
            self.out_conv = nn.Conv2d(
                200, self.n_joints, kernel_size=(1, T_cat), bias=True
            ).to(x.device)

        out = self.out_conv(x_cat)   # [B, dj, 1, 1]
        out = out.squeeze(-1).squeeze(-1)   # [B, dj]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(A_init: torch.Tensor = None) -> EEG2GAIT:
    """Construct EEG2GAIT with default configuration."""
    if A_init is None:
        # Identity fallback; caller should pass real adjacency
        A_init = torch.eye(N_CHANNELS)
    return EEG2GAIT(A_init=A_init)
