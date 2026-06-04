"""
model.py
--------
EEG2GAIT: A Hierarchical Graph Convolutional Network for EEG-Based Gait Decoding.

Paper architecture (Section III, Figure 2):

  1. LTL  — Local Temporal Learner (temporal Conv2d)
  2. GCM  — Graph Construction Module (learnable adjacency, one per HGP branch)
  3. HGP  — Hierarchical GCN Pyramid (K=2 branches, each with own GCM)
             + residual: cat(HGP_out, LTL_out) before GSL
  4. GSL  — Global Spatial Learner (depth-wise conv + BN + ELU + Dropout + AvgPool)
  5. FFN  — Feature Fusion Network (3 conv blocks [50,100,200] + pooling)
  6. GTL  — Global Temporal Learner (multi-head self-attention)
  7. OUT  — Task-specific output layer: cat(GTL_in, GTL_out) → weight-constrained conv
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import (
        N_CHANNELS, WINDOW_SAMPS, N_JOINTS,
        F_FILTERS, LTL_KERNEL, HGP_DEPTHS,
        DROPOUT_P, POOL_WIDTH, KERNEL_WIDTH,
        FFN_FILTERS, GTL_HEADS, GTL_DROPOUT,
    )
except ImportError:
    from config import (
        N_CHANNELS, WINDOW_SAMPS, N_JOINTS,
        F_FILTERS, LTL_KERNEL, HGP_DEPTHS,
        DROPOUT_P, POOL_WIDTH, KERNEL_WIDTH,
        FFN_FILTERS, GTL_HEADS, GTL_DROPOUT,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Weight-constrained Conv2d
# ──────────────────────────────────────────────────────────────────────────────

class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with per-filter L2 max-norm weight renormalization."""

    def __init__(self, *args, max_norm=2, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)


# ──────────────────────────────────────────────────────────────────────────────
# 1. LTL — Local Temporal Learner
# ──────────────────────────────────────────────────────────────────────────────

class LocalTemporalLearner(nn.Module):
    """
    Extracts local temporal features per channel via 1D convolution.
    Input:  [B, 1, C, T]
    Output: [B, F, C, T]
    """

    def __init__(self, n_filters=F_FILTERS, kernel=LTL_KERNEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.ZeroPad2d(((kernel - 1) // 2, kernel // 2, 0, 0)),
            Conv2dWithConstraint(1, n_filters, (1, kernel), max_norm=2),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2. GCM — Graph Construction Module (one per HGP branch)
# ──────────────────────────────────────────────────────────────────────────────

class GraphConstructionModule(nn.Module):
    """
    Learnable adjacency matrix for EEG channel graph (paper Sec. III-C).

    Initialization (eq.1): Ã_prior = ReLU(A_prior + A_prior^T) + I
    This pre-processing is done outside (in train.py) so A_init already
    encodes self-loops; the forward pass does NOT add I again.

    Forward normalization (eq.2-4):
        D_i = Σ_j A_ij + mask_i  (mask=1 for isolated nodes)
        Â = D^{-1/2} A D^{-1/2}
    """

    def __init__(self, n_channels=N_CHANNELS, A_init=None):
        super().__init__()
        if A_init is not None:
            self.A = nn.Parameter(A_init.float())
        else:
            # Fallback: identity (encodes self-loops with no edges)
            self.A = nn.Parameter(torch.eye(n_channels))

    def forward(self):
        A = torch.relu(self.A)                              # non-negative weights
        D_raw = A.sum(dim=1)
        mask = (D_raw == 0).float()                         # isolated-node mask
        D = D_raw + mask
        D_inv_sqrt = D.pow(-0.5)
        return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. HGP — Hierarchical GCN Pyramid
# ──────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """Single GCN layer: H' = ReLU(A_norm · H · W + b)  (paper eq.5, σ = ReLU)"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, A_norm):
        """
        x:      [B, F_in, C, T]
        A_norm: [C, C]
        →       [B, F_out, C, T]
        """
        x = x.permute(0, 3, 2, 1)          # [B, T, C, F]
        x = torch.matmul(A_norm, x)         # graph diffusion
        x = self.linear(x)                  # feature transform
        x = F.relu(x)                       # paper uses ReLU (eq.5)
        return x.permute(0, 3, 2, 1)        # [B, F_out, C, T]


class GCNBranch(nn.Module):
    """
    One HGP branch: `depth` stacked GCN layers with its own learnable A.
    Paper Sec. III-D: "Each encoder uses an independent learnable adjacency matrix."
    """

    def __init__(self, in_f, out_f, depth, n_channels=N_CHANNELS, A_init=None):
        super().__init__()
        self.gcm = GraphConstructionModule(n_channels, A_init)
        layers = []
        for i in range(depth):
            layers.append(GraphConvLayer(in_f if i == 0 else out_f, out_f))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        A = self.gcm()
        for layer in self.layers:
            x = layer(x, A)
        return x


class HierarchicalGCNPyramid(nn.Module):
    """
    K parallel GCN branches at increasing depths (paper Sec. III-D).
    K=2 ("two hierarchical graph encoders"), each with its own GCM.

    Input:  [B, F, C, T]  (LTL output)
    Output: [B, F*(K+1), C, T]  — branch outputs + residual (original LTL input)

    "The output from the HGP…is first integrated with the original features
     through a residual connection." (paper Sec. III-E)
    """

    def __init__(self, in_f=F_FILTERS, out_f=F_FILTERS, depths=None,
                 n_channels=N_CHANNELS, A_init=None):
        super().__init__()
        if depths is None:
            depths = HGP_DEPTHS
        self.branches = nn.ModuleList([
            GCNBranch(in_f, out_f, d, n_channels, A_init) for d in depths
        ])

    def forward(self, x):
        branch_outs = [b(x) for b in self.branches]
        # Residual: concatenate branch outputs with original LTL input
        return torch.cat(branch_outs + [x], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# 4. GSL — Global Spatial Learner
# ──────────────────────────────────────────────────────────────────────────────

class GlobalSpatialLearner(nn.Module):
    """
    Depth-wise convolution spanning all C channels → collapses spatial dim.
    Paper Sec. III-E: Conv(C,1) → BN → ELU → Dropout(0.5) → AvgPool(1,3)

    Input:  [B, F, C, T]
    Output: [B, F, 1, T//3]
    """

    def __init__(self, n_filters, n_channels=N_CHANNELS, dropout=DROPOUT_P):
        super().__init__()
        self.net = nn.Sequential(
            Conv2dWithConstraint(n_filters, n_filters, (n_channels, 1),
                                 bias=False, max_norm=2),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.AvgPool2d((1, 3), stride=(1, 3)),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 5. FFN — Feature Fusion Network
# ──────────────────────────────────────────────────────────────────────────────

class FeatureFusionNetwork(nn.Module):
    """
    3 conv blocks for feature refinement + temporal downsampling.
    Paper Sec. III-F: filters [50, 100, 200], each block:
        Dropout → ZeroPad → Conv → BN → ELU → MaxPool(1,3)

    Input:  [B, F_in, 1, T]
    Output: [B, 200, 1, T//27]  (3 × pool-by-3)
    """

    def __init__(self, in_f, filter_list=None, kernel=KERNEL_WIDTH,
                 pool=POOL_WIDTH, dropout=DROPOUT_P):
        super().__init__()
        if filter_list is None:
            filter_list = FFN_FILTERS
        blocks = []
        prev = in_f
        for f in filter_list:
            blocks.append(nn.Sequential(
                nn.Dropout(p=dropout),
                nn.ZeroPad2d(((kernel - 1) // 2, kernel // 2, 0, 0)),
                Conv2dWithConstraint(prev, f, (1, kernel), bias=False, max_norm=2),
                nn.BatchNorm2d(f),
                nn.ELU(),
                nn.MaxPool2d((1, pool), stride=(1, pool)),
            ))
            prev = f
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ──────────────────────────────────────────────────────────────────────────────
# 6. GTL — Global Temporal Learner
# ──────────────────────────────────────────────────────────────────────────────

class GlobalTemporalLearner(nn.Module):
    """
    Multi-head self-attention over the temporal dimension with residual connection.
    Paper Sec. III-G.

    Input:  [B, F, 1, T]
    Output: [B, F, 1, T]
    """

    def __init__(self, embed_dim, n_heads=GTL_HEADS, dropout=GTL_DROPOUT):
        super().__init__()
        self.attn    = nn.MultiheadAttention(embed_dim, n_heads,
                                             dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, Fdim, _, T = x.shape
        seq = x.squeeze(2).permute(0, 2, 1)             # [B, T, F]
        attn_out, _ = self.attn(seq, seq, seq)           # [B, T, F]
        out = self.norm(seq + self.dropout(attn_out))    # residual + LN
        return out.permute(0, 2, 1).unsqueeze(2)         # [B, F, 1, T]


# ──────────────────────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────────────────────

class EEG2Gait(nn.Module):
    """
    Full EEG2GAIT model (paper architecture).

    Input:  [B, 1, C, T]
    Output: [B, n_joints]

    Temporal dimension progression (T=100, pool=3, K=2 HGP branches):
        LTL  → [B, 25, 59, 100]
        HGP  → [B, 75, 59, 100]   (K*F + F = 3*25 residual cat)
        GSL  → [B, 75,  1,  33]   (AvgPool T//3)
        FFN  → [B, 200, 1,   1]   (3×MaxPool T//3 each: 33→11→3→1)
        GTL  → [B, 200, 1,   1]
        OUT  → cat([GTL_in, GTL_out], dim=3) → [B, 200, 1, 2]
             → Conv(200, 6, (1,2)) → [B, 6, 1, 1] → [B, 6]
    """

    def __init__(self,
                 n_channels: int = N_CHANNELS,
                 n_time: int = WINDOW_SAMPS,
                 n_joints: int = N_JOINTS,
                 A_init: torch.Tensor = None):
        super().__init__()

        n_hgp_branches = len(HGP_DEPTHS)
        # GSL input = branch outputs (K) + original LTL output (residual)
        n_gsl_in = F_FILTERS * (n_hgp_branches + 1)

        # 1. LTL
        self.ltl = LocalTemporalLearner(F_FILTERS, LTL_KERNEL)

        # 2+3. HGP (GCMs are owned by each branch)
        self.hgp = HierarchicalGCNPyramid(F_FILTERS, F_FILTERS, HGP_DEPTHS,
                                           n_channels, A_init)

        # 4. GSL
        self.gsl = GlobalSpatialLearner(n_gsl_in, n_channels)

        # 5. FFN
        self.ffn = FeatureFusionNetwork(n_gsl_in, FFN_FILTERS)

        # 6. GTL
        self.gtl = GlobalTemporalLearner(FFN_FILTERS[-1], GTL_HEADS, GTL_DROPOUT)

        # 7. Output
        # T progression: GSL pool-3 + 3 FFN pool-3 = 4 divisions by POOL_WIDTH
        T_final = n_time
        for _ in range(1 + len(FFN_FILTERS)):   # 1 GSL + len(FFN) FFN pools
            T_final = T_final // POOL_WIDTH
        # Output layer: cat(GTL_in, GTL_out) doubles temporal dim
        self.output = Conv2dWithConstraint(
            FFN_FILTERS[-1], n_joints, (1, T_final * 2), max_norm=0.5
        )

    def forward(self, x):
        x = self.ltl(x)                # [B, 25, 59, 100]
        x = self.hgp(x)               # [B, 75, 59, 100]  (cat of 2 branches + residual)
        x = self.gsl(x)               # [B, 75,  1,  33]
        x = self.ffn(x)               # [B, 200, 1,   1]
        gtl_in = x
        x = self.gtl(x)               # [B, 200, 1,   1]
        # Output: cat GTL input and output along temporal dim (paper Sec. III-H)
        x = torch.cat([gtl_in, x], dim=3)   # [B, 200, 1, 2]
        x = self.output(x)            # [B, 6, 1, 1]
        return x.squeeze(3).squeeze(2)       # [B, 6]


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(A_init=None, **kwargs) -> EEG2Gait:
    """Construct EEG2Gait. Pass A_init for GCM initialization (per paper eq.1)."""
    return EEG2Gait(A_init=A_init, **kwargs)
