# EEG2GAIT — Architecture Notes

## Paper

**EEG2GAIT: A Hierarchical Graph Convolutional Network for EEG-Based Gait Decoding**

## Architecture (7 Components)

```
Input [B, 1, 59, 100]
  │
  ├── 1. LTL  (Local Temporal Learner)
  │     Conv2d(1→25, 1×10) + BN + ELU
  │     → [B, 25, 59, 100]
  │
  ├── 2. GCM  (Graph Construction Module)
  │     Learnable adjacency A ∈ R^{59×59}
  │     Initialized from electrode positions
  │     Symmetric normalization: D^{-½} A D^{-½}
  │
  ├── 3. HGP  (Hierarchical GCN Pyramid)
  │     3 branches at depths [1, 2, 3]
  │     Each: stacked GCN layers (A·H·W + b → ELU)
  │     Concat → [B, 75, 59, 100]
  │
  ├── 4. GSL  (Global Spatial Learner)
  │     Conv2d(75, 75, 59×1) collapses all channels
  │     + BN + ELU → [B, 75, 1, 100]
  │
  ├── 5. FFN  (Feature Fusion Network)
  │     Block 1: Conv(75→100, 1×10) + BN + ELU + MaxPool(1,3) → [B, 100, 1, 33]
  │     Block 2: Conv(100→200, 1×10) + BN + ELU + MaxPool(1,3) → [B, 200, 1, 11]
  │
  ├── 6. GTL  (Global Temporal Learner)
  │     Multi-head self-attention (4 heads)
  │     Residual connection + LayerNorm
  │     → [B, 200, 1, 11]
  │
  └── 7. Output
        Conv2d(200→6, 1×11) with max-norm=0.5
        Squeeze → [B, 6]
```

**Parameters: 789,737**

---

## What Each Component Does

### 1. LTL — Local Temporal Learner
Temporal convolution applied identically across all 59 EEG channels. Extracts local oscillatory patterns (motor-related neural activity) while suppressing low-frequency drift. Kernel size 10 at 100 Hz = 100ms receptive field.

### 2. GCM — Graph Construction Module
Maintains a learnable 59×59 adjacency matrix representing EEG electrode connectivity. Initialized from physical electrode positions using a Gaussian distance kernel (30mm radius), then fine-tuned during training. Applied with symmetric normalization (self-loops + degree scaling).

### 3. HGP — Hierarchical GCN Pyramid
Three parallel GCN branches at depths 1, 2, and 3:
- **Depth 1** (1 GCN layer): immediate neighbor information
- **Depth 2** (2 GCN layers): 2-hop neighborhood patterns
- **Depth 3** (3 GCN layers): wider cortical connectivity

Each GCN layer: H' = ELU(A_norm · H · W + b). Outputs concatenated → 25×3 = 75 filters.

### 4. GSL — Global Spatial Learner
Single convolution with kernel spanning all 59 channels, collapsing the spatial dimension to 1. This learns a weighted combination of all electrode signals — the "global spatial summary" of the EEG.

### 5. FFN — Feature Fusion Network
Two conv blocks that progressively increase feature depth (75→100→200) while reducing temporal resolution through MaxPool(1,3). Each block includes dropout (0.5), batch normalization, and ELU activation.

### 6. GTL — Global Temporal Learner
4-head self-attention over the 11 remaining time steps. Captures long-range temporal dependencies across the entire gait window. Uses residual connection (x + attention(x)) and LayerNorm to preserve gradient flow.

### 7. Output Layer
Weight-constrained Conv2d (max-norm=0.5) that maps the 200-dimensional features across 11 time steps into 6 joint angle predictions.

---

## Weight Constraint

All Conv2d layers use `Conv2dWithConstraint` — before each forward pass, each filter's L2 norm is clamped: ‖w_i‖₂ ≤ max_norm. This prevents filter collapse and acts as implicit regularization. Most layers use max_norm=2; the output layer uses 0.5.

---

## Loss Function (HTSR)

Hybrid Temporal-Spectral Reward loss:
```
L_total = α · L_freq_reward + (1−α) · L_time_reward
```
Where L_time = MSE, L_freq = L1, and each is wrapped with a reward term: L + β·log(1 − e^{−L} + ε). Default: α=0.5, β=0.1.

---

## MoBI Dataset Adaptation

| Parameter | Value |
|-----------|-------|
| EEG channels | 59 (64 minus 5 EOG) |
| Sampling rate | 100 Hz (downsampled from 333 Hz) |
| Window size | 1 second (100 samples) |
| Stride | 100 ms (10 samples) |
| Joint angles | 6 (GHR, GKR, GAR, GHL, GKL, GAL) |
| Subjects | 8 (SL01–SL08) |
| Sessions | 3 (T01–T03) |

---

## Files

| File | Role |
|------|------|
| `src/model.py` | All 7 components + `build_model()` factory |
| `src/config.py` | All hyperparameters |
| `src/dataset.py` | MoBI data loading, windowing, adjacency matrix |
| `src/train.py` | Training loop with early stopping |
| `src/loss.py` | HTSR loss |
| `src/metrics.py` | Pearson r, R², MAE evaluation |
