# EEG2GAIT — Architecture Notes

## Paper

**EEG2GAIT: A Hierarchical Graph Convolutional Network for EEG-Based Gait Decoding**  
Xi Fu et al., IEEE TNSRE Vol. 34, 2026. DOI: 10.1109/TNSRE.2025.3647101

---

## Architecture (7 Components)

```
Input [B, 1, 59, 100]
  │
  ├── 1. LTL  (Local Temporal Learner)
  │     Conv2d(1→25, 1×10) + BN + ELU
  │     → [B, 25, 59, 100]
  │
  ├── 2+3. HGP  (Hierarchical GCN Pyramid)
  │     K=2 branches, each with its own learnable adjacency matrix (GCM)
  │     Branch 0 (depth 1): 1 GCN layer  → [B, 25, 59, 100]
  │     Branch 1 (depth 2): 2 GCN layers → [B, 25, 59, 100]
  │     + residual skip (LTL output)     → cat → [B, 75, 59, 100]
  │
  ├── 4. GSL  (Global Spatial Learner)
  │     Conv2d(75, 75, 59×1) + BN + ELU + Dropout(0.5) + AvgPool(1,3)
  │     → [B, 75, 1, 33]
  │
  ├── 5. FFN  (Feature Fusion Network)  — 3 blocks
  │     Block 1: Dropout + Conv(75→50,  1×10) + BN + ELU + MaxPool(1,3) → [B,  50, 1, 11]
  │     Block 2: Dropout + Conv(50→100, 1×10) + BN + ELU + MaxPool(1,3) → [B, 100, 1,  3]
  │     Block 3: Dropout + Conv(100→200,1×10) + BN + ELU + MaxPool(1,3) → [B, 200, 1,  1]
  │
  ├── 6. GTL  (Global Temporal Learner)
  │     Multi-head self-attention (4 heads) + residual + LayerNorm
  │     → [B, 200, 1, 1]
  │
  └── 7. Output
        cat(GTL_input, GTL_output) along time → [B, 200, 1, 2]
        Conv2dWithConstraint(200→6, 1×2, max_norm=0.5)
        squeeze → [B, 6]
```

**Trainable parameters: ~793,000**

---

## What Each Component Does

### 1. LTL — Local Temporal Learner
Temporal convolution applied identically across all 59 EEG channels. Extracts local oscillatory patterns (motor-related neural activity) while suppressing low-frequency drift. Kernel size 10 at 100 Hz = 100 ms receptive field.

### 2+3. HGP — Hierarchical GCN Pyramid
Two parallel GCN branches at depths 1 and 2, each with its **own independent learnable adjacency matrix**:
- **Depth 1** (1 GCN layer): immediate neighbour information — shallow, global connections
- **Depth 2** (2 GCN layers): 2-hop neighbourhood — deeper, localised cluster representations

Each GCN layer: `H' = ReLU(D^{-½} A D^{-½} · H · W + b)` (paper eq. 5, σ = ReLU).  
Branch outputs are concatenated with the original LTL output (residual connection) before GSL.

**Adjacency initialisation (paper eq. 1):** `Ã = ReLU(A_prior + A_prior^T) + I`  
where `A_prior` connects electrode pairs within 30 mm. The matrix is then learned end-to-end.

### 4. GSL — Global Spatial Learner
Depth-wise convolution with kernel spanning all 59 channels collapses the spatial dimension. Learns a weighted combination of all electrode signals. Followed by Dropout(0.5) and AvgPool(1,3) which reduces temporal resolution by 3×.

### 5. FFN — Feature Fusion Network
Three conv blocks that progressively increase feature depth (75→50→100→200) while reducing temporal resolution 3× per block via MaxPool. Each block includes dropout (0.5), batch normalization, and ELU activation.

### 6. GTL — Global Temporal Learner
4-head self-attention over the remaining temporal dimension. Captures long-range dependencies across the gait window. Uses residual connection and LayerNorm to preserve gradient flow.

### 7. Output Layer
GTL input and output are concatenated along the temporal dim (`T_final × 2 = 2`) before the final weight-constrained Conv2d (max-norm=0.5) that maps to 6 joint angle predictions.

---

## Weight Constraint

All Conv2d layers use `Conv2dWithConstraint` — before each forward pass, each filter's L2 norm is clamped: ‖w_i‖₂ ≤ max_norm. Prevents filter collapse and acts as implicit regularization. Most layers: max_norm=2; output layer: max_norm=0.5.

---

## Loss Function (HTSR)

Hybrid Temporal-Spectral Reward loss (paper Sec. III-I):

```
L_total = α · L_freq_reward + (1−α) · L_time_reward

L_time     = MSE(ŷ, y)                               (eq. 8)
L_freq     = L1(|FFT(ŷ)|, |FFT(y)|)                  (eq. 10, FFT over joint dim)
L_*_reward = L_* + β · log(1 − e^{−L_*} + ε)         (eqs. 9, 11)
```

Default: α = 0.5, β = 0.1. The reward term encourages learning from well-predicted samples.

---

## Preprocessing Pipeline

Order per paper Sec. IV-A:

1. Drop 5 EOG/artifact channels → 59 channels
2. **Band-pass filter** 0.1–48 Hz, minimum-phase Butterworth order 4 (`sosfilt`)
3. **Common Average Reference (CAR)**
4. Downsample ~333 Hz → 100 Hz (polyphase resampling)
5. Sliding windows: 1 s length, 100 ms stride

---

## Files

| File | Role |
|------|------|
| `src/model.py` | All 7 components + `build_model()` factory |
| `src/config.py` | All hyperparameters |
| `src/dataset.py` | MoBI data loading, preprocessing, windowing, adjacency matrix |
| `src/train.py` | Training loop with early stopping |
| `src/loss.py` | HTSR loss |
| `src/metrics.py` | Pearson r, R², MAE evaluation |
