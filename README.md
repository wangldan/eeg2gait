# EEG2GAIT

PyTorch implementation of **EEG2GAIT: A Hierarchical Graph Convolutional Network for EEG-Based Gait Decoding**  
Xi Fu et al. — *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, Vol. 34, 2026  
DOI: [10.1109/TNSRE.2025.3647101](https://doi.org/10.1109/TNSRE.2025.3647101)

---

## Overview

EEG2GAIT decodes lower-limb joint angles from EEG signals during walking. It combines:
- A **Hierarchical GCN Pyramid (HGP)** to capture multi-level spatial dependencies between EEG electrodes
- A **Hybrid Temporal-Spectral Reward (HTSR) loss** that jointly optimises time-domain (MSE) and frequency-domain (FFT L1) objectives with a reward term that emphasises well-predicted samples

Evaluated on the MoBI dataset: **r = 0.779, R² = 0.597, MAE = 4.384**

---

## Architecture

```
Input [B, 1, 59, 100]
  │
  ├── 1. LTL  — Local Temporal Learner
  │     Conv2d(1→25, 1×10) + BN + ELU
  │     → [B, 25, 59, 100]
  │
  ├── 2+3. HGP — Hierarchical GCN Pyramid  (K=2 branches, each with own learnable A)
  │     Branch 0 (depth 1): 1 GCN layer  → [B, 25, 59, 100]
  │     Branch 1 (depth 2): 2 GCN layers → [B, 25, 59, 100]
  │     + residual (original LTL output) → cat → [B, 75, 59, 100]
  │
  ├── 4. GSL  — Global Spatial Learner
  │     Conv2d(75, 75, 59×1) + BN + ELU + Dropout(0.5) + AvgPool(1,3)
  │     → [B, 75, 1, 33]
  │
  ├── 5. FFN  — Feature Fusion Network  (3 blocks)
  │     Block 1: Dropout + Conv(75→50,  1×10) + BN + ELU + MaxPool(1,3) → [B,  50, 1, 11]
  │     Block 2: Dropout + Conv(50→100, 1×10) + BN + ELU + MaxPool(1,3) → [B, 100, 1,  3]
  │     Block 3: Dropout + Conv(100→200,1×10) + BN + ELU + MaxPool(1,3) → [B, 200, 1,  1]
  │
  ├── 6. GTL  — Global Temporal Learner
  │     Multi-head self-attention (4 heads) + residual + LayerNorm
  │     → [B, 200, 1, 1]
  │
  └── 7. Output
        cat(GTL_input, GTL_output) along time → [B, 200, 1, 2]
        Conv2d(200→6, 1×2, max-norm=0.5) + squeeze → [B, 6]
```

Each GCN layer: `H' = ReLU(D^{-½} A D^{-½} · H · W + b)`  
Adjacency matrices are learnable, one per HGP branch, initialised from electrode positions via `Ã = ReLU(A + Aᵀ) + I` (paper eq. 1).

---

## Loss Function (HTSR)

```
L_total = α · L_freq_reward + (1−α) · L_time_reward

L_time        = MSE(ŷ, y)
L_freq        = L1(|FFT(ŷ)|, |FFT(y)|)          # FFT over joint dimension
L_*_reward    = L_* + β · log(1 − e^{−L_*} + ε)  # reward well-predicted samples
```

Default: α = 0.5, β = 0.1

---

## Preprocessing

Applied per session in this order (paper Sec. IV-A):

1. Drop 5 EOG/artifact channels → 59 channels
2. **Band-pass filter** 0.1–48 Hz, minimum-phase Butterworth (order 4)
3. **Common Average Reference (CAR)**
4. Downsample 333 Hz → 100 Hz (polyphase)
5. Sliding windows: 1 s / 100 ms stride

---

## Repository Structure

```
src/
  config.py      — all hyperparameters
  model.py       — LTL, GCM, HGP, GSL, FFN, GTL, output + build_model()
  dataset.py     — MoBI data loading, preprocessing, windowing, adjacency matrix
  loss.py        — HTSR loss
  metrics.py     — Pearson r, R², MAE
  train.py       — training loop with early stopping
kaggle_push/
  eeg2gait_notebook_v2.ipynb  — self-contained Kaggle training notebook
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Train on all subjects/sessions (auto-detects GPU)
python src/train.py --device auto --output-dir outputs/

# Single subject
python src/train.py --subjects SL01 --sessions T01 T02 T03
```

Checkpoints and results are saved to `outputs/best_model.pt` and `outputs/results.json`.

---

## MoBI Dataset

| Parameter | Value |
|---|---|
| Subjects | 8 (SL01–SL08) |
| Sessions per subject | 3 (T01–T03) |
| EEG channels | 59 (64 minus 5 EOG) |
| Sampling rate | 100 Hz (downsampled from ~333 Hz) |
| Window size | 1 s (100 samples) |
| Stride | 100 ms (10 samples) |
| Joint angles | 6 — GHR, GKR, GAR, GHL, GKL, GAL |
| Train split | first 13.5 min |
| Val split | next 1.5 min |
| Test split | final 5 min |

---

## Citation

```bibtex
@article{fu2026eeg2gait,
  title   = {{EEG2GAIT}: A Hierarchical Graph Convolutional Network for {EEG}-Based Gait Decoding},
  author  = {Fu, Xi and Liu, Rui and Wai, Aung Aung Phyo and Pulferer, Hannah and
             Robinson, Neethu and M{\"u}ller-Putz, Gernot R. and Guan, Cuntai},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume  = {34},
  pages   = {441--454},
  year    = {2026},
  doi     = {10.1109/TNSRE.2025.3647101}
}
```
