"""
config.py
---------
Central configuration for the EEG2GAIT pipeline.
All hyper-parameters are defined here so every other module
can import from a single source of truth.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "RepositoryData"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
SUBJECTS = [f"SL{i:02d}" for i in range(1, 9)]   # SL01 … SL08
SESSIONS = ["T01", "T02", "T03"]

# Raw sampling rate (from data inspection: ~333.3 Hz → 1/0.003)
RAW_FS   = 333.33   # Hz (approximate; actual dt = 0.003 s)

# Target sampling rate after downsampling
TARGET_FS = 100     # Hz

# Band-pass filter  [Hz]
BANDPASS_LO = 0.1
BANDPASS_HI = 48.0

# Number of EEG channels in the file (64 in header, first col is time → 64 data cols)
# 5 channels to drop (EOG / artifact): indices 0-based after removing timestamp
# Typical 64-ch BrainProducts layout: channels 62,63 are EOG; drop any non-brain ch
# Paper says 59 channels are used → drop 5 artifact channels
N_CHANNELS_RAW  = 64
EOG_CHAN_INDICES = [32, 38, 39, 62, 63]   # 0-based in the data matrix (after time col)
N_CHANNELS       = 59                     # channels kept

# EEG electrode 3D positions (approximate from standard 64-ch BrainProducts layout)
# Used to build the graph adjacency matrix (30 mm radius threshold)
# Format: dict mapping 0-based *kept* channel index → (x,y,z) in mm
# Full layout will be built inside dataset.py from the digitizer file if available,
# otherwise a standard 64-ch layout is used.
RADIUS_MM = 30.0    # adjacency radius for GCM

# ── Joints ───────────────────────────────────────────────────────────────────
# joints.txt header says: 6 joints (GHR GKR GAR GHL GKL GAL …)
# The first 6 columns after the timestamp are the *gait* joint angles
JOINT_NAMES = ["GHR", "GKR", "GAR", "GHL", "GKL", "GAL"]
N_JOINTS    = 6     # d_j in the paper

# ── MoBI Data-Split (per session, in minutes) ─────────────────────────────
TRAIN_MIN   = 13.5
VAL_MIN     = 1.5
TEST_MIN    = 5.0

# ── Window / stride ──────────────────────────────────────────────────────────
# v3 change: 1 s → 2 s window. Each sample now covers ~2 gait cycles instead of 1,
# giving the GTL/FFN convs cross-cycle context to compare. T at GTL goes 1 → 2.
WINDOW_SECS  = 2.0
STRIDE_SECS  = 0.1
WINDOW_SAMPS = int(WINDOW_SECS  * TARGET_FS)   # 200 samples
STRIDE_SAMPS = int(STRIDE_SECS * TARGET_FS)    # 10  samples

# ── Model (Paper Architecture: LTL → GCM → HGP → GSL → FFN → GTL → Output) ──
F_FILTERS       = 25              # LTL temporal filters
LTL_KERNEL      = 10              # LTL conv kernel width
# v3 change: K=2 → K=3 graph branches (depths 1, 2, 3). More diverse spatial
# receptive fields — shallower branch captures local clusters, deeper one captures
# global brain-network patterns. Matches the Kaggle-notebook variant.
HGP_DEPTHS      = [1, 2, 3]
DROPOUT_P       = 0.5             # dropout probability
POOL_WIDTH      = 3               # MaxPool temporal stride
KERNEL_WIDTH    = 10              # FFN conv kernel width
FFN_FILTERS     = [50, 100, 200]  # FFN filter progression: 3 blocks (paper Sec. III-F)
GTL_HEADS       = 4               # GTL multi-head self-attention heads
GTL_DROPOUT     = 0.1             # GTL attention dropout

# ── Training ─────────────────────────────────────────────────────────────────
# v3-speed: batch 100 → 256 (T4 has plenty of memory; ~2× faster per epoch)
BATCH_SIZE     = 256
LR             = 1e-3
MAX_EPOCHS     = 50
PATIENCE       = 20

# v3 additions: regularisation + LR schedule
WEIGHT_DECAY    = 1e-4            # AdamW weight decay (mild L2)
LR_WARMUP_EPOCHS = 5              # linear warmup before cosine decay

# ── Loss ─────────────────────────────────────────────────────────────────────
ALPHA   = 0.5    # freq / time weighting
BETA    = 0.1    # reward strength
EPSILON = 1e-8   # numerical stability

# v3 change: per-joint loss weighting. Knees (GKR, GKL) had MAE ~2× the others on
# the master baseline — they're the bottleneck. Up-weighting their MSE term gives
# the optimiser a stronger signal where it matters most.
#                       GHR  GKR  GAR  GHL  GKL  GAL
JOINT_LOSS_WEIGHTS = [1.0, 1.5, 1.0, 1.0, 1.5, 1.0]

# ── Augmentation (train split only) ──────────────────────────────────────────
# v3 additions: standard EEG augmentations to reduce overfitting on 8 subjects.
AUG_CHANNEL_DROPOUT = 0.10        # P(per-channel zeroing) per sample
AUG_NOISE_STD       = 0.02        # additive Gaussian noise (relative to signal std)
AUG_TIME_JITTER     = 5           # samples (±5 = ±50 ms at 100 Hz)

# ── Misc ─────────────────────────────────────────────────────────────────────
SEED        = 42
# v3-speed: 2 worker processes for overlapped data prep + GPU compute.
NUM_WORKERS = 2
DEVICE      = "auto"  # auto-selects cuda / mps / cpu
