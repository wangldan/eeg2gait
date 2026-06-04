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
WINDOW_SECS  = 1.0          # 1-second window
STRIDE_SECS  = 0.1          # 100 ms stride (10-fold overlap)
WINDOW_SAMPS = int(WINDOW_SECS  * TARGET_FS)   # 100 samples
STRIDE_SAMPS = int(STRIDE_SECS * TARGET_FS)    # 10  samples

# ── Model (Paper Architecture: LTL → GCM → HGP → GSL → FFN → GTL → Output) ──
F_FILTERS       = 25              # LTL temporal filters
LTL_KERNEL      = 10              # LTL conv kernel width
HGP_DEPTHS      = [1, 2]          # two hierarchical graph encoders (paper Sec. III-D)
DROPOUT_P       = 0.5             # dropout probability
POOL_WIDTH      = 3               # MaxPool temporal stride
KERNEL_WIDTH    = 10              # FFN conv kernel width
FFN_FILTERS     = [50, 100, 200]  # FFN filter progression: 3 blocks (paper Sec. III-F)
GTL_HEADS       = 4               # GTL multi-head self-attention heads
GTL_DROPOUT     = 0.1             # GTL attention dropout

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 100
LR             = 1e-3
MAX_EPOCHS     = 50
PATIENCE       = 30         # early-stop patience (on val Pearson r)

# ── Loss ─────────────────────────────────────────────────────────────────────
ALPHA   = 0.5    # freq / time weighting
BETA    = 0.1    # reward strength
EPSILON = 1e-8   # numerical stability

# ── Misc ─────────────────────────────────────────────────────────────────────
SEED        = 42
NUM_WORKERS = 0   # set >0 if you have plenty of RAM
DEVICE      = "auto"  # auto-selects cuda / mps / cpu
