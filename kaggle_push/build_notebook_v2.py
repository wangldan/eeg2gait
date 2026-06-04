#!/usr/bin/env python3
"""
build_notebook_v2.py
--------------------
Self-contained Kaggle notebook generator for EEG2GAIT.
All src/ files are embedded below (no external reads needed).

Usage:
    python3 kaggle_push/build_notebook_v2.py

Output:
    kaggle_push/eeg2gait_notebook_v2.ipynb
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "eeg2gait_notebook_v2.json"

# ════════════════════════════════════════════════════════════════════════════
# EMBEDDED SOURCE FILES
# ════════════════════════════════════════════════════════════════════════════

MODULES = {}

# ── config.py (Kaggle-adapted: paths + DEVICE) ────────────────────────────

MODULES['config.py'] = r'''
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
ROOT_DIR   = Path("/kaggle")
DATA_DIR   = Path("/kaggle/input/jwangldan09/eeg2gait-fall-prediction-dataset/RepositoryData")
OUTPUT_DIR = Path("/kaggle/working/outputs")
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
DEVICE      = "auto"   # auto-detect cuda / mps / cpu
'''[1:]


# ── dataset.py ──────────────────────────────────────────────────────────────

MODULES['dataset.py'] = r'''
"""
dataset.py
----------
Custom PyTorch Dataset & DataLoader for the MoBI EEG2GAIT dataset.

File layout expected:
    DATA_DIR/
        SL01-T01/
            eeg.txt     – header: "64 channels\n"
                          data rows: timestamp, ch0, ch1, …, ch63  (tab-separated)
            joints.txt  – header: "6 joints (GHR GKR GAR …)\n"
                          header2: "Joint Factor …\n"
                          data rows: timestamp, j0, j1, j2, j3, j4, j5, (more…)
        SL01-T02/ …

Preprocessing pipeline (per session):
    1. Read raw EEG + joint angles at ~333 Hz
    2. Drop 5 artifact / EOG channels  → 59 channels
    3. Common-Average Reference (CAR)
    4. Band-pass filter  0.1–48 Hz  (scipy butterworth, zero-phase)
    5. Resample to 100 Hz  (scipy.signal.resample_poly)
    6. Align EEG ↔ joints by timestamp (both files share the same timestamps)
    7. Split by time: first 13.5 min → train, next 1.5 min → val, last 5 min → test
    8. Sliding-window extraction: 1-second windows, 100 ms stride
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, sosfilt, resample_poly
from scipy.spatial.distance import cdist
from math import gcd

try:
    from .config import (
        DATA_DIR, SUBJECTS, SESSIONS,
        RAW_FS, TARGET_FS,
        BANDPASS_LO, BANDPASS_HI,
        N_CHANNELS_RAW, EOG_CHAN_INDICES, N_CHANNELS,
        N_JOINTS,
        TRAIN_MIN, VAL_MIN, TEST_MIN,
        WINDOW_SAMPS, STRIDE_SAMPS,
        RADIUS_MM,
    )
except ImportError:
    from config import (
        DATA_DIR, SUBJECTS, SESSIONS,
        RAW_FS, TARGET_FS,
        BANDPASS_LO, BANDPASS_HI,
        N_CHANNELS_RAW, EOG_CHAN_INDICES, N_CHANNELS,
        N_JOINTS,
        TRAIN_MIN, VAL_MIN, TEST_MIN,
        WINDOW_SAMPS, STRIDE_SAMPS,
        RADIUS_MM,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: I/O
# ──────────────────────────────────────────────────────────────────────────────

def _read_eeg(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps [N], data [N, 64]) from eeg.txt."""
    with open(path, "r", errors="replace") as f:
        _ = f.readline()  # skip "64 channels" header
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split("\t")
            try:
                rows.append([float(v) for v in vals if v])
            except ValueError:
                continue
    arr = np.array(rows, dtype=np.float32)
    timestamps = arr[:, 0]          # column 0 is time in seconds
    data       = arr[:, 1:]         # columns 1…64  (64 channels)
    return timestamps, data


def _read_joints(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (timestamps [N], joint_angles [N, 6]) from joints.txt.
    The file has two header lines; the joint factor line contains scale factors
    but the raw values in the file are already the actual joint angles in degrees.
    Only the first 6 joints (GHR,GKR,GAR,GHL,GKL,GAL) are used per the paper.
    """
    with open(path, "r", errors="replace") as f:
        h1 = f.readline()   # "6 joints (…)\n"
        h2 = f.readline()   # "Joint Factor …\n"
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split("\t")
            try:
                rows.append([float(v) for v in vals if v])
            except ValueError:
                continue
    arr = np.array(rows, dtype=np.float32)
    timestamps   = arr[:, 0]
    joint_angles = arr[:, 1:7]      # first 6 joint cols (GHR … GAL)
    return timestamps, joint_angles


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _drop_eog_channels(data: np.ndarray) -> np.ndarray:
    """Drop EOG/artifact channels; keep 59 of the 64 channels."""
    keep = [i for i in range(N_CHANNELS_RAW) if i not in EOG_CHAN_INDICES]
    return data[:, keep]   # [N, 59]


def _common_average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the mean across channels at each time point."""
    return data - data.mean(axis=1, keepdims=True)


def _bandpass_filter(data: np.ndarray, fs: float) -> np.ndarray:
    """Minimum-phase Butterworth band-pass filter (0.1–48 Hz), paper Sec. IV-A."""
    nyq  = fs / 2.0
    lo   = BANDPASS_LO / nyq
    hi   = min(BANDPASS_HI / nyq, 0.999)   # must be < 1
    sos  = butter(4, [lo, hi], btype="band", output="sos")
    # sosfilt is causal (minimum-phase), numerically stable for high-order filters
    return sosfilt(sos, data, axis=0).astype(data.dtype)


def _resample(data: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    Resample from fs_in → fs_out using polyphase method.
    Works channel-by-channel.
    """
    # Build up/down ratio via GCD reduction
    fs_in_int  = int(round(fs_in  * 3))   # × 3 to avoid fractional FS (333.33 Hz → 1000)
    fs_out_int = int(round(fs_out * 3))
    g          = gcd(fs_in_int, fs_out_int)
    up, down   = fs_out_int // g, fs_in_int // g

    resampled = np.empty((round(data.shape[0] * up / down), data.shape[1]), dtype=np.float32)
    for ch in range(data.shape[1]):
        resampled[:, ch] = resample_poly(data[:, ch], up, down).astype(np.float32)
    return resampled


def _preprocess_session(eeg_raw: np.ndarray, joints_raw: np.ndarray,
                         fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline applied to one session.

    Args:
        eeg_raw:    [N, 64]
        joints_raw: [N, 6]
        fs:         raw sampling frequency

    Returns:
        eeg_pp:    [M, 59]   at TARGET_FS
        joints_pp: [M, 6]    at TARGET_FS
    """
    # 1. Drop artifact channels
    eeg = _drop_eog_channels(eeg_raw)       # [N, 59]

    # 2. Band-pass filter (minimum-phase, paper Sec. IV-A: "First, a minimum-phase
    #    band-pass filter…was applied, followed by re-referencing to the common average")
    eeg = _bandpass_filter(eeg, fs)

    # 3. Common-Average Reference
    eeg = _common_average_reference(eeg)

    # 4. Resample EEG
    eeg = _resample(eeg, fs, TARGET_FS)     # [M, 59]

    # 5. Resample joints
    joints = _resample(joints_raw, fs, TARGET_FS)   # [M, 6]

    # Trim to same length (resampling may differ by ±1)
    n = min(len(eeg), len(joints))
    return eeg[:n], joints[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Windowing
# ──────────────────────────────────────────────────────────────────────────────

def _extract_windows(eeg: np.ndarray, joints: np.ndarray,
                     window: int = WINDOW_SAMPS,
                     stride: int = STRIDE_SAMPS
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window extraction.

    Args:
        eeg:    [T, C]
        joints: [T, J]
    Returns:
        X: [W, C, window]
        y: [W, J]          (label = mean of joint angles in window)
    """
    T = eeg.shape[0]
    indices = list(range(0, T - window + 1, stride))
    X_list, y_list = [], []
    for s in indices:
        e = s + window
        X_list.append(eeg[s:e].T)          # [C, window]
        y_list.append(joints[s:e].mean(0)) # [J]
    if not X_list:
        return np.empty((0, eeg.shape[1], window), dtype=np.float32), \
               np.empty((0, joints.shape[1]), dtype=np.float32)
    return np.stack(X_list).astype(np.float32), \
           np.stack(y_list).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Electrode positions (standard BrainProducts 64-ch layout subset, in mm)
# ──────────────────────────────────────────────────────────────────────────────

def _build_standard_positions() -> np.ndarray:
    """
    Returns approximate 3-D positions (mm) for 64 EEG channels in a standard
    BrainProducts layout projected onto a unit sphere of radius 85 mm.
    Only the 59 kept channels (after removing EOG_CHAN_INDICES) are returned.
    Shape: [59, 3]
    """
    # Azimuth / elevation (degrees) for standard 64-ch layout
    # Generated from MNE standard_1020 template reduced to 64 channels.
    # (phi=azimuth, theta=elevation from top, r=85mm)
    az_el_64 = [
        (0,   0),   # Cz
        (180, 18),  # Fz
        (0,  18),   # Pz
        (270, 18),  # C3 (left)
        (90,  18),  # C4 (right)
        (225, 18),  # F3
        (135, 18),  # F4  (approx)
        (315, 18),  # P3
        (45,  18),  # P4
        (180, 36),  # Fpz
        (0,   36),  # Oz
        (270, 36),  # T7
        (90,  36),  # T8
        (225, 36),  # F7
        (135, 36),  # F8
        (315, 36),  # P7
        (45,  36),  # P8
        (247, 28),  # FC5
        (113, 28),  # FC6
        (203, 28),  # FC1
        (157, 28),  # FC2
        (293, 28),  # CP5
        (67,  28),  # CP6
        (247, 28),  # FT9 (approx)
        (113, 28),  # FT10
        (180, 52),  # AF7
        (0,   52),  # O1  (approx)
        (270, 52),  # TP7
        (90,  52),  # TP8
        (225, 52),  # F5
        (135, 52),  # F6
        (315, 52),  # P5
        (45,  52),  # P6
        # fill remaining 31 with evenly spaced positions
        *[(i * (360/31), 72) for i in range(31)],
    ]
    r = 85.0   # sphere radius in mm
    positions = np.zeros((64, 3), dtype=np.float32)
    for i, (az, el) in enumerate(az_el_64):
        az_r = np.radians(az)
        el_r = np.radians(el)
        positions[i, 0] = r * np.sin(el_r) * np.cos(az_r)
        positions[i, 1] = r * np.sin(el_r) * np.sin(az_r)
        positions[i, 2] = r * np.cos(el_r)
    # Keep only non-EOG channels
    keep = [i for i in range(64) if i not in EOG_CHAN_INDICES]
    return positions[keep]   # [59, 3]


def build_adjacency_matrix(positions: Optional[np.ndarray] = None,
                            radius: float = RADIUS_MM) -> np.ndarray:
    """
    Build a binary adjacency matrix A ∈ {0,1}^{C×C} where A_ij=1 if the
    Euclidean distance between electrode i and j is ≤ radius (mm).
    Self-loops are included (A_ii = 1).

    Args:
        positions: [C, 3] electrode coordinates in mm.  If None, the
                   standard layout is used.
        radius:    connectivity radius in mm.

    Returns:
        A: [C, C] float32 array.
    """
    if positions is None:
        positions = _build_standard_positions()   # [59, 3]
    # Vectorised distance computation
    dist_matrix = cdist(positions, positions, metric='euclidean')
    A = (dist_matrix <= radius).astype(np.float32)
    # Self-loops are NOT added here; they are added via +I in eq.1 (train.py),
    # which also symmetrizes: Ã_prior = ReLU(A + A^T) + I
    np.fill_diagonal(A, 0.0)
    return A


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MoBISessionDataset(Dataset):
    """
    Dataset for a single (subject, session, split) tuple.

    Each item: (X, y)
        X: torch.FloatTensor [C, T]   — preprocessed EEG window
        y: torch.FloatTensor [J]      — target joint angles
    """

    def __init__(self,
                 subject: str,
                 session: str,
                 split: str,            # "train" | "val" | "test"
                 data_dir: Path = DATA_DIR,
                 verbose: bool = True):
        assert split in ("train", "val", "test")
        self.subject = subject
        self.session = session
        self.split   = split

        folder = data_dir / f"{subject}-{session}"
        if not folder.exists():
            raise FileNotFoundError(f"Session folder not found: {folder}")

        if verbose:
            print(f"  Loading {subject}-{session} [{split}] …", flush=True)

        # ── Load raw data ──────────────────────────────────────────────────
        ts_eeg,    eeg_raw    = _read_eeg   (folder / "eeg.txt")
        ts_joints, joints_raw = _read_joints(folder / "joints.txt")

        # Infer actual sampling rate from timestamps
        dt = np.median(np.diff(ts_eeg[:500]))
        fs = 1.0 / dt

        # ── Preprocess ────────────────────────────────────────────────────
        eeg_pp, joints_pp = _preprocess_session(eeg_raw, joints_raw, fs)

        # ── Split by time ─────────────────────────────────────────────────
        total_samps = len(eeg_pp)
        train_samps = int(TRAIN_MIN * 60 * TARGET_FS)
        val_samps   = int(VAL_MIN   * 60 * TARGET_FS)
        # test uses remainder (capped at TEST_MIN)
        test_end    = min(total_samps,
                          train_samps + val_samps + int(TEST_MIN * 60 * TARGET_FS))

        if split == "train":
            eeg_split    = eeg_pp   [:train_samps]
            joints_split = joints_pp[:train_samps]
        elif split == "val":
            s = train_samps
            e = train_samps + val_samps
            eeg_split    = eeg_pp   [s:e]
            joints_split = joints_pp[s:e]
        else:   # test
            s = train_samps + val_samps
            eeg_split    = eeg_pp   [s:test_end]
            joints_split = joints_pp[s:test_end]

        # ── Sliding windows ───────────────────────────────────────────────
        self.X, self.y = _extract_windows(eeg_split, joints_split)

        if verbose:
            print(f"    → {len(self.X)} windows, "
                  f"EEG: {eeg_split.shape}, joints: {joints_split.shape}", flush=True)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


class MoBIDataset(Dataset):
    """
    Aggregates data across all (subject, session) pairs for a given split.
    """

    def __init__(self,
                 subjects: List[str] = SUBJECTS,
                 sessions: List[str] = SESSIONS,
                 split: str = "train",
                 data_dir: Path = DATA_DIR,
                 verbose: bool = True):
        self.datasets: List[MoBISessionDataset] = []
        for subj in subjects:
            for sess in sessions:
                folder = data_dir / f"{subj}-{sess}"
                if not folder.exists():
                    if verbose:
                        print(f"  Skipping missing: {subj}-{sess}")
                    continue
                try:
                    ds = MoBISessionDataset(subj, sess, split, data_dir, verbose)
                    if len(ds) > 0:
                        self.datasets.append(ds)
                except Exception as exc:
                    print(f"  Error loading {subj}-{sess}: {exc}")

        # Pre-concatenate for fast indexing
        if self.datasets:
            self.X = np.concatenate([d.X for d in self.datasets], axis=0)
            self.y = np.concatenate([d.y for d in self.datasets], axis=0)
        else:
            self.X = np.empty((0, N_CHANNELS, WINDOW_SAMPS), dtype=np.float32)
            self.y = np.empty((0, N_JOINTS),                 dtype=np.float32)

        if verbose:
            print(f"\n[MoBIDataset-{split}] Total windows: {len(self.X)}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # Official model expects [1, C, T] input (4D with 1 channel dim)
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)  # [C, T] → [1, C, T]
        y = torch.from_numpy(self.y[idx])
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(subjects: List[str] = SUBJECTS,
                    sessions: List[str] = SESSIONS,
                    batch_size: int      = 100,
                    num_workers: int     = 0,
                    data_dir: Path       = DATA_DIR,
                    verbose: bool        = True
                    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for the given subject / session list.
    """
    print("Building Train dataset …")
    train_ds = MoBIDataset(subjects, sessions, "train", data_dir, verbose)
    print("Building Val   dataset …")
    val_ds   = MoBIDataset(subjects, sessions, "val",   data_dir, verbose)
    print("Building Test  dataset …")
    test_ds  = MoBIDataset(subjects, sessions, "test",  data_dir, verbose)

    _pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=_pin, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=_pin)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=_pin)

    return train_dl, val_dl, test_dl
'''[1:]


# ── model.py ────────────────────────────────────────────────────────────────

MODULES['model.py'] = r'''
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
'''[1:]


# ── loss.py ─────────────────────────────────────────────────────────────────

MODULES['loss.py'] = r'''
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
'''[1:]


# ── metrics.py ──────────────────────────────────────────────────────────────

MODULES['metrics.py'] = r'''
"""
metrics.py
----------
Evaluation metrics for EEG2GAIT gait angle prediction.

Metrics (per-joint and averaged):
  - Pearson correlation coefficient (r)
  - R² score (coefficient of determination)
  - Mean Absolute Error (MAE)
"""

import torch
import numpy as np
from typing import Dict, Tuple

try:
    from .config import JOINT_NAMES, N_JOINTS
except ImportError:
    from config import JOINT_NAMES, N_JOINTS


def pearson_r(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Pearson correlation coefficient per joint column.

    Args:
        y_pred: [N, J]
        y_true: [N, J]
    Returns:
        r: [J]
    """
    r_vals = []
    for j in range(y_pred.shape[1]):
        p = y_pred[:, j]
        t = y_true[:, j]
        # Avoid NaN for constant predictions
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            r_vals.append(0.0)
        else:
            corr = np.corrcoef(p, t)[0, 1]
            r_vals.append(float(corr) if not np.isnan(corr) else 0.0)
    return np.array(r_vals)


def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    R² (coefficient of determination) per joint.

    Args:
        y_pred: [N, J]
        y_true: [N, J]
    Returns:
        r2: [J]
    """
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    # Avoid division by zero
    r2 = np.where(ss_tot < 1e-12, 0.0, 1.0 - ss_res / ss_tot)
    return r2


def mae_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Error per joint.

    Args:
        y_pred: [N, J]
        y_true: [N, J]
    Returns:
        mae: [J]
    """
    return np.abs(y_pred - y_true).mean(axis=0)


def compute_metrics(y_pred: np.ndarray,
                    y_true: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics and return as a flat dict.

    Keys: per-joint (e.g. "r_GHR") and averaged ("r_mean", "r2_mean", "mae_mean").
    """
    r   = pearson_r(y_pred, y_true)
    r2  = r2_score (y_pred, y_true)
    mae = mae_score(y_pred, y_true)

    results: Dict[str, float] = {}
    for j, name in enumerate(JOINT_NAMES):
        results[f"r_{name}"]   = float(r[j])
        results[f"r2_{name}"]  = float(r2[j])
        results[f"mae_{name}"] = float(mae[j])

    results["r_mean"]   = float(r.mean())
    results["r2_mean"]  = float(r2.mean())
    results["mae_mean"] = float(mae.mean())
    return results


def evaluate_loader(model, loader, device: str) -> Dict[str, float]:
    """
    Run inference over the entire DataLoader and compute metrics.

    Returns:
        Dict of metric names → values.
    """
    import torch
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y_hat = model(X)
            preds.append(y_hat.cpu().numpy())
            targets.append(y.numpy())

    y_pred = np.concatenate(preds,   axis=0)  # [N, J]
    y_true = np.concatenate(targets, axis=0)  # [N, J]
    return compute_metrics(y_pred, y_true)
'''[1:]


# ── train.py ────────────────────────────────────────────────────────────────

MODULES['train.py'] = r'''
"""
train.py
--------
Training loop for EEG2GAIT.

Features:
  - Adam optimiser (lr=0.001)
  - HTSR custom loss
  - Early stopping: patience=30, monitored metric = mean Pearson r on val set
  - Per-epoch logging: loss + all evaluation metrics
  - Checkpoint saving (best val r model)
  - Final test-set evaluation with per-joint breakdown
"""

import os
import sys
import time
import json
import copy
import random
import argparse
from pathlib import Path

import torch
import numpy as np

# ── Kaggle/PyTorch compatibility fix ────────────────────────────────────────
# Some Kaggle PyTorch builds fail when torch._dynamo lazily imports torch._utils.
# Force-loading torch._utils now prevents that crash.
try:
    import torch._utils   # noqa: F401  – must happen before Adam is used
except Exception:
    pass

# ── Path setup ──────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    DATA_DIR, OUTPUT_DIR, SUBJECTS, SESSIONS,
    BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE,
    SEED, NUM_WORKERS, DEVICE,
    N_CHANNELS, WINDOW_SAMPS, N_JOINTS,
)
from dataset import get_dataloaders, build_adjacency_matrix, _build_standard_positions
from model  import build_model
from loss   import HTSRLoss
from metrics import evaluate_loader, compute_metrics


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Logging helper ───────────────────────────────────────────────────────────

def log(msg: str, log_file=None):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


# ── Main training function ──────────────────────────────────────────────────

def train(subjects=None, sessions=None, device_str=None,
          batch_size=BATCH_SIZE, lr=LR, max_epochs=MAX_EPOCHS,
          patience=PATIENCE, output_dir=OUTPUT_DIR):
    """
    Full training pipeline.

    Args:
        subjects:    list of subject IDs (default: all 8)
        sessions:    list of session IDs (default: all 3)
        device_str:  "cpu" | "cuda" | "mps"
        batch_size:  mini-batch size
        lr:          Adam learning rate
        max_epochs:  maximum training epochs
        patience:    early-stopping patience (epochs)
        output_dir:  directory for checkpoints and logs
    """
    set_seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None: subjects = SUBJECTS
    if sessions is None: sessions = SESSIONS

    # ── Device ──────────────────────────────────────────────────────────
    if device_str is None:
        device_str = DEVICE
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  EEG2GAIT Training Run")
    print(f"  Device:   {device}")
    print(f"  Subjects: {subjects}")
    print(f"  Sessions: {sessions}")
    print(f"{'='*60}\n")

    log_path = output_dir / "train_log.txt"
    log_file = open(log_path, "w")

    # ── Data ────────────────────────────────────────────────────────────
    log("Loading data …", log_file)
    train_dl, val_dl, test_dl = get_dataloaders(
        subjects=subjects, sessions=sessions,
        batch_size=batch_size, num_workers=NUM_WORKERS,
        verbose=True
    )
    log(f"Train batches: {len(train_dl)} | "
        f"Val batches: {len(val_dl)} | "
        f"Test batches: {len(test_dl)}", log_file)

    # ── Adjacency matrix (GCM initialization, paper eq.1) ────────────
    log("Building adjacency matrix …", log_file)
    positions = _build_standard_positions()           # [59, 3]
    A_np      = build_adjacency_matrix(positions)     # [59, 59], no self-loops
    A_tensor  = torch.from_numpy(A_np)
    # eq.1: Ã_prior = ReLU(A_prior + A_prior^T) + I
    A_init = torch.relu(A_tensor + A_tensor.T) + torch.eye(A_tensor.size(0))

    # ── Model ────────────────────────────────────────────────────────────
    log("Building model …", log_file)
    model = build_model(A_init=A_init).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable parameters: {n_params:,}", log_file)

    # ── Optimiser & loss ─────────────────────────────────────────────────
    # Use SGD-compatible init path to dodge torch._dynamo import bug on some
    # Kaggle environments; falls back to plain Adam if the bug is already fixed.
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    except AttributeError:
        # _dynamo import failed; rebuild optimizer manually via Optimizer base
        import importlib
        _utils = importlib.import_module("torch._utils")
        torch._utils = _utils                   # patch into torch namespace
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = HTSRLoss()

    # ── Training state ───────────────────────────────────────────────────
    best_val_r    = -1.0
    best_epoch    = 0
    epochs_no_imp = 0
    best_state    = None
    history       = {"train_loss": [], "val_r": [], "val_r2": [], "val_mae": []}

    # ── Epoch loop ───────────────────────────────────────────────────────
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss  = criterion(y_hat, y)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_dl), 1)

        # ── Validate ─────────────────────────────────────────────────────
        val_metrics = evaluate_loader(model, val_dl, device_str)
        val_r       = val_metrics["r_mean"]
        val_r2      = val_metrics["r2_mean"]
        val_mae     = val_metrics["mae_mean"]

        elapsed = time.time() - t0
        msg = (f"Epoch {epoch:03d}/{max_epochs} | "
               f"Loss: {avg_train_loss:.4f} | "
               f"Val r: {val_r:.4f} | "
               f"Val R²: {val_r2:.4f} | "
               f"Val MAE: {val_mae:.4f} | "
               f"Time: {elapsed:.1f}s")
        log(msg, log_file)

        # Track history
        history["train_loss"].append(avg_train_loss)
        history["val_r"].append(val_r)
        history["val_r2"].append(val_r2)
        history["val_mae"].append(val_mae)

        # ── Early stopping ───────────────────────────────────────────────
        if val_r > best_val_r:
            best_val_r    = val_r
            best_epoch    = epoch
            epochs_no_imp = 0
            best_state    = copy.deepcopy(model.state_dict())
            ckpt_path     = output_dir / "best_model.pt"
            torch.save({
                "epoch":      epoch,
                "state_dict": best_state,
                "val_r":      best_val_r,
                "val_r2":     val_r2,
                "val_mae":    val_mae,
            }, ckpt_path)
            log(f"  ✓ New best model saved (val r={best_val_r:.4f})", log_file)
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= patience:
            log(f"\n  Early stopping triggered after {epoch} epochs "
                f"(best epoch={best_epoch}, best val r={best_val_r:.4f})", log_file)
            break

    # ── Load best model ──────────────────────────────────────────────────
    log(f"\nLoading best model (epoch {best_epoch}) …", log_file)
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test evaluation ───────────────────────────────────────────────────
    log("\nEvaluating on TEST set …", log_file)
    test_metrics = evaluate_loader(model, test_dl, device_str)

    log("\n" + "="*60, log_file)
    log("TEST RESULTS", log_file)
    log("="*60, log_file)
    log(f"  Mean Pearson r : {test_metrics['r_mean']:.4f}", log_file)
    log(f"  Mean R²        : {test_metrics['r2_mean']:.4f}", log_file)
    log(f"  Mean MAE       : {test_metrics['mae_mean']:.4f}", log_file)
    log("-"*60, log_file)
    log("  Per-joint breakdown:", log_file)
    from config import JOINT_NAMES
    for name in JOINT_NAMES:
        log(f"    {name:5s}  r={test_metrics[f'r_{name}']:.4f}  "
            f"R²={test_metrics[f'r2_{name}']:.4f}  "
            f"MAE={test_metrics[f'mae_{name}']:.4f}", log_file)
    log("="*60, log_file)

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "best_epoch":    best_epoch,
        "best_val_r":    best_val_r,
        "test_metrics":  test_metrics,
        "history":       history,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {results_path}", log_file)

    log_file.close()
    return model, results


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train EEG2GAIT on the MoBI dataset")
    p.add_argument("--subjects",    nargs="+", default=None,
                   help="Subject IDs to include (e.g. SL01 SL02). Default: all 8.")
    p.add_argument("--sessions",    nargs="+", default=None,
                   help="Session IDs (e.g. T01 T02). Default: all 3.")
    p.add_argument("--device",      default="auto",
                   help="Device: cpu | cuda | mps | auto  (default: auto)")
    p.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--max-epochs",  type=int, default=MAX_EPOCHS)
    p.add_argument("--patience",    type=int, default=PATIENCE)
    p.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        subjects   = args.subjects,
        sessions   = args.sessions,
        device_str = args.device,
        batch_size = args.batch_size,
        lr         = args.lr,
        max_epochs = args.max_epochs,
        patience   = args.patience,
        output_dir = args.output_dir,
    )
'''[1:]


# ════════════════════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ════════════════════════════════════════════════════════════════════════════

DATA_PATH = "/kaggle/input/jwangldan09/eeg2gait-fall-prediction-dataset/RepositoryData"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def cc(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells = []

# 0 — Title
cells.append(md(
    "# EEG2GAIT — Hierarchical GCN for EEG-Based Gait Decoding  *(v2)*\n\n"
    "Full training pipeline on the MoBI dataset.  \n"
    "GPU accelerated · HTSR loss · early stopping on val Pearson *r*\n\n"
    "> **v2** — regenerated from authoritative `src/` files (June 2026).  \n"
    "> Architecture: K=2 HGP branches (depths [1,2]), depth-wise GSL, FFN [50,100,200].\n"
))

# 1 — Install deps
cells.append(cc(
    "import subprocess, sys\n"
    "subprocess.run(\n"
    "    [sys.executable, '-m', 'pip', 'install', '-q', 'scipy', 'scikit-learn'],\n"
    "    check=True\n"
    ")\n"
    "print('Dependencies ready.')"
))

# 2 — Verify data
cells.append(cc(
    "import os\n"
    "from pathlib import Path\n\n"
    f"DATA_DIR = Path('{DATA_PATH}')\n"
    "OUTPUT_DIR = Path('/kaggle/working/outputs')\n"
    "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\n"
    "if not DATA_DIR.exists():\n"
    "    print(f'ERROR: {DATA_DIR} does not exist.')\n"
    "    print('Contents of /kaggle/input:')\n"
    "    os.system('find /kaggle/input -maxdepth 3 -type d')\n"
    "else:\n"
    "    sessions = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])\n"
    "    print(f'Found {len(sessions)} session folders')\n"
    "    for s in sessions[:6]:\n"
    "        print(f'  {s.name}  eeg={(s/\"eeg.txt\").exists()}  joints={(s/\"joints.txt\").exists()}')\n"
    "    if len(sessions) > 6:\n"
    "        print(f'  ... and {len(sessions)-6} more')"
))

# 3–8 — Write source modules
MODULE_ORDER = ['config.py', 'dataset.py', 'model.py', 'loss.py', 'metrics.py', 'train.py']
for name in MODULE_ORDER:
    section = name.replace('.py', '').upper()
    cells.append(md(f"## {section} — `{name}`"))
    cells.append(cc(f"%%writefile /kaggle/working/{name}\n" + MODULES[name]))

# 9 — Path patch + device check
cells.append(md(
    "## Kaggle path patch\n\n"
    "Override `DATA_DIR` and `OUTPUT_DIR` in the imported config module."
))
cells.append(cc(
    "import sys\n"
    "sys.path.insert(0, '/kaggle/working')\n\n"
    "import config as cfg\n"
    "from pathlib import Path\n\n"
    f"cfg.DATA_DIR   = Path('{DATA_PATH}')\n"
    "cfg.OUTPUT_DIR = Path('/kaggle/working/outputs')\n"
    "cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\n"
    "import torch\n"
    "print('PyTorch version :', torch.__version__)\n"
    "print('CUDA available  :', torch.cuda.is_available())\n"
    "if torch.cuda.is_available():\n"
    "    print('GPU             :', torch.cuda.get_device_name(0))"
))

# 10 — Run training
cells.append(md(
    "## Run Training\n\n"
    "Trains all 8 subjects × 3 sessions.  \n"
    "Best checkpoint → `/kaggle/working/outputs/best_model.pt`"
))
cells.append(cc(
    "import torch\n"
    "from train import train\n\n"
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "print(f'Training on: {device}')\n\n"
    "model, results = train(\n"
    "    device_str  = device,\n"
    "    batch_size  = 100,\n"
    "    max_epochs  = 50,\n"
    "    patience    = 30,\n"
    "    output_dir  = '/kaggle/working/outputs',\n"
    ")"
))

# 11 — Results
cells.append(md("## Results"))
cells.append(cc(
    "import json\n\n"
    "with open('/kaggle/working/outputs/results.json') as f:\n"
    "    res = json.load(f)\n\n"
    "tm = res['test_metrics']\n"
    "print('='*50)\n"
    "print('TEST SET RESULTS')\n"
    "print('='*50)\n"
    "print(f\"Mean Pearson r : {tm['r_mean']:.4f}\")\n"
    "print(f\"Mean R\\u00b2        : {tm['r2_mean']:.4f}\")\n"
    "print(f\"Mean MAE       : {tm['mae_mean']:.4f}\")\n"
    "print('-'*50)\n"
    "joints = ['GHR','GKR','GAR','GHL','GKL','GAL']\n"
    "print(f\"{'Joint':6s}  {'r':>7s}  {'R2':>7s}  {'MAE':>7s}\")\n"
    "print('-'*34)\n"
    "for j in joints:\n"
    "    print(f\"{j:6s}  {tm[f'r_{j}']:7.4f}  {tm[f'r2_{j}']:7.4f}  {tm[f'mae_{j}']:7.4f}\")\n"
    "print('='*50)\n"
    "print(f\"Best epoch: {res['best_epoch']} | Best val r: {res['best_val_r']:.4f}\")"
))

# ── Assemble ────────────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"✓ Wrote {OUT}")
print(f"  Cells: {len(cells)}")
for i, c in enumerate(cells):
    preview = "".join(c["source"])[:70].replace("\n", " ")
    print(f"  [{i:02d}] {c['cell_type']:8s}  {preview}")
