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
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd

import sys
sys.path.insert(0, str(Path(__file__).parent))
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
    """Zero-phase Butterworth band-pass filter (0.1–48 Hz)."""
    nyq  = fs / 2.0
    lo   = BANDPASS_LO / nyq
    hi   = min(BANDPASS_HI / nyq, 0.999)   # must be < 1
    b, a = butter(4, [lo, hi], btype="band")
    # filtfilt expects (samples,) per channel
    filtered = np.empty_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = filtfilt(b, a, data[:, ch])
    return filtered


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

    # 2. Common-Average Reference
    eeg = _common_average_reference(eeg)

    # 3. Band-pass filter
    eeg = _bandpass_filter(eeg, fs)

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
    C = positions.shape[0]
    A = np.zeros((C, C), dtype=np.float32)
    for i in range(C):
        for j in range(C):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= radius:
                A[i, j] = 1.0
    # Self-loops
    np.fill_diagonal(A, 1.0)
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
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


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

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=False, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False)

    return train_dl, val_dl, test_dl
