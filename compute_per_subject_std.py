"""
compute_per_subject_std.py
--------------------------
Slices eeg2gait_predictions.npz back by subject/session,
computes per-subject mean Pearson r, then reports mean ± std
across subjects (the ± numbers that match the paper's Table I format).

Usage:
    python3 compute_per_subject_std.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent
NPZ_PATH   = Path("/Users/jigmetwangldan/Downloads/eeg2gait_predictions.npz")
DATA_DIR   = REPO_ROOT / "RepositoryData"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config (must match training) ──────────────────────────────────────────────
SUBJECTS        = [f"SL{i:02d}" for i in range(1, 9)]
SESSIONS        = ["T01", "T02", "T03"]
JOINT_NAMES     = ["GHR", "GKR", "GAR", "GHL", "GKL", "GAL"]
TARGET_FS       = 100
RAW_FS          = 333.33
BANDPASS_LO     = 0.1
BANDPASS_HI     = 48.0
N_CHANNELS_RAW  = 64
EOG_CHAN_INDICES = [32, 38, 39, 62, 63]
TRAIN_MIN       = 13.5
VAL_MIN         = 1.5
TEST_MIN        = 5.0
WINDOW_SAMPS    = int(1.0  * TARGET_FS)   # 100
STRIDE_SAMPS    = int(0.1  * TARGET_FS)   # 10

# ── Load predictions ──────────────────────────────────────────────────────────
print(f"Loading: {NPZ_PATH}")
data     = np.load(NPZ_PATH, allow_pickle=True)
all_meas = data["all_meas"].astype(np.float32)   # [W_total, 6]
all_pred = data["all_pred"].astype(np.float32)
print(f"  Total windows in .npz: {len(all_meas)}")

# ── Figure out how many test windows each session contributed ─────────────────
# We replay the exact same logic as dataset.py to count windows per session.

def _count_test_windows(session_folder: Path) -> int:
    """Re-run preprocessing logic just to count windows, without loading EEG."""
    eeg_path    = session_folder / "eeg.txt"
    joints_path = session_folder / "joints.txt"
    if not eeg_path.exists() or not joints_path.exists():
        return 0

    # Read timestamps only (first column) to infer fs and length
    with open(eeg_path, "r", errors="replace") as f:
        f.readline()  # skip header
        ts = []
        for line in f:
            vals = line.strip().split("\t")
            try:
                ts.append(float(vals[0]))
            except (ValueError, IndexError):
                continue
            if len(ts) > 600:  # only need first ~600 to infer fs & length
                break

    if len(ts) < 10:
        return 0

    # Count total raw rows (need actual length)
    raw_n = 0
    with open(eeg_path, "r", errors="replace") as f:
        f.readline()
        for line in f:
            if line.strip():
                raw_n += 1

    dt = float(np.median(np.diff(ts[:500])))
    fs = 1.0 / dt

    # After resampling: approximate resampled length
    fs_in_int  = int(round(fs * 3))
    fs_out_int = int(round(TARGET_FS * 3))
    g          = gcd(fs_in_int, fs_out_int)
    up, down   = fs_out_int // g, fs_in_int // g
    m = round(raw_n * up / down)

    # Split boundaries
    train_samps = int(TRAIN_MIN * 60 * TARGET_FS)
    val_samps   = int(VAL_MIN   * 60 * TARGET_FS)
    test_end    = min(m, train_samps + val_samps + int(TEST_MIN * 60 * TARGET_FS))
    s           = train_samps + val_samps
    test_len    = max(0, test_end - s)

    # Window count
    n_windows = max(0, (test_len - WINDOW_SAMPS) // STRIDE_SAMPS + 1)
    return n_windows


print("\nCounting test windows per session (replaying dataset logic) …")
session_info = []   # list of (subject, n_windows_for_this_subject_across_sessions)
for subj in SUBJECTS:
    subj_windows = 0
    for sess in SESSIONS:
        folder = DATA_DIR / f"{subj}-{sess}"
        n = _count_test_windows(folder)
        print(f"  {subj}-{sess}: {n} test windows")
        subj_windows += n
    session_info.append((subj, subj_windows))
    print(f"  → {subj} total: {subj_windows}")

total_counted = sum(w for _, w in session_info)
print(f"\nTotal windows counted: {total_counted}  (npz has: {len(all_meas)})")

if total_counted != len(all_meas):
    print("\n⚠ WARNING: Mismatch! Some sessions may be missing from the data dir.")
    print("  Std dev will still be computed but window boundaries may be off.\n")

# ── Slice predictions per subject and compute per-subject Pearson r ───────────
print("\nComputing per-subject Pearson r …")
subject_results = {}
offset = 0
for subj, n_win in session_info:
    if n_win == 0:
        print(f"  {subj}: skipped (0 windows)")
        continue

    meas_s = all_meas[offset : offset + n_win]   # [n_win, 6]
    pred_s = all_pred[offset : offset + n_win]
    offset += n_win

    joint_r = {}
    for j, jname in enumerate(JOINT_NAMES):
        m = meas_s[:, j]
        p = pred_s[:, j]
        if m.std() < 1e-6 or p.std() < 1e-6:
            r = 0.0
        else:
            r, _ = pearsonr(m, p)
        joint_r[jname] = float(r)

    mean_r  = float(np.mean(list(joint_r.values())))
    subject_results[subj] = {**joint_r, "mean_r": mean_r}
    print(f"  {subj}: r = {mean_r:.4f}  "
          f"({', '.join(f'{k}={v:.3f}' for k,v in joint_r.items())})")

# ── Aggregate: mean ± std across subjects ─────────────────────────────────────
print("\n" + "="*60)
print("MEAN ± STD ACROSS SUBJECTS")
print("="*60)

all_subject_means = np.array([v["mean_r"] for v in subject_results.values()])
grand_mean = all_subject_means.mean()
grand_std  = all_subject_means.std()
print(f"  Overall Pearson r : {grand_mean:.4f} ± {grand_std:.4f}")

print("\n  Per-joint breakdown:")
for jname in JOINT_NAMES:
    vals = np.array([subject_results[s][jname]
                     for s in subject_results if jname in subject_results[s]])
    print(f"    {jname:5s}: {vals.mean():.4f} ± {vals.std():.4f}  (n={len(vals)})")

# ── Save results ──────────────────────────────────────────────────────────────
out = {
    "per_subject":   subject_results,
    "summary": {
        "mean_r":  grand_mean,
        "std_r":   grand_std,
        "per_joint": {
            jname: {
                "mean": float(np.array([subject_results[s][jname]
                                         for s in subject_results
                                         if jname in subject_results[s]]).mean()),
                "std":  float(np.array([subject_results[s][jname]
                                         for s in subject_results
                                         if jname in subject_results[s]]).std()),
            }
            for jname in JOINT_NAMES
        }
    }
}
save_path = OUTPUT_DIR / "per_subject_results.json"
with open(save_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n✅  Full results saved → {save_path}")
