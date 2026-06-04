"""
evaluate_gait_cycles.py
-----------------------
Plot Figure 3-style 3×2 gait-cycle grid from pre-saved Kaggle predictions.

Pre-requisite
-------------
Download  eeg2gait_predictions.npz  from your Kaggle notebook output and
place it at:
    outputs/eeg2gait_predictions.npz

The file must contain:
    all_meas  [W, 6]  – ground-truth window-mean joint angles (degrees)
    all_pred  [W, 6]  – model predictions (degrees)
    joint_names       – ["GHR","GKR","GAR","GHL","GKL","GAL"]

Pipeline
--------
1. Load all_meas / all_pred from .npz
2. Segment into gait cycles via Left Knee (GKL) peak detection
3. Interpolate each cycle to 400 time points
4. Compute mean ± std across cycles
5. Plot 3×2 Figure 3-style grid and save to outputs/gait_cycle_plot.png

Usage
-----
    python3 evaluate_gait_cycles.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
# Updated for Kaggle Notebooks
OUTPUT_DIR  = Path("/kaggle/working/outputs")
NPZ_PATH    = OUTPUT_DIR / "eeg2gait_predictions.npz"
PLOT_PATH   = OUTPUT_DIR / "gait_cycle_plot.png"

GAIT_CYCLE_LEN = 400      # time points per normalised gait cycle
STRIDE_SAMPS   = 10       # window stride in samples at 100 Hz (= 0.1 s)
TARGET_FS      = 100      # Hz

# JOINT_NAMES order from config: ["GHR","GKR","GAR","GHL","GKL","GAL"]
JOINT_NAMES = ["GHR", "GKR", "GAR", "GHL", "GKL", "GAL"]
JOINT_IDX   = {n: i for i, n in enumerate(JOINT_NAMES)}

# 3×2 grid layout (row-major): Left column = Left side, Right column = Right side
PLOT_ORDER = [
    ("GHL", "Left Hip"),
    ("GHR", "Right Hip"),
    ("GKL", "Left Knee"),
    ("GKR", "Right Knee"),
    ("GAL", "Left Ankle"),
    ("GAR", "Right Ankle"),
]

# Colour palette
C_MEAS_LINE  = "#1f77b4"    # solid blue
C_MEAS_SHADE = "#aec7e8"    # light blue
C_PRED_LINE  = "#d62728"    # dashed red
C_PRED_SHADE = "#f4a7a7"    # light red / salmon

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load predictions
# ─────────────────────────────────────────────────────────────────────────────
if not NPZ_PATH.exists():
    print(f"\n[ERROR] Predictions file not found:\n  {NPZ_PATH}")
    print("\nSteps to fix:")
    print("  1. Open  kaggle_save_predictions.py  and paste its contents")
    print("     as a new cell in your Kaggle notebook.")
    print("  2. Run the cell → it will produce  eeg2gait_predictions.npz")
    print("  3. Download that file from the Kaggle output panel.")
    print(f"  4. Place it at:  {NPZ_PATH}")
    print("  5. Re-run this script.\n")
    raise RuntimeError("Predictions file not found")

print(f"Loading predictions from:\n  {NPZ_PATH}")
data     = np.load(NPZ_PATH, allow_pickle=True)
all_meas = data["all_meas"].astype(np.float32)   # [W, 6]
all_pred = data["all_pred"].astype(np.float32)   # [W, 6]

print(f"  Measurements : {all_meas.shape}")
print(f"  Predictions  : {all_pred.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Gait-cycle segmentation via Left Knee (GKL) peak detection
# ─────────────────────────────────────────────────────────────────────────────
print("\nSegmenting gait cycles via GKL peak detection …")

gkl_signal = all_meas[:, JOINT_IDX["GKL"]]

# At 10 Hz effective resolution (stride = 0.1 s), a typical gait cycle
# (~1–2 s) spans ~10–20 samples.  min_distance = 0.5 s = 5 samples.
min_dist = max(5, int(0.5 * TARGET_FS / STRIDE_SAMPS))
peaks, _ = find_peaks(gkl_signal, distance=min_dist, prominence=1.0)

print(f"  GKL peaks detected: {len(peaks)}")

if len(peaks) < 2:
    print("  [WARNING] Too few peaks — retrying with looser parameters …")
    peaks, _ = find_peaks(gkl_signal, distance=3)
    print(f"  Retry → {len(peaks)} peaks")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Slice + normalise each gait cycle to GAIT_CYCLE_LEN points
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nNormalising each cycle to {GAIT_CYCLE_LEN} time points …")

x_norm       = np.linspace(0, 1, GAIT_CYCLE_LEN)
cycles_meas  = {k: [] for k, _ in PLOT_ORDER}
cycles_pred  = {k: [] for k, _ in PLOT_ORDER}

for k in range(len(peaks) - 1):
    s, e = peaks[k], peaks[k + 1]
    n    = e - s
    if n < 5:           # skip implausibly short segments
        continue
    x_orig = np.linspace(0, 1, n)

    for joint_key, _ in PLOT_ORDER:
        j = JOINT_IDX[joint_key]
        f_m = interp1d(x_orig, all_meas[s:e, j], kind="linear")
        f_p = interp1d(x_orig, all_pred[s:e, j], kind="linear")
        cycles_meas[joint_key].append(f_m(x_norm))
        cycles_pred[joint_key].append(f_p(x_norm))

n_cycles = len(cycles_meas["GKL"])
print(f"  Valid gait cycles: {n_cycles}")

if n_cycles == 0:
    print("[ERROR] No valid gait cycles found. "
          "Check peak detection on your GKL signal.")
    raise RuntimeError("No valid gait cycles found")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Aggregate: mean ± std, then normalise to [0, 1] per measurement range
# ─────────────────────────────────────────────────────────────────────────────
stats = {}
for joint_key, _ in PLOT_ORDER:
    arr_m = np.array(cycles_meas[joint_key])   # [N, 400]
    arr_p = np.array(cycles_pred[joint_key])

    mean_m = arr_m.mean(0)
    std_m  = arr_m.std(0)
    mean_p = arr_p.mean(0)
    std_p  = arr_p.std(0)

    # Min-max normalise both using measurement range → "Normalised Angle"
    lo, hi = mean_m.min(), mean_m.max()
    span   = (hi - lo) if (hi - lo) > 1e-6 else 1.0

    stats[joint_key] = {
        "mean_m": (mean_m - lo) / span,
        "std_m" : std_m  / span,
        "mean_p": (mean_p - lo) / span,
        "std_p" : std_p  / span,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot — Figure 3 style
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 3-style plot …")

fig, axes = plt.subplots(
    nrows=3, ncols=2,
    figsize=(10, 9),
    sharey=False,
)
fig.subplots_adjust(top=0.85, hspace=0.50, wspace=0.38)

t = np.arange(GAIT_CYCLE_LEN)   # 0 … 399

for ax, (joint_key, joint_label) in zip(axes.flatten(), PLOT_ORDER):
    s = stats[joint_key]

    # ── Measurement ───────────────────────────────────────────────────────
    ax.fill_between(t,
                    s["mean_m"] - s["std_m"],
                    s["mean_m"] + s["std_m"],
                    color=C_MEAS_SHADE, alpha=0.55, zorder=1)
    ax.plot(t, s["mean_m"],
            color=C_MEAS_LINE, linewidth=1.8, linestyle="-", zorder=3)

    # ── Prediction ────────────────────────────────────────────────────────
    ax.fill_between(t,
                    s["mean_p"] - s["std_p"],
                    s["mean_p"] + s["std_p"],
                    color=C_PRED_SHADE, alpha=0.45, zorder=2)
    ax.plot(t, s["mean_p"],
            color=C_PRED_LINE, linewidth=1.8, linestyle="--", zorder=4)

    # ── Formatting ────────────────────────────────────────────────────────
    ax.set_title(joint_label, fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel("Time Points", fontsize=9)
    ax.set_ylabel("Normalised Angle", fontsize=9)
    ax.set_xlim(0, GAIT_CYCLE_LEN - 1)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)

# ── Centralised legend above all subplots ────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=C_MEAS_LINE,  linewidth=2, linestyle="-",
           label="Measurement Mean"),
    mpatches.Patch(color=C_MEAS_SHADE, alpha=0.55,
                   label="Measurement Std Dev"),
    Line2D([0], [0], color=C_PRED_LINE,  linewidth=2, linestyle="--",
           label="Prediction Mean"),
    mpatches.Patch(color=C_PRED_SHADE, alpha=0.45,
                   label="Prediction Std Dev"),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=4,
    fontsize=9.5,
    frameon=True,
    bbox_to_anchor=(0.5, 0.975),
    framealpha=0.95,
    edgecolor="#cccccc",
)
fig.suptitle(
    "EEG2GAIT — Gait Cycle Joint Angles  (Mean ± Std, Test Set)",
    fontsize=13, fontweight="bold", y=1.0,
)

# ── Save ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"\n✅  Plot saved → {PLOT_PATH}")
