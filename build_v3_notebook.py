"""
Build a Kaggle-ready Jupyter notebook for the v3 model.

Mirrors the structure of the existing eeg2gait.ipynb:
  - %%writefile cells for each src/*.py
  - A run cell that imports + trains
  - A results display cell

The config.py cell is rewritten with Kaggle paths baked in. All other source
files are taken verbatim from the v3 branch.
"""

import json
from pathlib import Path

SRC   = Path("src")
OUT   = Path("eeg2gait_v3.ipynb")
KGGL_DATA = "/kaggle/input/datasets/jwangldan09/eeg2gait-fall-prediction-dataset/RepositoryData"
KGGL_WORK = "/kaggle/working"


def read(p):
    return Path(p).read_text()


# ── config.py rewritten with Kaggle paths ─────────────────────────────────────
CONFIG_BODY = f'''%%writefile {KGGL_WORK}/config.py
"""
config.py — v3 (Kaggle)
"""
from pathlib import Path

# ── Paths (Kaggle) ───────────────────────────────────────────────────────────
ROOT_DIR   = Path("/kaggle")
DATA_DIR   = Path("{KGGL_DATA}")
OUTPUT_DIR = Path("{KGGL_WORK}/outputs")
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass   # ignore if read-only (e.g. on local test runs)

# ── Dataset ──────────────────────────────────────────────────────────────────
SUBJECTS = [f"SL{{i:02d}}" for i in range(1, 9)]
SESSIONS = ["T01", "T02", "T03"]

RAW_FS   = 333.33
TARGET_FS = 100
BANDPASS_LO = 0.1
BANDPASS_HI = 48.0
N_CHANNELS_RAW  = 64
EOG_CHAN_INDICES = [32, 38, 39, 62, 63]
N_CHANNELS       = 59
RADIUS_MM = 30.0

JOINT_NAMES = ["GHR", "GKR", "GAR", "GHL", "GKL", "GAL"]
N_JOINTS    = 6

TRAIN_MIN   = 13.5
VAL_MIN     = 1.5
TEST_MIN    = 5.0

# ── v3: 2-second window (was 1 s) — 2 gait cycles of context per sample
WINDOW_SECS  = 2.0
STRIDE_SECS  = 0.1
WINDOW_SAMPS = int(WINDOW_SECS  * TARGET_FS)   # 200 samples
STRIDE_SAMPS = int(STRIDE_SECS * TARGET_FS)    # 10  samples

# ── Model ────────────────────────────────────────────────────────────────────
F_FILTERS       = 25
LTL_KERNEL      = 10
# v3: 3 graph branches (was 2) — more diverse spatial receptive fields
HGP_DEPTHS      = [1, 2, 3]
DROPOUT_P       = 0.5
POOL_WIDTH      = 3
KERNEL_WIDTH    = 10
FFN_FILTERS     = [50, 100, 200]
GTL_HEADS       = 4
GTL_DROPOUT     = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 100
LR             = 1e-3
MAX_EPOCHS     = 50
PATIENCE       = 20

# v3 additions
WEIGHT_DECAY    = 1e-4
LR_WARMUP_EPOCHS = 5

# ── Loss ─────────────────────────────────────────────────────────────────────
ALPHA   = 0.5
BETA    = 0.1
EPSILON = 1e-8

# v3: up-weight knees (the per-joint bottleneck in the master baseline)
#                    GHR  GKR  GAR  GHL  GKL  GAL
JOINT_LOSS_WEIGHTS = [1.0, 1.5, 1.0, 1.0, 1.5, 1.0]

# ── v3: train-only EEG augmentation ──────────────────────────────────────────
AUG_CHANNEL_DROPOUT = 0.10
AUG_NOISE_STD       = 0.02
AUG_TIME_JITTER     = 5

# ── Misc ─────────────────────────────────────────────────────────────────────
SEED        = 42
NUM_WORKERS = 0
DEVICE      = "auto"
'''


def writefile_cell(path_in_kaggle, body):
    return f"%%writefile {path_in_kaggle}\n{body}"


def make_code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.split("\n") if isinstance(src, str) else src}


def make_md_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.split("\n") if isinstance(text, str) else text}


# Source bodies (verbatim from v3 except config which is rewritten above)
dataset_body = read(SRC / "dataset.py")
model_body   = read(SRC / "model.py")
loss_body    = read(SRC / "loss.py")
metrics_body = read(SRC / "metrics.py")
train_body   = read(SRC / "train.py")

# Build cells
cells = []

cells.append(make_md_cell("# EEG2GAIT v3 — Improved Hierarchical GCN for EEG-Based Gait Decoding\n\n"
                          "**v3 changes over master:**\n"
                          "1. Window 1 s → 2 s (more temporal context per sample)\n"
                          "2. HGP K=2 → K=3 graph branches\n"
                          "3. Per-joint loss weights (knees ×1.5 — the master bottleneck)\n"
                          "4. Train-only EEG augmentation (channel dropout / Gaussian noise / temporal jitter)\n"
                          "5. AdamW + linear-warmup + cosine LR schedule, 80 epochs / patience 50"))

cells.append(make_code_cell("# Verify deps (all preinstalled on Kaggle — no pip install needed)\n"
                            "import torch, numpy, scipy\n"
                            "print(f'torch {torch.__version__} | numpy {numpy.__version__} | scipy {scipy.__version__}')"))

cells.append(make_code_cell("import os\nfrom pathlib import Path\n\n"
                            f"WORK = Path('{KGGL_WORK}')\nWORK.mkdir(parents=True, exist_ok=True)\n"
                            "os.chdir(str(WORK))\nprint('Working dir:', os.getcwd())"))

cells.append(make_md_cell("## config.py"))
cells.append(make_code_cell(CONFIG_BODY))

cells.append(make_md_cell("## dataset.py"))
cells.append(make_code_cell(writefile_cell(f"{KGGL_WORK}/dataset.py", dataset_body)))

cells.append(make_md_cell("## model.py"))
cells.append(make_code_cell(writefile_cell(f"{KGGL_WORK}/model.py", model_body)))

cells.append(make_md_cell("## loss.py"))
cells.append(make_code_cell(writefile_cell(f"{KGGL_WORK}/loss.py", loss_body)))

cells.append(make_md_cell("## metrics.py"))
cells.append(make_code_cell(writefile_cell(f"{KGGL_WORK}/metrics.py", metrics_body)))

cells.append(make_md_cell("## train.py"))
cells.append(make_code_cell(writefile_cell(f"{KGGL_WORK}/train.py", train_body)))

cells.append(make_md_cell("## Run training"))
cells.append(make_code_cell(
    "import sys\n"
    f"sys.path.insert(0, '{KGGL_WORK}')\n\n"
    "import importlib, config as cfg\n"
    "from pathlib import Path\n"
    f"cfg.DATA_DIR   = Path('{KGGL_DATA}')\n"
    f"cfg.OUTPUT_DIR = Path('{KGGL_WORK}/outputs')\n"
    "cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\n"
    "import torch\n"
    "print('CUDA available:', torch.cuda.is_available())\n"
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "print('Using device:', device)\n\n"
    "from train import train\n"
    "from config import BATCH_SIZE   # v3-speed: 256 instead of the old hardcoded 100\n"
    "model, results = train(\n"
    "    device_str  = device,\n"
    "    batch_size  = BATCH_SIZE,\n"
    "    max_epochs  = 50,\n"
    "    patience    = 20,\n"
    f"    output_dir  = '{KGGL_WORK}/outputs',\n"
    ")"
))

cells.append(make_md_cell("## Results"))
cells.append(make_code_cell(
    "import json\n"
    f"with open('{KGGL_WORK}/outputs/results.json') as f:\n"
    "    res = json.load(f)\n\n"
    "tm = res['test_metrics']\n"
    "print(f\"Best epoch : {res['best_epoch']}\")\n"
    "print(f\"Best val r : {res['best_val_r']:.4f}\")\n"
    "print()\n"
    "print('=== TEST SET RESULTS (v3) ===')\n"
    "print(f\"  Mean Pearson r : {tm['r_mean']:.4f}\")\n"
    "print(f\"  Mean R\\u00b2        : {tm['r2_mean']:.4f}\")\n"
    "print(f\"  Mean MAE       : {tm['mae_mean']:.4f}\")\n"
    "print()\n"
    "joints = ['GHR','GKR','GAR','GHL','GKL','GAL']\n"
    "print(f\"{'Joint':<6} {'r':>8} {'R\\u00b2':>8} {'MAE':>8}\")\n"
    "print('-'*34)\n"
    "for j in joints:\n"
    "    print(f\"{j:<6} {tm[f'r_{j}']:>8.4f} {tm[f'r2_{j}']:>8.4f} {tm[f'mae_{j}']:>8.4f}\")\n"
    "print()\n"
    "print('=== Compared to master baseline (your earlier run) ===')\n"
    "print('Master mean r=0.8366, R\\u00b2=0.6915, MAE=2.5479')\n"
))

# Convert source lists: Jupyter expects each list element to end with \n except the last
def fix_source(cells):
    for c in cells:
        s = c.get("source", [])
        if isinstance(s, str):
            s = s.split("\n")
        # rejoin with \n then split keeping line endings except last
        text = "\n".join(s)
        lines = text.split("\n")
        out = []
        for i, ln in enumerate(lines):
            out.append(ln + ("\n" if i < len(lines) - 1 else ""))
        c["source"] = out
    return cells


cells = fix_source(cells)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language":     "python",
            "name":         "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat":       4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT}  ({OUT.stat().st_size:,} bytes, {len(cells)} cells)")
