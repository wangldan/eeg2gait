"""
kaggle_save_predictions.py
--------------------------
Run this as a STANDALONE cell in your Kaggle notebook.
DO NOT re-run the training cell — best_model.pt already exists.

Steps:
  1. Loads best_model.pt (handles both raw state_dict AND the checkpoint
     dict format saved by train.py: {"epoch":…, "state_dict":…, …})
  2. Loads the MoBI test split
  3. Runs inference — no optimizer, no training
  4. Saves /kaggle/working/eeg2gait_predictions.npz

After downloading the .npz, place it at:
    outputs/eeg2gait_predictions.npz
and run locally:
    python3 evaluate_gait_cycles.py
"""

# ── Fix: pre-import torch._utils before _dynamo lazily touches it ─────────────
# Kaggle's PyTorch build (post-2.5) has a bug where torch._dynamo tries to
# access torch._utils as a lazy attribute, which fails. Importing it first fixes it.
import torch
try:
    import torch._utils  # noqa: F401
except Exception:
    pass

import sys
import os
import numpy as np
from torch.utils.data import DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKING_DIR = "/kaggle/working"
MODEL_PATH  = f"{WORKING_DIR}/best_model.pt"
SAVE_PATH   = f"{WORKING_DIR}/eeg2gait_predictions.npz"
sys.path.insert(0, WORKING_DIR)

# ── Project imports ───────────────────────────────────────────────────────────
from config  import SUBJECTS, SESSIONS, DATA_DIR, BATCH_SIZE, JOINT_NAMES
from dataset import MoBIDataset, build_adjacency_matrix
from model   import build_model

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Sanity check ──────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nbest_model.pt not found at {MODEL_PATH}\n"
        "The Kaggle session may have been reset. Re-run your training cell first."
    )
print(f"Model checkpoint: {MODEL_PATH}  ({os.path.getsize(MODEL_PATH)/1e6:.1f} MB)")

# ── Load test dataset ─────────────────────────────────────────────────────────
print("\nLoading MoBI test dataset ...")
test_ds = MoBIDataset(
    subjects=SUBJECTS,
    sessions=SESSIONS,
    split="test",
    data_dir=DATA_DIR,
    verbose=True,
)
print(f"Test windows: {len(test_ds)}")

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,       # ORDER MUST be preserved for time-series reconstruction
    num_workers=0,
    drop_last=False,
)

# ── Build model ───────────────────────────────────────────────────────────────
print("\nBuilding model and loading weights ...")
A_init = torch.tensor(build_adjacency_matrix(), dtype=torch.float32)
model  = build_model(A_init=A_init).to(DEVICE)

# train.py saves: {"epoch": int, "state_dict": OrderedDict, "val_r": float, …}
# Handle both that format and a raw state_dict file
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    print(f"  Checkpoint: epoch {checkpoint.get('epoch','?')}, "
          f"val_r={checkpoint.get('val_r', '?'):.4f}")
else:
    state_dict = checkpoint      # raw state_dict (older save format)
    print("  Checkpoint: raw state_dict format")

model.load_state_dict(state_dict)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# ── Inference ─────────────────────────────────────────────────────────────────
print(f"\nRunning inference over {len(test_loader)} batches ...")
all_meas, all_pred = [], []

with torch.no_grad():
    for i, (X_batch, y_batch) in enumerate(test_loader):
        pred = model(X_batch.to(DEVICE))       # [B, 6]
        all_meas.append(y_batch.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(test_loader)}", flush=True)

all_meas = np.concatenate(all_meas, axis=0)   # [W, 6]
all_pred = np.concatenate(all_pred, axis=0)   # [W, 6]
print(f"\nInference complete:")
print(f"  Measurements : {all_meas.shape}")
print(f"  Predictions  : {all_pred.shape}")

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez_compressed(
    SAVE_PATH,
    all_meas    = all_meas,
    all_pred    = all_pred,
    joint_names = np.array(JOINT_NAMES),   # ["GHR","GKR","GAR","GHL","GKL","GAL"]
)
size_mb = os.path.getsize(SAVE_PATH) / 1e6
print(f"\n✅  Saved: {SAVE_PATH}  ({size_mb:.1f} MB)")
print("   → Go to Kaggle Output panel (right sidebar) → download eeg2gait_predictions.npz")
print("   → Place at:  outputs/eeg2gait_predictions.npz")
print("   → Then run:  python3 evaluate_gait_cycles.py")
