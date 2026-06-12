"""
train.py - YOLOv8 Training Script for Parking Space Occupancy Detection.
Reads dataset/data.yaml, trains YOLOv8n, and saves best.pt to models/.
"""

import os
import sys
import shutil
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_YAML   = Path("dataset/data.yaml").resolve()  # absolute path — required by Ultralytics
MODEL_DIR   = Path("models")
RUNS_DIR    = Path("runs")
RUN_NAME    = "parking_detector"
BASE_WEIGHTS = Path("models/yolov8n.pt")   # local copy (paste here to avoid download)

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS     = 50
IMGSZ      = 640
BATCH      = 16
LR0        = 0.001
PATIENCE   = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_data_yaml(yaml_path: Path) -> dict:
    """Parse data.yaml and return its contents as a dict."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU detected: {gpu_name}")
        return "cuda"
    print("  No GPU found — training on CPU (will be slower).")
    return "cpu"


def find_best_pt(runs_dir: Path, run_name: str) -> Path | None:
    """
    Locate the best.pt produced by training.
    Handles exist_ok=True which may append numbers: parking_detector, parking_detector2 …
    """
    candidates = sorted(runs_dir.glob(f"{run_name}*/weights/best.pt"), reverse=True)
    if candidates:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  YOLOv8 Parking Space Occupancy Detector — Training")
    print("=" * 70)

    # ── 1. Validate data.yaml ────────────────────────────────────────────
    if not DATA_YAML.exists():
        print(f"\n✗ data.yaml not found at: {DATA_YAML}")
        print("  Place your Roboflow dataset inside the 'dataset/' folder.")
        sys.exit(1)

    cfg = read_data_yaml(DATA_YAML)
    nc          = cfg.get("nc", "?")
    class_names = cfg.get("names", [])
    print(f"\n✓ Dataset config: {DATA_YAML}")
    print(f"  Classes ({nc}): {class_names}")

    # ── 2. Pick base weights ─────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if BASE_WEIGHTS.exists():
        weights = str(BASE_WEIGHTS)
        print(f"\n✓ Using local base weights: {weights}")
    else:
        # YOLO will auto-download yolov8n.pt on first use
        weights = "yolov8n.pt"
        print(f"\n  Base weights not found locally — will auto-download '{weights}'.")

    # ── 3. Load model ────────────────────────────────────────────────────
    print(f"\n  Loading model: {weights} ...")
    model = YOLO(weights)

    # ── 4. Detect device ─────────────────────────────────────────────────
    device = get_device()

    # ── 5. Train ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Starting training …")
    print(f"{'='*70}\n")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=LR0,
        patience=PATIENCE,
        device=device,
        save=True,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    # ── 6. Copy best.pt → models/ ────────────────────────────────────────
    best_pt = find_best_pt(RUNS_DIR, RUN_NAME)
    if best_pt and best_pt.exists():
        dest = MODEL_DIR / "best.pt"
        shutil.copy(str(best_pt), str(dest))
        print(f"\n✅ Model saved to {dest}")
    else:
        print("\n⚠  Could not find best.pt in runs/ — check training output.")
        sys.exit(1)

    # ── 7. Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Training Complete!")
    print("=" * 70)

    # Extract metrics safely
    try:
        metrics = results.results_dict
        map50    = metrics.get("metrics/mAP50(B)",    metrics.get("metrics/mAP_50",    "N/A"))
        map5095  = metrics.get("metrics/mAP50-95(B)", metrics.get("metrics/mAP_50-95", "N/A"))
        if isinstance(map50,   float): map50   = f"{map50:.4f}"
        if isinstance(map5095, float): map5095 = f"{map5095:.4f}"
        print(f"  Epochs       : {EPOCHS}")
        print(f"  mAP@50       : {map50}")
        print(f"  mAP@50-95    : {map5095}")
    except Exception:
        print("  (Metrics not available — check runs/ folder for details.)")

    print(f"\n  Best weights : {MODEL_DIR / 'best.pt'}")
    print(f"  Run folder   : {RUNS_DIR / RUN_NAME}")
    print("\n  Next step → run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
