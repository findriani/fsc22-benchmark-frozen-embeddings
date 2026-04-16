"""
Download FSC22 dataset.

FSC22 (Forest Sound Classification 2022) is hosted on Kaggle:
  https://www.kaggle.com/datasets/irmiot22/fsc22-dataset

This script downloads via the Kaggle API. Requires kaggle credentials:
  1. Get your API token from kaggle.com → Profile → Settings → API → Create New Token
  2. Place kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
  3. Run: python data/download_fsc22.py

Alternatively, download manually and set:
  export FSC22_AUDIO_DIR=/path/to/FSC22
"""

import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

KAGGLE_DATASET = "irmiot22/fsc22-dataset"


def download_fsc22():
    """Download and extract FSC22 dataset via Kaggle API."""
    audio_dir = cfg.FSC22_AUDIO_DIR
    if audio_dir.exists() and any(audio_dir.iterdir()):
        print(f"FSC22 already exists at {audio_dir}")
        n_classes = sum(1 for d in audio_dir.iterdir() if d.is_dir())
        n_files = sum(1 for d in audio_dir.iterdir() if d.is_dir() for f in d.glob("*.wav"))
        print(f"  {n_classes} classes, {n_files} audio files")
        return

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print("\nTo set up:")
        print("  1. Go to kaggle.com → Profile → Settings → API → Create New Token")
        print("  2. Upload kaggle.json to your machine")
        print("  3. Run: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
        print("\nOr download manually and set: export FSC22_AUDIO_DIR=/path/to/FSC22")
        sys.exit(1)

    # Ensure kaggle package is available
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

    print(f"Downloading FSC22 from Kaggle ({KAGGLE_DATASET})...")
    subprocess.check_call([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(cfg.DATA_DIR),
    ])

    # Find the downloaded zip
    zip_candidates = list(cfg.DATA_DIR.glob("*.zip"))
    if not zip_candidates:
        print("ERROR: No zip file found after download.")
        sys.exit(1)
    zip_path = zip_candidates[0]

    print(f"Extracting {zip_path.name} to {cfg.DATA_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cfg.DATA_DIR)
    zip_path.unlink()
    print("Extraction complete.")

    # FSC22 may extract as a subfolder — find it
    if not audio_dir.exists():
        # Look for a folder containing class subdirectories with .wav files
        for candidate in cfg.DATA_DIR.iterdir():
            if candidate.is_dir() and any(candidate.glob("*/*.wav")):
                print(f"Found FSC22 data at {candidate}")
                print(f"Set: export FSC22_AUDIO_DIR={candidate}")
                audio_dir = candidate
                break
        else:
            print("WARNING: Could not auto-detect FSC22 folder structure.")
            print(f"Check contents of {cfg.DATA_DIR} and set FSC22_AUDIO_DIR accordingly.")
            return

    n_classes = sum(1 for d in audio_dir.iterdir() if d.is_dir())
    n_files = sum(1 for d in audio_dir.iterdir() if d.is_dir() for f in d.glob("*.wav"))
    print(f"FSC22 ready: {n_classes} classes, {n_files} audio files")


if __name__ == "__main__":
    download_fsc22()
