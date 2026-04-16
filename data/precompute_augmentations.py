"""
Precompute augmentation bank for FSC22 (one-time, before training).

Generates one pitch-shifted and one time-stretched copy per clip for all
2025 FSC22 files. Output: data/augmented/v1/pitch/{idx:05d}.wav
                           data/augmented/v1/time/{idx:05d}.wav

Design:
  - Bank covers ALL 2025 clips so any train split can draw from it.
  - Augmentation parameters are deterministic (selected by clip index),
    so the bank is reproducible across machines.
  - Script is resume-safe: skips files that already exist.
  - Uses Python multiprocessing (up to 16 workers) for parallelism.

Pitch parameters (cycled by index): [-2, -1, 1, 2] semitones
Time-stretch rates (cycled by index): [0.85, 0.90, 0.95, 1.05, 1.10, 1.15]

Expected runtime on RunPod (CPU workers):
  ~0.5-2 s/clip for librosa pitch_shift and time_stretch
  With 8 workers: roughly 10-20 min per bank, 20-40 min total.

Usage (RunPod):
    cd /workspace/fsc22
    python data/precompute_augmentations.py
"""

import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

PITCH_STEPS  = [-2, -1, 1, 2]
TIME_RATES   = [0.85, 0.90, 0.95, 1.05, 1.10, 1.15]
N_WORKERS    = min(16, max(1, cpu_count() - 1))
OUT_DIR      = cfg.DATA_DIR / "augmented" / "v1"
SR           = cfg.SAMPLE_RATE
EXPECTED_LEN = int(SR * cfg.AUDIO_DURATION)


def process_clip(args):
    idx, fpath, aug_type = args
    out_path = OUT_DIR / aug_type / f"{idx:05d}.wav"
    if out_path.exists():
        return idx, aug_type, "skipped"

    try:
        y, _ = librosa.load(fpath, sr=SR, duration=cfg.AUDIO_DURATION, mono=True)
        if len(y) < EXPECTED_LEN:
            y = np.pad(y, (0, EXPECTED_LEN - len(y)))
        else:
            y = y[:EXPECTED_LEN]

        if aug_type == "pitch":
            n_steps = PITCH_STEPS[idx % len(PITCH_STEPS)]
            y_aug = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
        else:
            rate = TIME_RATES[idx % len(TIME_RATES)]
            y_aug = librosa.effects.time_stretch(y, rate=rate)
            if len(y_aug) < EXPECTED_LEN:
                y_aug = np.pad(y_aug, (0, EXPECTED_LEN - len(y_aug)))
            else:
                y_aug = y_aug[:EXPECTED_LEN]

        sf.write(out_path, y_aug, SR)
        return idx, aug_type, "ok"

    except Exception as e:
        return idx, aug_type, f"error: {e}"


def main():
    meta_path = cfg.SPLITS_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run data/prepare_splits.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    file_paths = metadata["files"]
    n_clips = len(file_paths)

    print(f"FSC22: {n_clips} clips")
    print(f"Output: {OUT_DIR}")
    print(f"Workers: {N_WORKERS}")
    print(f"Pitch steps (by index mod 4): {PITCH_STEPS}")
    print(f"Time-stretch rates (by index mod 6): {TIME_RATES}")

    (OUT_DIR / "pitch").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "time").mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx, fpath in enumerate(file_paths):
        tasks.append((idx, fpath, "pitch"))
        tasks.append((idx, fpath, "time"))

    print(f"\nTotal tasks: {len(tasks)} ({n_clips} pitch + {n_clips} time)")
    print("Skipping files that already exist (resume-safe).\n")

    errors = []
    with Pool(N_WORKERS) as pool:
        for idx, aug_type, status in tqdm(
            pool.imap_unordered(process_clip, tasks), total=len(tasks)
        ):
            if status.startswith("error"):
                errors.append((idx, aug_type, status))

    print(f"\nDone. Errors: {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(f"  {e}")

    # Write manifest so training scripts can locate files by original index
    manifest = {
        "n_clips": n_clips,
        "pitch_steps": [PITCH_STEPS[i % len(PITCH_STEPS)] for i in range(n_clips)],
        "time_rates":  [TIME_RATES[i  % len(TIME_RATES)]  for i in range(n_clips)],
        "pitch": [str(OUT_DIR / "pitch" / f"{i:05d}.wav") for i in range(n_clips)],
        "time":  [str(OUT_DIR / "time"  / f"{i:05d}.wav") for i in range(n_clips)],
    }
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")
    print(f"\nBank size estimate: ~{n_clips * 2 * 0.44 / 1024:.1f} GB")


if __name__ == "__main__":
    main()
