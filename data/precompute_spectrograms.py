"""
Pre-compute log-mel spectrograms for all FSC22 clips and save as float32 .npy files.

Run once before CNN training:
    python data/precompute_spectrograms.py

This replaces on-the-fly librosa computation in the DataLoader with a fast
numpy load, giving 5-20x speedup on CNN training (see LESSONS_LEARNED.md §10).

Output: data/spectrograms/<stem>.npy, shape (224, 224), float32, range [0, 1].
"""

import json
import sys
from pathlib import Path

import numpy as np
import librosa
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def precompute_all():
    meta_file = cfg.SPLITS_DIR / "metadata.json"
    if not meta_file.exists():
        print(f"ERROR: {meta_file} not found. Run data/prepare_splits.py first.")
        sys.exit(1)

    with open(meta_file) as f:
        metadata = json.load(f)

    file_paths = metadata["files"]
    input_size = cfg.CNN_TRAINING["input_size"]  # (224, 224)

    spec_dir = cfg.DATA_DIR / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)

    total = len(file_paths)
    skipped = 0
    done = 0

    print(f"Pre-computing {total} spectrograms → {spec_dir}")
    print(f"Input size: {input_size}  |  n_mels=128  |  n_fft=2048  |  hop=512")

    for i, audio_path in enumerate(file_paths):
        audio_path = Path(audio_path)
        out_path = spec_dir / (audio_path.stem + ".npy")

        if out_path.exists():
            skipped += 1
            continue

        # Load audio
        y, _ = librosa.load(
            audio_path,
            sr=cfg.SAMPLE_RATE,
            duration=cfg.AUDIO_DURATION,
            mono=True,
        )

        # Pad / trim to exact length
        expected_len = int(cfg.SAMPLE_RATE * cfg.AUDIO_DURATION)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        # Compute log-mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=cfg.SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Normalise to [0, 1]
        lo, hi = log_mel.min(), log_mel.max()
        log_mel = (log_mel - lo) / (hi - lo + 1e-8)

        # Resize to CNN input size
        img = Image.fromarray((log_mel * 255).astype(np.uint8))
        img = img.resize(input_size, Image.BILINEAR)
        spec_array = np.array(img, dtype=np.float32) / 255.0  # (H, W) float32

        np.save(out_path, spec_array)
        done += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total}  saved={done}  skipped={skipped}")

    print(f"\nDone. {done} saved, {skipped} already existed.")
    print(f"Total .npy files: {len(list(spec_dir.glob('*.npy')))}")


if __name__ == "__main__":
    precompute_all()
