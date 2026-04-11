"""
Log-mel spectrogram summary statistics extractor.

Computes a 128-bin log-mel spectrogram and summarizes each bin with
8 statistics (mean, std, min, max, median, skew, kurtosis, range),
yielding a 1024-dim feature vector.
"""

import numpy as np
import librosa
from scipy import stats as scipy_stats
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg


def extract(file_paths):
    """Extract log-mel summary statistics for all audio files.

    Returns:
        np.ndarray of shape (N, 1024)  — 128 mel bins x 8 statistics
    """
    n_mels = 128
    expected_len = int(cfg.SAMPLE_RATE * cfg.AUDIO_DURATION)

    embeddings = []
    for fpath in tqdm(file_paths, desc="Log-mel stats"):
        y, _ = librosa.load(fpath, sr=cfg.SAMPLE_RATE, duration=cfg.AUDIO_DURATION, mono=True)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=cfg.SAMPLE_RATE, n_mels=n_mels, n_fft=2048, hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        # log_mel shape: (n_mels, T)

        feature_vec = []
        for i in range(n_mels):
            band = log_mel[i]
            # Skew/kurtosis are undefined for constant bands; return 0 to avoid
            # RuntimeWarning from catastrophic cancellation in scipy moment calc
            is_constant = np.std(band) < 1e-8
            feature_vec.extend([
                np.mean(band),
                np.std(band),
                np.min(band),
                np.max(band),
                np.median(band),
                0.0 if is_constant else float(scipy_stats.skew(band)),
                0.0 if is_constant else float(scipy_stats.kurtosis(band)),
                np.max(band) - np.min(band),
            ])

        embeddings.append(np.array(feature_vec, dtype=np.float32))

    return np.array(embeddings, dtype=np.float32)
