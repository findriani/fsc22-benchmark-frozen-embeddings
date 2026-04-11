"""
MFCC feature extractor.

Extracts 40 MFCCs and computes summary statistics (mean, std, min, max,
median, skew, kurtosis, range) per coefficient, yielding a 320-dim vector.
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
    """Extract MFCC summary statistics for all audio files.

    Returns:
        np.ndarray of shape (N, 320)  — 40 MFCCs x 8 statistics
    """
    n_mfcc = 40
    expected_len = int(cfg.SAMPLE_RATE * cfg.AUDIO_DURATION)

    embeddings = []
    for fpath in tqdm(file_paths, desc="MFCC"):
        y, _ = librosa.load(fpath, sr=cfg.SAMPLE_RATE, duration=cfg.AUDIO_DURATION, mono=True)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        mfccs = librosa.feature.mfcc(y=y, sr=cfg.SAMPLE_RATE, n_mfcc=n_mfcc)
        # mfccs shape: (n_mfcc, T)

        feature_vec = []
        for i in range(n_mfcc):
            coeff = mfccs[i]
            is_constant = np.std(coeff) < 1e-8
            feature_vec.extend([
                np.mean(coeff),
                np.std(coeff),
                np.min(coeff),
                np.max(coeff),
                np.median(coeff),
                0.0 if is_constant else float(scipy_stats.skew(coeff)),
                0.0 if is_constant else float(scipy_stats.kurtosis(coeff)),
                np.max(coeff) - np.min(coeff),
            ])

        embeddings.append(np.array(feature_vec, dtype=np.float32))

    return np.array(embeddings, dtype=np.float32)
