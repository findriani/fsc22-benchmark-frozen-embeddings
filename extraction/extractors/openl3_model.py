"""
OpenL3 embedding extractor.

Uses the openl3 library to extract 6144-dim embeddings from the
OpenL3 model trained with audio-visual self-supervision on AudioSet.
"""

import numpy as np
import librosa
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg


def extract(file_paths):
    """Extract OpenL3 embeddings for all audio files.

    Returns:
        np.ndarray of shape (N, 6144)
    """
    # Force TF to CPU before openl3 (and its internal TF import) initializes.
    # The env var CUDA_VISIBLE_DEVICES is unreliable once PyTorch has loaded CUDA
    # into the process. TF's own API is the correct handle.
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import openl3

    target_sr = cfg.EMBEDDING_MODELS["openl3"]["sample_rate"]  # 48000
    expected_len = int(target_sr * cfg.AUDIO_DURATION)

    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type="env",  # environmental sounds
        embedding_size=6144,
    )

    embeddings = []
    for fpath in tqdm(file_paths, desc="OpenL3"):
        y, _ = librosa.load(fpath, sr=target_sr, duration=cfg.AUDIO_DURATION, mono=True)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        emb, _ = openl3.get_audio_embedding(
            y, target_sr,
            model=model,
            input_repr="mel256",
            content_type="env",
            embedding_size=6144,
            hop_size=cfg.AUDIO_DURATION,  # single embedding per clip
            center=False,
        )
        # Mean-pool if multiple frames returned
        emb = emb.mean(axis=0)
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)
