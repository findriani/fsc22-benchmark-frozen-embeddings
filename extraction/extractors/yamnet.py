"""
YAMNet embedding extractor.

Uses TensorFlow Hub YAMNet model to extract 1024-dim embeddings.
"""

import numpy as np
import librosa
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg


def extract(file_paths):
    """Extract YAMNet embeddings for all audio files.

    Returns:
        np.ndarray of shape (N, 1024)
    """
    import tensorflow as tf
    # Force CPU — TF binary compiled with CuDNN 9.3, runtime has 9.1.
    # Must be called before any TF operation (including hub.load).
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_hub as hub

    model = hub.load("https://tfhub.dev/google/yamnet/1")

    target_sr = cfg.EMBEDDING_MODELS["yamnet"]["sample_rate"]  # 16000
    expected_len = int(target_sr * cfg.AUDIO_DURATION)

    embeddings = []
    for fpath in tqdm(file_paths, desc="YAMNet"):
        y, _ = librosa.load(fpath, sr=target_sr, duration=cfg.AUDIO_DURATION, mono=True)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        waveform = tf.constant(y, dtype=tf.float32)
        scores, embeddings_out, spectrogram = model(waveform)
        # Mean-pool over time frames
        emb = tf.reduce_mean(embeddings_out, axis=0).numpy()
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)
