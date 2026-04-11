"""
CLAP (Contrastive Language-Audio Pretraining) embedding extractor.

Uses the LAION CLAP model to extract 512-dim audio embeddings.
"""

import numpy as np
import torch
import librosa
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg


def extract(file_paths):
    """Extract CLAP audio embeddings for all audio files.

    Returns:
        np.ndarray of shape (N, 512)
    """
    import laion_clap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use HTSAT-tiny — the default checkpoint downloaded by load_ckpt() is HTSAT-tiny.
    # HTSAT-base requires a different checkpoint; using mismatched arch causes a
    # size_mismatch RuntimeError at load time. Output projection is always 512-dim
    # regardless of HTSAT variant, so the embedding dimension is unaffected.
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    model.load_ckpt()

    target_sr = cfg.EMBEDDING_MODELS["clap"]["sample_rate"]  # 48000
    expected_len = int(target_sr * cfg.AUDIO_DURATION)

    embeddings = []
    batch_size = 32
    batch_audio = []
    batch_indices = []

    all_embeddings = [None] * len(file_paths)

    for i, fpath in enumerate(tqdm(file_paths, desc="CLAP")):
        y, _ = librosa.load(fpath, sr=target_sr, duration=cfg.AUDIO_DURATION, mono=True)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        batch_audio.append(y)
        batch_indices.append(i)

        if len(batch_audio) == batch_size or i == len(file_paths) - 1:
            audio_array = np.stack(batch_audio).astype(np.float32)
            with torch.no_grad():
                embs = model.get_audio_embedding_from_data(
                    x=audio_array, use_tensor=False
                )
            for j, idx in enumerate(batch_indices):
                all_embeddings[idx] = embs[j]
            batch_audio = []
            batch_indices = []

    return np.array(all_embeddings, dtype=np.float32)
