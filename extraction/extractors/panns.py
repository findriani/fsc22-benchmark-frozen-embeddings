"""
PANNs CNN14 embedding extractor.

Uses the panns_inference library to extract 2048-dim embeddings
from the CNN14 model pretrained on AudioSet.
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
    """Extract PANNs CNN14 embeddings for all audio files.

    Returns:
        np.ndarray of shape (N, 2048)
    """
    from panns_inference import AudioTagging

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioTagging(checkpoint_path=None, device=device)

    target_sr = cfg.EMBEDDING_MODELS["panns_cnn14"]["sample_rate"]  # 32000
    expected_len = int(target_sr * cfg.AUDIO_DURATION)

    embeddings = []
    for fpath in tqdm(file_paths, desc="PANNs CNN14"):
        y, _ = librosa.load(fpath, sr=target_sr, duration=cfg.AUDIO_DURATION, mono=True)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        # panns_inference expects (batch, samples) as float32
        waveform = y[np.newaxis, :].astype(np.float32)
        _, embedding = model.inference(waveform)
        embeddings.append(embedding.flatten())

    return np.array(embeddings, dtype=np.float32)
