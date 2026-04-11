"""
VGGish embedding extractor.

Uses the torchvggish library to extract 128-dim embeddings from the
VGGish model pretrained on AudioSet.
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
    """Extract VGGish embeddings for all audio files.

    Returns:
        np.ndarray of shape (N, 128)
    """
    model = torch.hub.load("harritaylor/torchvggish", "vggish")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    target_sr = cfg.EMBEDDING_MODELS["vggish"]["sample_rate"]  # 16000
    expected_len = int(target_sr * cfg.AUDIO_DURATION)

    embeddings = []
    with torch.no_grad():
        for fpath in tqdm(file_paths, desc="VGGish"):
            y, _ = librosa.load(fpath, sr=target_sr, duration=cfg.AUDIO_DURATION, mono=True)
            if len(y) < expected_len:
                y = np.pad(y, (0, expected_len - len(y)))
            else:
                y = y[:expected_len]

            # torchvggish accepts raw waveform or file path
            emb = model.forward(y, target_sr)
            # Mean-pool over time frames if multiple
            if emb.dim() > 1 and emb.shape[0] > 1:
                emb = emb.mean(dim=0)
            emb = emb.cpu().numpy().flatten()
            embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)
