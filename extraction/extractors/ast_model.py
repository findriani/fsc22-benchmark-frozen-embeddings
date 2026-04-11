"""
AST (Audio Spectrogram Transformer) embedding extractor.

Uses the HuggingFace transformers library to extract 768-dim embeddings
from the AST model pretrained on AudioSet.
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
    """Extract AST embeddings for all audio files.

    Returns:
        np.ndarray of shape (N, 768)
    """
    from transformers import ASTModel, ASTFeatureExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
    model = ASTModel.from_pretrained(model_name).to(device)
    model.eval()

    target_sr = cfg.EMBEDDING_MODELS["ast"]["sample_rate"]  # 16000
    expected_len = int(target_sr * cfg.AUDIO_DURATION)

    embeddings = []
    with torch.no_grad():
        for fpath in tqdm(file_paths, desc="AST"):
            y, _ = librosa.load(fpath, sr=target_sr, duration=cfg.AUDIO_DURATION, mono=True)
            if len(y) < expected_len:
                y = np.pad(y, (0, expected_len - len(y)))
            else:
                y = y[:expected_len]

            inputs = feature_extractor(
                y, sampling_rate=target_sr, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Use CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)
