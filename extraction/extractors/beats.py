"""
BEATs embedding extractor.

Uses the official BEATs model to extract 768-dim embeddings.
Requires downloading the BEATs checkpoint.
"""

import numpy as np
import torch
import librosa
from tqdm import tqdm

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg

# BEATs is not pip-installable — add its source directory to sys.path.
# Default: /workspace/unilm/beats  (RunPod convention)
# Override with environment variable: export BEATS_SRC=/your/path/to/unilm/beats
_beats_src = Path(os.environ.get("BEATS_SRC", "/workspace/unilm/beats"))
if _beats_src.exists() and str(_beats_src) not in sys.path:
    sys.path.insert(0, str(_beats_src))

BEATS_CKPT_URL = "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

_MANUAL_DOWNLOAD_MSG = """
BEATs checkpoint could not be downloaded automatically.
The Microsoft Azure storage URL has restricted public access (HTTP 409).

To fix:
  1. Download the checkpoint manually from the BEATs GitHub page:
       https://github.com/microsoft/unilm/tree/master/beats
     (look for the 'BEATs_iter3+_AS2M' checkpoint link)
  2. Place it at: {ckpt_path}
     OR set the env var: export BEATS_CKPT_PATH=/path/to/your/BEATs_iter3_plus_AS2M.pt

Alternatively, skip BEATs for now — all other models are unaffected.
"""


def download_checkpoint():
    """Download BEATs checkpoint if not present, with env var override support."""
    # Allow user to point to a manually downloaded checkpoint
    env_path = os.environ.get("BEATS_CKPT_PATH")
    if env_path:
        ckpt_path = Path(env_path)
        if ckpt_path.exists():
            return ckpt_path
        raise FileNotFoundError(f"BEATS_CKPT_PATH set but file not found: {ckpt_path}")

    ckpt_dir = cfg.PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "BEATs_iter3_plus_AS2M.pt"
    if ckpt_path.exists():
        return ckpt_path

    print(f"Downloading BEATs checkpoint to {ckpt_path}...")
    import urllib.request
    try:
        urllib.request.urlretrieve(BEATS_CKPT_URL, ckpt_path)
    except Exception as e:
        if ckpt_path.exists():
            ckpt_path.unlink()  # remove partial download
        raise RuntimeError(
            f"Failed to download BEATs checkpoint ({e}).\n"
            + _MANUAL_DOWNLOAD_MSG.format(ckpt_path=ckpt_path)
        ) from e
    return ckpt_path


def extract(file_paths):
    """Extract BEATs embeddings for all audio files.

    Returns:
        np.ndarray of shape (N, 768)
    """
    from BEATs import BEATs, BEATsConfig

    ckpt_path = download_checkpoint()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    beat_cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(beat_cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    target_sr = cfg.EMBEDDING_MODELS["beats"]["sample_rate"]  # 16000
    expected_len = int(target_sr * cfg.AUDIO_DURATION)

    # Fine-tuned BEATs: extract_features() returns (batch, 527) class logits, not
    # (batch, time, 768) encoder features. To get the 768-dim encoder representation,
    # register a forward hook on model.predictor to capture its input — which is
    # exactly the time-pooled encoder output the predictor head operates on.
    hook_storage = {}
    if beat_cfg.finetuned_model:
        def _capture_encoder_output(module, inp, out):
            hook_storage["features"] = inp[0].detach()  # (batch, 768)
        hook = model.predictor.register_forward_hook(_capture_encoder_output)
    else:
        hook = None

    embeddings = []
    with torch.no_grad():
        for fpath in tqdm(file_paths, desc="BEATs"):
            y, _ = librosa.load(fpath, sr=target_sr, duration=cfg.AUDIO_DURATION, mono=True)
            if len(y) < expected_len:
                y = np.pad(y, (0, expected_len - len(y)))
            else:
                y = y[:expected_len]

            waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            padding_mask = torch.zeros(1, waveform.shape[1], dtype=torch.bool).to(device)
            _ = model.extract_features(waveform, padding_mask=padding_mask)

            if beat_cfg.finetuned_model:
                # hook captured (batch, time, 768) — mean-pool over time then flatten
                emb = hook_storage["features"].mean(dim=1).cpu().numpy().flatten()
            else:
                # pre-trained: extract_features returns (batch, time, 768)
                rep = _[0]
                emb = rep.mean(dim=1).cpu().numpy().flatten()

            embeddings.append(emb)

    if hook is not None:
        hook.remove()

    return np.array(embeddings, dtype=np.float32)
