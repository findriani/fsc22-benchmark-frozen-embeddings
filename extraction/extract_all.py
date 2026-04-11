"""
Extract embeddings from all audio foundation models.

Each extractor loads the model, processes all FSC22 audio files,
and saves embeddings as a single NPZ file.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def load_audio_paths():
    """Load audio file paths and labels from metadata."""
    meta_file = cfg.SPLITS_DIR / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_file}. Run prepare_splits.py first.")
    with open(meta_file) as f:
        metadata = json.load(f)
    return metadata["files"], metadata["labels"]


def extract_model_embeddings(model_name):
    """Extract embeddings for a given model and save to NPZ."""
    output_file = cfg.EMBEDDINGS_DIR / f"{model_name}.npz"
    if output_file.exists():
        print(f"  {model_name} embeddings already exist at {output_file}")
        return

    cfg.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    file_paths, labels = load_audio_paths()
    print(f"  Extracting {model_name} for {len(file_paths)} files...")

    # Import the appropriate extractor
    if model_name == "panns_cnn14":
        from extraction.extractors.panns import extract as extractor
    elif model_name == "beats":
        from extraction.extractors.beats import extract as extractor
    elif model_name == "ast":
        from extraction.extractors.ast_model import extract as extractor
    elif model_name == "clap":
        from extraction.extractors.clap_model import extract as extractor
    elif model_name == "openl3":
        from extraction.extractors.openl3_model import extract as extractor
    elif model_name == "yamnet":
        from extraction.extractors.yamnet import extract as extractor
    elif model_name == "vggish":
        from extraction.extractors.vggish import extract as extractor
    elif model_name == "mfcc":
        from extraction.extractors.mfcc import extract as extractor
    elif model_name == "logmel_stats":
        from extraction.extractors.logmel_stats import extract as extractor
    else:
        raise ValueError(f"Unknown model: {model_name}")

    embeddings = extractor(file_paths)

    # Shape assertions: catch extractor bugs before saving
    expected_n = len(file_paths)
    assert embeddings.ndim == 2, (
        f"{model_name}: expected 2D embeddings (N, dim), got shape {embeddings.shape}"
    )
    assert embeddings.shape[0] == expected_n, (
        f"{model_name}: expected {expected_n} embeddings, got {embeddings.shape[0]}"
    )
    all_models = {**cfg.EMBEDDING_MODELS, **cfg.HANDCRAFTED_FEATURES}
    if model_name in all_models:
        expected_dim = all_models[model_name]["dim"]
        assert embeddings.shape[1] == expected_dim, (
            f"{model_name}: expected dim={expected_dim}, got dim={embeddings.shape[1]}. "
            f"Check extractor pooling strategy."
        )

    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        labels=np.array(labels),
        file_paths=np.array(file_paths),
        model_name=model_name,
    )
    print(f"  Saved {embeddings.shape} to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Extract specific model only")
    args = parser.parse_args()

    if args.model:
        extract_model_embeddings(args.model)
    else:
        all_models = list(cfg.EMBEDDING_MODELS.keys()) + list(cfg.HANDCRAFTED_FEATURES.keys())
        for name in all_models:
            try:
                extract_model_embeddings(name)
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
