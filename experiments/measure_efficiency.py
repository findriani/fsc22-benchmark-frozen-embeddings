"""
E5: Efficiency measurements for all models.

Measures per-sample extraction time, classifier training time,
inference time, model size, and parameter counts.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def measure_extraction_time(model_name, n_samples=20):
    """Measure per-sample embedding extraction time."""
    # Load a few real audio files from metadata
    meta_file = cfg.SPLITS_DIR / "metadata.json"
    with open(meta_file) as f:
        metadata = json.load(f)
    sample_files = metadata["files"][:n_samples]

    # Import extractor
    extractors = {
        "panns_cnn14": "extraction.extractors.panns",
        "beats": "extraction.extractors.beats",
        "ast": "extraction.extractors.ast_model",
        "clap": "extraction.extractors.clap_model",
        "openl3": "extraction.extractors.openl3_model",
        "yamnet": "extraction.extractors.yamnet",
        "vggish": "extraction.extractors.vggish",
        "mfcc": "extraction.extractors.mfcc",
        "logmel_stats": "extraction.extractors.logmel_stats",
    }

    if model_name not in extractors:
        return None

    import importlib
    mod = importlib.import_module(extractors[model_name])

    # Warm-up run (1 sample)
    try:
        mod.extract(sample_files[:1])
    except Exception:
        pass

    # Timed run
    start = time.time()
    mod.extract(sample_files)
    elapsed = time.time() - start
    per_sample = elapsed / len(sample_files)

    return {
        "model": model_name,
        "n_samples": len(sample_files),
        "total_time_s": round(elapsed, 4),
        "per_sample_s": round(per_sample, 4),
    }


def measure_classifier_time(embedding_name, classifier_name, seed=42):
    """Measure classifier training + inference time on pre-extracted embeddings."""
    from experiments.run_embedding_clf import (
        load_embeddings, load_split, create_classifier, grid_search,
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import f1_score

    embeddings, labels = load_embeddings(embedding_name)
    train_idx, val_idx, test_idx = load_split(seed)

    le = LabelEncoder()
    le.fit(labels)
    y_all = le.transform(labels)

    X_train, y_train = embeddings[train_idx], y_all[train_idx]
    X_val, y_val = embeddings[val_idx], y_all[val_idx]
    X_test, y_test = embeddings[test_idx], y_all[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    clf_config = cfg.CLASSIFIERS[classifier_name]

    # Measure grid search + training time
    start = time.time()
    best_params, _ = grid_search(X_train, y_train, X_val, y_val, classifier_name, clf_config["param_grid"])
    clf = create_classifier(classifier_name, best_params)
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    # Measure inference time
    start = time.time()
    for _ in range(10):
        clf.predict(X_test)
    infer_time = (time.time() - start) / 10
    per_sample_infer = infer_time / len(X_test)

    return {
        "embedding": embedding_name,
        "classifier": classifier_name,
        "train_time_s": round(train_time, 4),
        "inference_time_s": round(infer_time, 4),
        "per_sample_inference_s": round(per_sample_infer, 6),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def measure_cnn_params():
    """Measure CNN model sizes and parameter counts."""
    from experiments.run_cnn_baseline import build_model

    results = []
    for arch_name in cfg.CNN_ARCHITECTURES:
        model = build_model(arch_name, num_classes=27)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate model size in MB
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        results.append({
            "model": arch_name,
            "total_params": n_params,
            "trainable_params": n_trainable,
            "model_size_mb": round(param_size_mb, 2),
        })
        del model

    return results


def measure_cnn_inference_time(n_samples=20):
    """
    Measure end-to-end CNN inference time per sample.

    Times the full pipeline: audio load → log-mel spectrogram → CNN forward pass.
    This is directly comparable to the embedding extraction time measured for
    the frozen embedding arm (audio → embedding vector).

    Uses pretrained weights (no saved training checkpoint needed).
    """
    from experiments.run_cnn_baseline import build_model
    from PIL import Image

    meta_file = cfg.SPLITS_DIR / "metadata.json"
    with open(meta_file) as f:
        metadata = json.load(f)
    sample_files = [Path(p) for p in metadata["files"][:n_samples]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for arch_name in cfg.CNN_ARCHITECTURES:
        model = build_model(arch_name, num_classes=27)
        model = model.to(device)
        model.eval()

        # Warm-up: one sample
        try:
            y, _ = librosa.load(str(sample_files[0]), sr=cfg.SAMPLE_RATE,
                                duration=cfg.AUDIO_DURATION, mono=True)
            expected_len = int(cfg.SAMPLE_RATE * cfg.AUDIO_DURATION)
            if len(y) < expected_len:
                y = np.pad(y, (0, expected_len - len(y)))
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=cfg.SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
            img = Image.fromarray((log_mel * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            tensor = torch.tensor(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(tensor)
        except Exception:
            pass

        # Timed run: full pipeline per sample
        start = time.time()
        for fpath in sample_files:
            y, _ = librosa.load(str(fpath), sr=cfg.SAMPLE_RATE,
                                duration=cfg.AUDIO_DURATION, mono=True)
            expected_len = int(cfg.SAMPLE_RATE * cfg.AUDIO_DURATION)
            if len(y) < expected_len:
                y = np.pad(y, (0, expected_len - len(y)))
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=cfg.SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
            img = Image.fromarray((log_mel * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            tensor = torch.tensor(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(tensor)
        elapsed = time.time() - start
        per_sample = elapsed / len(sample_files)

        results.append({
            "model": arch_name,
            "n_samples": len(sample_files),
            "total_time_s": round(elapsed, 4),
            "per_sample_s": round(per_sample, 4),
            "device": str(device),
        })
        print(f"  {arch_name}: {per_sample:.4f} s/sample ({device})")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def measure_all_efficiency():
    """Run all efficiency measurements and save results."""
    cfg.TIMING_DIR.mkdir(parents=True, exist_ok=True)

    results = {"extraction": [], "classifier": [], "cnn_params": [], "cnn_inference": []}

    # 1. Extraction times for all embedding models
    all_models = list(cfg.EMBEDDING_MODELS.keys()) + list(cfg.HANDCRAFTED_FEATURES.keys())
    for model_name in all_models:
        print(f"Measuring extraction time for {model_name}...")
        try:
            timing = measure_extraction_time(model_name)
            if timing:
                results["extraction"].append(timing)
                print(f"  {timing['per_sample_s']:.4f} s/sample")
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")

    # 2. Classifier training + inference times
    for emb_name in cfg.EMBEDDING_MODELS:
        emb_file = cfg.EMBEDDINGS_DIR / f"{emb_name}.npz"
        if not emb_file.exists():
            continue
        for clf_name in cfg.CLASSIFIERS:
            print(f"Measuring classifier time for {emb_name}+{clf_name}...")
            try:
                timing = measure_classifier_time(emb_name, clf_name)
                results["classifier"].append(timing)
            except Exception as e:
                print(f"  [FAIL] {emb_name}+{clf_name}: {e}")

    # 3. CNN parameter counts
    print("Measuring CNN model sizes...")
    try:
        results["cnn_params"] = measure_cnn_params()
    except Exception as e:
        print(f"  [FAIL] CNN params: {e}")

    # 4. CNN end-to-end inference time (audio → spectrogram → forward pass)
    print("Measuring CNN end-to-end inference time...")
    try:
        results["cnn_inference"] = measure_cnn_inference_time()
    except Exception as e:
        print(f"  [FAIL] CNN inference timing: {e}")

    # Save all results
    with open(cfg.TIMING_DIR / "efficiency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved efficiency results to {cfg.TIMING_DIR / 'efficiency_results.json'}")


if __name__ == "__main__":
    measure_all_efficiency()
