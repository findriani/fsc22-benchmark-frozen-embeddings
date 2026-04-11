"""
Auxiliary experiment (E6b): CLAP+SVM with augmentation.
Runs 5 seeds at 100% data. Results saved to results/augmented_embedding_results.csv.

Purpose: Small ablation to check whether frozen embeddings also benefit from
augmentation, and whether the embedding gap over CNNs holds even when both arms
receive the same augmentation set.

How it works:
  - For each seed, one augmented copy per training clip is generated before
    CLAP extraction and appended to the training set (doubles training size)
  - SVM is trained on original + augmented training embeddings
  - Val and test sets are NOT augmented (same protocol as main experiment)

Augmentation applied before CLAP extraction:
  - Gaussian noise  : amplitude = Uniform(0.001, 0.005) x signal std
  (Time-stretch and pitch-shift excluded: torchaudio resample/phase-vocoder at
  48 kHz is prohibitively slow per-sample on CPU. Gaussian noise is fast
  (tensor addition) and sufficient to test embedding stability under input
  perturbation.)

N_AUGMENTED = 1 (one augmented copy per training clip).
Expected runtime: ~30-60 min on RunPod (CLAP extraction is the bottleneck).

Usage (RunPod):
    cd /workspace/fsc22-benchmark
    python experiments/run_augmented_embedding.py
"""

import json
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

SEEDS = [42, 50, 100, 200, 300]
N_AUGMENTED = 1   # augmented copies per training clip

NOISE_AMP_RANGE = (0.001, 0.005)

TARGET_SR    = cfg.EMBEDDING_MODELS["clap"]["sample_rate"]   # 48000
EXPECTED_LEN = int(TARGET_SR * cfg.AUDIO_DURATION)


def load_clap_model():
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    model.load_ckpt()
    model.eval()
    return model


def augment_waveform(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply Gaussian noise (fast tensor addition, no resampling)."""
    wav = torch.tensor(y, dtype=torch.float32)
    amp = random.uniform(*NOISE_AMP_RANGE) * wav.std().item()
    wav = wav + amp * torch.randn_like(wav)
    return wav.numpy()


def extract_clap_embeddings(model, file_paths, augment=False, n_aug=1):
    """
    Extract CLAP embeddings. If augment=True, each file produces n_aug augmented
    waveforms in addition to the original (original is not included when augment=True,
    so this function is called once for originals and once for augmented copies).
    """
    import laion_clap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    all_embeddings = []
    batch_audio, batch_idx = [], []

    def flush_batch():
        if not batch_audio:
            return
        audio_tensor = torch.tensor(np.stack(batch_audio), dtype=torch.float32).to(device)
        with torch.no_grad():
            embs = model.get_audio_embedding_from_data(audio_tensor, use_tensor=True)
        all_embeddings.extend(embs.cpu().numpy())
        batch_audio.clear()
        batch_idx.clear()

    desc = "CLAP (augmented)" if augment else "CLAP (original)"
    for fpath in tqdm(file_paths, desc=desc):
        y, _ = librosa.load(fpath, sr=TARGET_SR, duration=cfg.AUDIO_DURATION, mono=True)
        if len(y) < EXPECTED_LEN:
            y = np.pad(y, (0, EXPECTED_LEN - len(y)))
        else:
            y = y[:EXPECTED_LEN]

        copies = [augment_waveform(y, TARGET_SR) for _ in range(n_aug)] if augment else [y]
        for wav in copies:
            batch_audio.append(wav)
            if len(batch_audio) == 32:
                flush_batch()

    flush_batch()
    return np.array(all_embeddings)


def run_seed(seed, clap_model):
    print(f"\n{'='*60}")
    print(f"CLAP+SVM + augmentation | seed={seed}")
    print(f"{'='*60}")
    t0 = time.time()

    random.seed(seed)
    np.random.seed(seed)

    with open(cfg.SPLITS_DIR / f"split_seed{seed}.json") as f:
        split_data = json.load(f)
    with open(cfg.SPLITS_DIR / "metadata.json") as f:
        metadata = json.load(f)

    file_paths = np.array(metadata["files"])
    labels_str = np.array(metadata["labels"])
    le = LabelEncoder()
    le.fit(labels_str)
    labels_int = le.transform(labels_str)

    train_idx = np.array(split_data["split"]["train"])
    val_idx   = np.array(split_data["split"]["val"])
    test_idx  = np.array(split_data["split"]["test"])

    train_files = file_paths[train_idx]
    train_labels = labels_int[train_idx]

    # Extract original training embeddings
    print("Extracting original training embeddings...")
    X_train_orig = extract_clap_embeddings(clap_model, train_files, augment=False)

    # Extract augmented training embeddings (N_AUGMENTED copies per clip)
    print(f"Extracting {N_AUGMENTED} augmented copy per training clip...")
    X_train_aug = extract_clap_embeddings(clap_model, train_files, augment=True, n_aug=N_AUGMENTED)
    y_train_aug = np.tile(train_labels, N_AUGMENTED)

    # Combine original + augmented
    X_train = np.vstack([X_train_orig, X_train_aug])
    y_train  = np.concatenate([train_labels, y_train_aug])

    # Extract val and test (no augmentation)
    print("Extracting val/test embeddings...")
    X_val  = extract_clap_embeddings(clap_model, file_paths[val_idx],  augment=False)
    X_test = extract_clap_embeddings(clap_model, file_paths[test_idx], augment=False)
    y_val  = labels_int[val_idx]
    y_test = labels_int[test_idx]

    # Standardise
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Grid search: C in {0.1, 1, 10, 100} (same as main experiment)
    best_val_f1, best_C = -1, None
    for C in [0.1, 1, 10, 100]:
        svm = SVC(kernel="rbf", C=C, gamma="scale")
        svm.fit(X_train, y_train)
        val_f1 = f1_score(y_val, svm.predict(X_val), average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1, best_C = val_f1, C

    # Retrain best on train+val
    svm = SVC(kernel="rbf", C=best_C, gamma="scale")
    X_trainval = np.vstack([X_train, scaler.transform(X_val)])
    y_trainval  = np.concatenate([y_train, y_val])
    svm.fit(X_trainval, y_trainval)

    y_pred   = svm.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    elapsed  = time.time() - t0

    print(f"Seed {seed}: macro_f1={macro_f1:.4f}  accuracy={accuracy:.4f}  best_C={best_C}  ({elapsed/60:.1f} min)")
    return {
        "model": "clap_svm_augmented",
        "seed": seed,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "val_macro_f1": best_val_f1,
        "best_C": best_C,
        "n_augmented": N_AUGMENTED,
        "time_s": elapsed,
    }


def main():
    print("E6b: CLAP+SVM + Augmentation Ablation")
    print(f"  N_AUGMENTED    : {N_AUGMENTED} copy per training clip")
    print(f"  Gaussian noise : amp={NOISE_AMP_RANGE} x signal_std")
    print(f"  Seeds          : {SEEDS}")
    print()

    print("Loading CLAP model (once, shared across seeds)...")
    clap_model = load_clap_model()

    results = []
    for seed in SEEDS:
        results.append(run_seed(seed, clap_model))

    df = pd.DataFrame(results)
    out = cfg.RESULTS_DIR / "augmented_embedding_results.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")
    print(f"\nSummary:")
    print(f"  CLAP+SVM + aug  : {df['macro_f1'].mean():.3f} ± {df['macro_f1'].std():.3f}")
    print(f"  CLAP+SVM no aug : 0.896 ± 0.008  (from main experiment)")
    print(f"  ResNet-18 no aug: 0.814 ± 0.022  (from main experiment)")


if __name__ == "__main__":
    main()
