"""
Auxiliary experiment (E6a): ResNet-18 with augmentation (precomputed bank).
Runs 5 seeds at 100% data. Results saved to results/augmented_cnn_results.csv.

Purpose: Quantify how much augmentation narrows the gap between CNNs and frozen
embeddings, to preempt the reviewer objection "augmentation closes the gap."
Expected runtime: ~2-3 GPU hours on RunPod.

Augmentation strategy: precomputed bank (data/augmented/v1/).
  - Each training clip is paired with one pitch-shifted copy and one time-stretched
    copy from the bank. Training set is therefore 3x the original size.
  - Val and test sets: unchanged (original clips only).
  - Bank parameters: PITCH_STEPS = [-2,-1,1,2] semitones (cycled by index)
                     TIME_RATES  = [0.85,0.90,0.95,1.05,1.10,1.15] (cycled by index)

Prerequisite: run data/precompute_augmentations.py once to generate the bank.

Why precomputed bank (not online augmentation)?
  torchaudio.functional.pitch_shift and resample run in CPU-bound DataLoader workers
  and take ~21s/sample at 44.1 kHz. Precomputing once avoids this per-epoch overhead.

Usage (RunPod):
    cd /workspace/fsc22
    python data/precompute_augmentations.py   # one-time, ~20-40 min
    python experiments/run_augmented_cnn.py
"""

import json
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

SEEDS         = [42, 50, 100, 200, 300]
ARCHITECTURE  = "resnet18"
DATA_FRACTION = 1.0
AUG_BANK_DIR  = cfg.DATA_DIR / "augmented" / "v1"


class SpectrogramDataset(Dataset):
    """Load audio file and return log-mel spectrogram as 3-channel 224x224 tensor."""

    def __init__(self, file_paths, labels, input_size=(224, 224)):
        self.file_paths = list(file_paths)
        self.labels     = list(labels)
        self.input_size = input_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        y, _ = librosa.load(
            self.file_paths[idx],
            sr=cfg.SAMPLE_RATE, duration=cfg.AUDIO_DURATION, mono=True
        )
        expected_len = int(cfg.SAMPLE_RATE * cfg.AUDIO_DURATION)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=cfg.SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

        from PIL import Image
        img = Image.fromarray((log_mel * 255).astype(np.uint8))
        img = img.resize(self.input_size, Image.BILINEAR)
        spec_array = np.array(img, dtype=np.float32) / 255.0

        spec_tensor = torch.tensor(spec_array).unsqueeze(0).repeat(3, 1, 1)
        return spec_tensor, self.labels[idx]


def build_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            all_preds.extend(model(X).argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)


def run_seed(seed, bank):
    print(f"\n{'='*60}")
    print(f"ResNet-18 + augmentation bank | seed={seed}")
    print(f"{'='*60}")
    t0 = time.time()

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(cfg.SPLITS_DIR / f"split_seed{seed}.json") as f:
        split_data = json.load(f)
    with open(cfg.SPLITS_DIR / "metadata.json") as f:
        metadata = json.load(f)

    file_paths  = np.array(metadata["files"])
    labels_str  = np.array(metadata["labels"])
    le          = LabelEncoder()
    le.fit(labels_str)
    labels_int  = le.transform(labels_str)
    num_classes = len(le.classes_)

    train_idx = np.array(split_data["split"]["train"])
    val_idx   = np.array(split_data["split"]["val"])
    test_idx  = np.array(split_data["split"]["test"])

    # Build training set: original + pitch bank + time bank (3x size)
    pitch_files = np.array(bank["pitch"])
    time_files  = np.array(bank["time"])

    all_train_files = np.concatenate([
        file_paths[train_idx],
        pitch_files[train_idx],
        time_files[train_idx],
    ])
    all_train_labels = np.concatenate([
        labels_int[train_idx],
        labels_int[train_idx],
        labels_int[train_idx],
    ])
    print(f"Training set: {len(train_idx)} original + {len(train_idx)} pitch + {len(train_idx)} time = {len(all_train_files)} samples")

    bs = cfg.CNN_TRAINING["batch_size"]
    train_ds = SpectrogramDataset(all_train_files,        all_train_labels)
    val_ds   = SpectrogramDataset(file_paths[val_idx],   labels_int[val_idx])
    test_ds  = SpectrogramDataset(file_paths[test_idx],  labels_int[test_idx])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    model     = build_resnet18(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.CNN_TRAINING["label_smoothing"])
    optimizer = optim.Adam(model.parameters(), lr=cfg.CNN_TRAINING["initial_lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=cfg.CNN_TRAINING["lr_factor"],
        patience=cfg.CNN_TRAINING["lr_patience"],
    )

    best_val_f1, best_state, patience_counter, epoch = -1, None, 0, 0
    for epoch in range(cfg.CNN_TRAINING["max_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | loss={train_loss:.4f} acc={train_acc:.3f} | val_f1={val_f1:.3f} (best={best_val_f1:.3f})")

        if patience_counter >= cfg.CNN_TRAINING["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    y_pred, y_test = evaluate(model, test_loader, device)

    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy    = accuracy_score(y_test, y_pred)
    elapsed     = time.time() - t0

    # Explicit cleanup to free GPU and RAM before next seed
    del model, train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
    del best_state, optimizer, scheduler, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nSeed {seed}: macro_f1={macro_f1:.4f}  accuracy={accuracy:.4f}  ({elapsed/60:.1f} min)")
    return {
        "model": "resnet18_augmented",
        "seed": seed,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "val_macro_f1": best_val_f1,
        "epochs_trained": epoch + 1,
        "time_s": elapsed,
    }


def main():
    manifest_path = AUG_BANK_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Augmentation bank not found at {AUG_BANK_DIR}")
        print("Run first:  python data/precompute_augmentations.py")
        sys.exit(1)

    with open(manifest_path) as f:
        bank = json.load(f)

    print("E6a: ResNet-18 + Augmentation Bank")
    print(f"  Bank: {AUG_BANK_DIR}")
    print(f"  Augmentations: pitch-shift + time-stretch (precomputed) per training clip")
    print(f"  Training set: 3x original (original + pitch + time per clip)")
    print(f"  Seeds: {SEEDS}")

    results = []
    for seed in SEEDS:
        results.append(run_seed(seed, bank))

    df = pd.DataFrame(results)
    out = cfg.RESULTS_DIR / "augmented_cnn_results.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")
    print(f"\nSummary:")
    print(f"  ResNet-18 + aug  : {df['macro_f1'].mean():.3f} ± {df['macro_f1'].std():.3f}")
    print(f"  ResNet-18 no aug : 0.814 ± 0.022  (main experiment)")
    print(f"  CLAP+SVM  no aug : 0.896 ± 0.008  (main experiment)")


if __name__ == "__main__":
    main()
