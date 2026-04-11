"""
Run a single CNN baseline training experiment.

Trains MobileNetV2, ResNet-18, or EfficientNet-B0 on log-mel spectrograms
with early stopping and ReduceLROnPlateau.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class FSC22SpectrogramDataset(Dataset):
    """Dataset that generates log-mel spectrograms on the fly."""

    def __init__(self, file_paths, labels, input_size=(224, 224), augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.input_size = input_size
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio
        y, sr = librosa.load(audio_path, sr=cfg.SAMPLE_RATE, duration=cfg.AUDIO_DURATION, mono=True)

        # Pad if shorter than expected
        expected_len = int(cfg.SAMPLE_RATE * cfg.AUDIO_DURATION)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))
        else:
            y = y[:expected_len]

        # Compute log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=cfg.SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

        # Resize to input_size
        from PIL import Image
        img = Image.fromarray((log_mel * 255).astype(np.uint8))
        img = img.resize(self.input_size, Image.BILINEAR)
        spec_array = np.array(img, dtype=np.float32) / 255.0

        # Stack to 3 channels (for pretrained models)
        spec_tensor = torch.tensor(spec_array).unsqueeze(0).repeat(3, 1, 1)

        return spec_tensor, label


def build_model(architecture, num_classes):
    """Build CNN model with pretrained weights."""
    if architecture == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
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
            out = model(X)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)


def run_single_cnn_experiment(architecture, seed, data_fraction=1.0):
    """Train and evaluate a CNN baseline."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load split
    split_file = cfg.SPLITS_DIR / f"split_seed{seed}.json"
    with open(split_file) as f:
        split_data = json.load(f)

    meta_file = cfg.SPLITS_DIR / "metadata.json"
    with open(meta_file) as f:
        metadata = json.load(f)

    file_paths = np.array(metadata["files"])
    labels_str = np.array(metadata["labels"])

    le = LabelEncoder()
    le.fit(labels_str)
    labels_int = le.transform(labels_str)
    num_classes = len(le.classes_)

    # Get indices
    train_idx = split_data["split"]["train"]
    if data_fraction < 1.0:
        frac_key = f"{data_fraction:.2f}"
        if frac_key in split_data.get("low_data_train", {}):
            train_idx = split_data["low_data_train"][frac_key]

    val_idx = split_data["split"]["val"]
    test_idx = split_data["split"]["test"]

    # Create datasets
    train_ds = FSC22SpectrogramDataset(file_paths[train_idx], labels_int[train_idx])
    val_ds = FSC22SpectrogramDataset(file_paths[val_idx], labels_int[val_idx])
    test_ds = FSC22SpectrogramDataset(file_paths[test_idx], labels_int[test_idx])

    train_loader = DataLoader(train_ds, batch_size=cfg.CNN_TRAINING["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.CNN_TRAINING["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg.CNN_TRAINING["batch_size"], shuffle=False, num_workers=4)

    # Build model
    model = build_model(architecture, num_classes).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.CNN_TRAINING["label_smoothing"])
    optimizer = optim.Adam(model.parameters(), lr=cfg.CNN_TRAINING["initial_lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=cfg.CNN_TRAINING["lr_factor"],
        patience=cfg.CNN_TRAINING["lr_patience"],
    )

    # Training loop with early stopping
    best_val_f1 = -1
    best_state = None
    patience_counter = 0
    training_history = []  # epoch-by-epoch curves

    for epoch in range(cfg.CNN_TRAINING["max_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss), 6),
            "train_acc": round(float(train_acc), 6),
            "val_f1": round(float(val_f1), 6),
            "val_acc": round(float(val_acc), 6),
        })

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.CNN_TRAINING["early_stopping_patience"]:
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    model.to(device)
    y_pred, y_test = evaluate(model, test_loader, device)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    class_names = le.classes_
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)

    stem = f"{architecture}_seed{seed}_frac{data_fraction:.2f}"
    cfg.PER_CLASS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.PER_CLASS_DIR / f"{stem}.json", "w") as f:
        json.dump({
            "classes": class_names.tolist(),
            "f1_scores": per_class_f1.tolist(),
            "precision_scores": per_class_precision.tolist(),
            "recall_scores": per_class_recall.tolist(),
        }, f, indent=2)

    # Save classification report as text
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    with open(cfg.PER_CLASS_DIR / f"{stem}_report.txt", "w") as f:
        f.write(f"Model: {architecture}\n")
        f.write(f"Seed: {seed}  Data fraction: {data_fraction}\n")
        f.write(f"Epochs trained: {epoch + 1}\n\n")
        f.write(report)

    # Save training curves
    curves_dir = cfg.RESULTS_DIR / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    with open(curves_dir / f"{stem}.json", "w") as f:
        json.dump(training_history, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    cfg.CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cfg.CONFUSION_DIR / f"{stem}.npy", cm)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "arm": "custom_cnn",
        "model": architecture,
        "classifier": "end_to_end",
        "seed": seed,
        "data_fraction": data_fraction,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "val_macro_f1": best_val_f1,
        "best_params": json.dumps({"epochs_trained": epoch + 1}),
        "repr_size": "",     # N/A for CNNs (not a fixed-dim embedding)
        "n_params": n_params,
    }
