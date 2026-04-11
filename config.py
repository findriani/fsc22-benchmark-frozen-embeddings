"""
Configuration for FSC22 Frozen Embedding Benchmark.

Central config for all experiments: paths, models, classifiers, hyperparams, seeds.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
EMBEDDINGS_DIR = PROJECT_ROOT / "extraction" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
PROGRESS_FILE = RESULTS_DIR / "progress.json"
RESULTS_CSV = RESULTS_DIR / "all_results.csv"
TIMING_DIR = RESULTS_DIR / "timing"
PER_CLASS_DIR = RESULTS_DIR / "per_class"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

# FSC22 dataset path — set via environment variable or default
FSC22_AUDIO_DIR = Path(os.environ.get("FSC22_AUDIO_DIR", DATA_DIR / "FSC22"))
FSC22_METADATA = Path(os.environ.get("FSC22_METADATA", DATA_DIR / "metadata.csv"))

# ── Dataset ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100  # FSC22 native sample rate
AUDIO_DURATION = 5.0  # seconds — FSC22 clips are 5s
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test

# ── Seeds ──────────────────────────────────────────────────────────────────────
SEEDS = [42, 50, 100, 200, 300]

# ── Embedding Models ───────────────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "panns_cnn14": {
        "dim": 2048,
        "sample_rate": 32000,
        "description": "PANNs CNN14, supervised on AudioSet (527 classes)",
    },
    "beats": {
        "dim": 768,
        "sample_rate": 16000,
        "description": "BEATs, self-supervised + fine-tuned on AudioSet",
    },
    "ast": {
        "dim": 768,
        "sample_rate": 16000,
        "description": "AST, supervised on AudioSet + ImageNet",
    },
    "clap": {
        "dim": 512,
        "sample_rate": 48000,
        "description": "CLAP, contrastive audio-text on AudioSet",
    },
    "openl3": {
        "dim": 6144,
        "sample_rate": 48000,
        "description": "OpenL3, self-supervised audio-visual on AudioSet",
    },
    "yamnet": {
        "dim": 1024,
        "sample_rate": 16000,
        "description": "YAMNet, supervised MobileNet-v1 on AudioSet",
    },
    "vggish": {
        "dim": 128,
        "sample_rate": 16000,
        "description": "VGGish, supervised VGG-like on AudioSet",
    },
}

# ── Handcrafted Features ───────────────────────────────────────────────────────
HANDCRAFTED_FEATURES = {
    "mfcc": {
        "dim": 320,  # 40 MFCCs x 8 statistics (mean, std, min, max, median, skew, kurtosis, range)
        "description": "MFCC summary statistics",
    },
    "logmel_stats": {
        "dim": 1024,  # 128 mel bins x 8 statistics
        "description": "Log-mel spectrogram summary statistics",
    },
}

# ── Classifiers for Frozen Embeddings ──────────────────────────────────────────
# Three classifiers covering a range of downstream head complexity:
#   lr  = Logistic Regression  → linear probe (standard "frozen embedding" baseline)
#   svm = SVM with RBF kernel  → nonlinear shallow head
#   mlp = MLP                  → small neural head (still trained on embeddings only)
# Paper framing: lr tests whether embeddings are linearly separable;
# svm/mlp test the best achievable with shallow heads.
CLASSIFIERS = {
    "lr": {
        "name": "Logistic Regression (linear probe)",
        "param_grid": {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        },
    },
    "svm": {
        "name": "SVM (RBF kernel)",
        "param_grid": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
    },
    "mlp": {
        "name": "MLP (shallow neural head)",
        "param_grid": {
            "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
        },
    },
}

# ── Handcrafted Classifiers ────────────────────────────────────────────────────
HANDCRAFTED_CLASSIFIERS = {
    "mfcc": "svm",       # MFCC + SVM
    "logmel_stats": "xgboost",  # Log-mel stats + XGBoost
}

# ── CNN Baselines ──────────────────────────────────────────────────────────────
CNN_ARCHITECTURES = {
    "mobilenetv2": {
        "name": "MobileNetV2",
        "pretrained": True,
    },
    "resnet18": {
        "name": "ResNet-18",
        "pretrained": True,
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "pretrained": True,
    },
}

CNN_TRAINING = {
    "optimizer": "adam",
    "initial_lr": 1e-3,
    "batch_size": 32,
    "max_epochs": 60,
    "early_stopping_patience": 10,
    "lr_scheduler": "reduce_on_plateau",
    "lr_factor": 0.1,
    "lr_patience": 5,
    "label_smoothing": 0.1,
    "input_size": (224, 224),
    "feature_type": "logmel",  # log-mel spectrogram images
}

# ── Low-Data Regime (E3) ───────────────────────────────────────────────────────
DATA_FRACTIONS = [0.10, 0.25, 0.50, 1.00]

# ── Experiment Phases ──────────────────────────────────────────────────────────
PHASES = [
    "extraction",    # Phase 1: Extract all embeddings
    "e1_embeddings", # Phase 2a: E1 frozen embedding experiments
    "e1_handcrafted",# Phase 2b: E1 handcrafted experiments
    "e1_cnn",        # Phase 2c: E1 CNN baseline training
    "e3_lowdata",    # Phase 3: Low-data regime
    "e5_efficiency", # Phase 4: Efficiency measurements
]
