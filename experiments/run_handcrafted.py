"""
Run a single handcrafted feature + classifier experiment.

Supports MFCC+SVM and log-mel stats+XGBoost.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def load_features(feature_name):
    """Load pre-extracted handcrafted features from NPZ file."""
    feat_file = cfg.EMBEDDINGS_DIR / f"{feature_name}.npz"
    data = np.load(feat_file, allow_pickle=True)
    return data["embeddings"], data["labels"]


def load_split(seed, data_fraction=1.0):
    """Load train/val/test indices."""
    split_file = cfg.SPLITS_DIR / f"split_seed{seed}.json"
    with open(split_file) as f:
        split_data = json.load(f)

    train_idx = split_data["split"]["train"]
    val_idx = split_data["split"]["val"]
    test_idx = split_data["split"]["test"]

    if data_fraction < 1.0:
        frac_key = f"{data_fraction:.2f}"
        if frac_key in split_data.get("low_data_train", {}):
            train_idx = split_data["low_data_train"][frac_key]

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def run_single_handcrafted_experiment(feature_name, classifier_type, seed, data_fraction=1.0):
    """Run one handcrafted feature experiment."""
    features, labels = load_features(feature_name)
    train_idx, val_idx, test_idx = load_split(seed, data_fraction)

    le = LabelEncoder()
    le.fit(labels)
    y_all = le.transform(labels)

    X_train, y_train = features[train_idx], y_all[train_idx]
    X_val, y_val = features[val_idx], y_all[val_idx]
    X_test, y_test = features[test_idx], y_all[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if classifier_type == "svm":
        # Grid search SVM
        best_score, best_params = -1, {}
        for C in [0.1, 1.0, 10.0, 100.0]:
            for gamma in ["scale", "auto", 0.001, 0.01, 0.1]:
                clf = SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=42)
                clf.fit(X_train, y_train)
                score = f1_score(y_val, clf.predict(X_val), average="macro", zero_division=0)
                if score > best_score:
                    best_score = score
                    best_params = {"C": C, "gamma": gamma}

        clf = SVC(**best_params, kernel="rbf", probability=True, random_state=42)

    elif classifier_type == "xgboost":
        from xgboost import XGBClassifier
        best_score, best_params = -1, {}
        for n_est in [100, 200]:
            for max_depth in [3, 5, 7]:
                for lr in [0.01, 0.1]:
                    xgb = XGBClassifier(
                        n_estimators=n_est, max_depth=max_depth,
                        learning_rate=lr, random_state=42,
                        use_label_encoder=False, eval_metric="mlogloss",
                    )
                    xgb.fit(X_train, y_train)
                    score = f1_score(y_val, xgb.predict(X_val), average="macro", zero_division=0)
                    if score > best_score:
                        best_score = score
                        best_params = {"n_estimators": n_est, "max_depth": max_depth, "learning_rate": lr}

        clf = XGBClassifier(
            **best_params, random_state=42,
            use_label_encoder=False, eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

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

    stem = f"{feature_name}_{classifier_type}_seed{seed}_frac{data_fraction:.2f}"
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
        f.write(f"Model: {feature_name}  Classifier: {classifier_type}\n")
        f.write(f"Seed: {seed}  Data fraction: {data_fraction}\n")
        f.write(f"Best params: {best_params}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    cfg.CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cfg.CONFUSION_DIR / f"{stem}.npy", cm)

    return {
        "arm": "handcrafted",
        "model": feature_name,
        "classifier": classifier_type,
        "seed": seed,
        "data_fraction": data_fraction,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "val_macro_f1": best_score,
        "best_params": json.dumps(best_params),
        "repr_size": features.shape[1],   # handcrafted feature vector dimension
        "n_params": "",                    # N/A for handcrafted features
    }
