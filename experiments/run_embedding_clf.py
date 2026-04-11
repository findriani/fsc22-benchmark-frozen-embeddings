"""
Run a single frozen embedding + classifier experiment.

Loads pre-extracted embeddings, applies a classifier with grid search,
evaluates on the test set, and returns comprehensive metrics.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def load_embeddings(embedding_name):
    """Load pre-extracted embeddings from NPZ file."""
    emb_file = cfg.EMBEDDINGS_DIR / f"{embedding_name}.npz"
    data = np.load(emb_file, allow_pickle=True)
    return data["embeddings"], data["labels"]


def load_split(seed, data_fraction=1.0):
    """Load train/val/test indices for a given seed and data fraction."""
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


def create_classifier(clf_name, params):
    """Create a classifier with given hyperparameters."""
    if clf_name == "lr":
        return LogisticRegression(
            C=params["C"], max_iter=1000, random_state=42
        )
    elif clf_name == "svm":
        return SVC(
            C=params["C"], gamma=params["gamma"], kernel="rbf",
            probability=True, random_state=42
        )
    elif clf_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            max_iter=1000, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")


def grid_search(X_train, y_train, X_val, y_val, clf_name, param_grid):
    """Simple grid search on validation set. Returns best params and score."""
    best_score = -1
    best_params = None

    # Generate all param combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    from itertools import product
    for combo in product(*values):
        params = dict(zip(keys, combo))
        clf = create_classifier(clf_name, params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        score = f1_score(y_val, y_pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


def run_single_embedding_experiment(embedding_name, classifier_name, seed, data_fraction=1.0):
    """Run one embedding + classifier experiment and return metrics dict."""
    # Load data
    embeddings, labels = load_embeddings(embedding_name)
    train_idx, val_idx, test_idx = load_split(seed, data_fraction)

    # Encode labels
    le = LabelEncoder()
    le.fit(labels)
    y_all = le.transform(labels)

    X_train, y_train = embeddings[train_idx], y_all[train_idx]
    X_val, y_val = embeddings[val_idx], y_all[val_idx]
    X_test, y_test = embeddings[test_idx], y_all[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Grid search on validation set
    clf_config = cfg.CLASSIFIERS[classifier_name]
    best_params, val_score = grid_search(
        X_train, y_train, X_val, y_val,
        classifier_name, clf_config["param_grid"]
    )

    # Train final model with best params
    clf = create_classifier(classifier_name, best_params)
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)

    # Compute metrics
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

    # Save per-class results
    stem = f"{embedding_name}_{classifier_name}_seed{seed}_frac{data_fraction:.2f}"
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
        f.write(f"Model: {embedding_name}  Classifier: {classifier_name}\n")
        f.write(f"Seed: {seed}  Data fraction: {data_fraction}\n")
        f.write(f"Best params: {best_params}\n\n")
        f.write(report)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cfg.CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cfg.CONFUSION_DIR / f"{stem}.npy", cm)

    return {
        "arm": "frozen_embedding",
        "model": embedding_name,
        "classifier": classifier_name,
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
        "val_macro_f1": val_score,
        "best_params": json.dumps(best_params),
        "repr_size": embeddings.shape[1],   # embedding vector dimension
        "n_params": "",                        # N/A for frozen embeddings
    }
