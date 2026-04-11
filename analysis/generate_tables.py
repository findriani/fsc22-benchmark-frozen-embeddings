"""
Generate paper tables from experiment results.

Tables:
  Table 3: E1 — All embedding-classifier macro-F1 results (mean ± std over 5 seeds)
  Table 4: Best per arm comparison (embeddings vs handcrafted vs CNNs)
  Table 5: Per-class F1 for top models
  Table 6: Efficiency comparison
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def load_results():
    """Load the unified results CSV."""
    if not cfg.RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results not found at {cfg.RESULTS_CSV}. Run experiments first.")
    return pd.read_csv(cfg.RESULTS_CSV)


def table3_embedding_results(df):
    """Table 3: Macro-F1 for all embedding-classifier combinations at 100% data."""
    e1 = df[(df["arm"] == "frozen_embedding") & (df["data_fraction"] == 1.0)]
    pivot = e1.groupby(["model", "classifier"])["macro_f1"].agg(["mean", "std"]).reset_index()
    pivot["result"] = pivot.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    table = pivot.pivot(index="model", columns="classifier", values="result")
    print("\n=== TABLE 3: E1 Embedding + Classifier Results (Macro-F1) ===")
    print(table.to_string())
    table.to_csv(cfg.RESULTS_DIR / "table3_embedding_results.csv")
    return table


def table4_arm_comparison(df):
    """Table 4: All models per arm — best classifier result per model, all arms."""
    e1 = df[df["data_fraction"] == 1.0]
    rows = []
    for arm in ["frozen_embedding", "handcrafted", "custom_cnn"]:
        arm_df = e1[e1["arm"] == arm]
        if arm_df.empty:
            continue
        grouped = arm_df.groupby(["model", "classifier"])["macro_f1"].agg(["mean", "std"])
        # For each model, keep only the best classifier
        for model, model_group in grouped.groupby(level="model"):
            best_idx = model_group["mean"].idxmax()
            best = model_group.loc[best_idx]
            rows.append({
                "arm": arm,
                "model": best_idx[0],
                "best_classifier": best_idx[1],
                "macro_f1_mean": round(best["mean"], 4),
                "macro_f1_std": round(best["std"], 4),
            })
    table = pd.DataFrame(rows).sort_values(["arm", "macro_f1_mean"], ascending=[True, False])
    print("\n=== TABLE 4: All Models Per Arm (Best Classifier) ===")
    print(table.to_string(index=False))
    table.to_csv(cfg.RESULTS_DIR / "table4_arm_comparison.csv", index=False)
    return table


def table5_per_class(df):
    """Table 5: Per-class F1 for top models from each arm."""
    # Find best model per arm
    e1 = df[df["data_fraction"] == 1.0]
    per_class_results = {}

    for arm in ["frozen_embedding", "handcrafted", "custom_cnn"]:
        arm_df = e1[e1["arm"] == arm]
        if arm_df.empty:
            continue
        grouped = arm_df.groupby(["model", "classifier"])["macro_f1"].mean()
        best_model, best_clf = grouped.idxmax()

        # Load per-class F1 files for this combo
        all_class_f1 = []
        first_data = None  # capture class names from first successfully loaded file
        for seed in cfg.SEEDS:
            frac_str = "1.00"
            fname = f"{best_model}_{best_clf}_seed{seed}_frac{frac_str}.json"
            if arm == "custom_cnn":
                fname = f"{best_model}_seed{seed}_frac{frac_str}.json"
            fpath = cfg.PER_CLASS_DIR / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                all_class_f1.append(data["f1_scores"])
                if first_data is None:
                    first_data = data

        if all_class_f1 and first_data is not None:
            mean_f1 = np.mean(all_class_f1, axis=0)
            per_class_results[f"{best_model}+{best_clf}"] = dict(zip(first_data["classes"], mean_f1.round(3)))

    if per_class_results:
        table = pd.DataFrame(per_class_results)
        print("\n=== TABLE 5: Per-Class F1 (Top Models) ===")
        print(table.to_string())
        table.to_csv(cfg.RESULTS_DIR / "table5_per_class.csv")
        return table
    return None


def table6_efficiency():
    """Table 6: Efficiency comparison — extraction time, inference time, model params."""
    timing_file = cfg.TIMING_DIR / "efficiency_results.json"
    if not timing_file.exists():
        print("No efficiency results found. Run E5 first.")
        return None

    with open(timing_file) as f:
        data = json.load(f)

    # Extraction times (one row per model — embedding and handcrafted arms)
    extract_rows = {e["model"]: e["per_sample_s"] for e in data.get("extraction", [])}

    # Classifier inference times: best (lowest) per embedding across classifiers
    infer_rows = {}
    for entry in data.get("classifier", []):
        emb = entry["embedding"]
        t = entry["per_sample_inference_s"]
        if emb not in infer_rows or t < infer_rows[emb]:
            infer_rows[emb] = t

    # CNN params
    cnn_rows = {e["model"]: (e["total_params"], e["model_size_mb"]) for e in data.get("cnn_params", [])}

    # CNN end-to-end inference time (audio → spectrogram → forward pass)
    cnn_infer_rows = {e["model"]: e["per_sample_s"] for e in data.get("cnn_inference", [])}

    rows = []
    all_models = (
        list(cfg.EMBEDDING_MODELS.keys())
        + list(cfg.HANDCRAFTED_FEATURES.keys())
        + list(cfg.CNN_ARCHITECTURES.keys())
    )
    for model in all_models:
        row = {"model": model}
        # For embedding/handcrafted: extraction time + classifier inference time
        # For CNN: end-to-end inference time (spectrogram + forward pass) in extraction_s_per_sample
        if model in cnn_infer_rows:
            row["extraction_s_per_sample"] = ""
            row["inference_s_per_sample"] = cnn_infer_rows[model]
        else:
            row["extraction_s_per_sample"] = extract_rows.get(model, "")
            row["inference_s_per_sample"] = infer_rows.get(model, "")
        if model in cnn_rows:
            row["total_params"] = cnn_rows[model][0]
            row["model_size_mb"] = cnn_rows[model][1]
        else:
            row["total_params"] = ""
            row["model_size_mb"] = ""
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\n=== TABLE 6: Efficiency ===")
    print(table.to_string(index=False))
    table.to_csv(cfg.RESULTS_DIR / "table6_efficiency.csv", index=False)
    return table


def main():
    df = load_results()
    print(f"Loaded {len(df)} experiment results")

    table3_embedding_results(df)
    table4_arm_comparison(df)
    table5_per_class(df)
    table6_efficiency()

    print("\nAll tables saved to", cfg.RESULTS_DIR)


if __name__ == "__main__":
    main()
