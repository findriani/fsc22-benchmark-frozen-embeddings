"""
Generate paper figures from experiment results.

Figures:
  Fig 2: Bar chart — overall comparison across arms
  Fig 3: Learning curves — macro-F1 vs % training data (E3)
  Fig 4: Heatmap — per-class F1 across top models
  Fig 5: Scatter — accuracy vs model size / inference time
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

FIGURES_DIR = cfg.RESULTS_DIR / "figures"


def load_results():
    return pd.read_csv(cfg.RESULTS_CSV)


def fig2_arm_comparison(df):
    """Bar chart comparing best model per arm."""
    e1 = df[df["data_fraction"] == 1.0]
    arms = {"frozen_embedding": "Frozen Embedding", "handcrafted": "Handcrafted", "custom_cnn": "Custom CNN"}

    means, stds, labels = [], [], []
    for arm_key, arm_label in arms.items():
        arm_df = e1[e1["arm"] == arm_key]
        if arm_df.empty:
            continue
        grouped = arm_df.groupby(["model", "classifier"])["macro_f1"].agg(["mean", "std"])
        best = grouped.loc[grouped["mean"].idxmax()]
        means.append(best["mean"])
        stds.append(best["std"])
        labels.append(arm_label)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=["#2196F3", "#FF9800", "#4CAF50"][:len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{m:.3f}",
                ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_arm_comparison.png", dpi=300)
    plt.close()
    print("Saved fig2_arm_comparison.png")


def fig3_learning_curves(df):
    """Learning curves: macro-F1 vs data fraction for top models."""
    arms = {"frozen_embedding": "Frozen Embedding", "handcrafted": "Handcrafted", "custom_cnn": "Custom CNN"}

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"frozen_embedding": "#2196F3", "handcrafted": "#FF9800", "custom_cnn": "#4CAF50"}

    for arm_key, arm_label in arms.items():
        arm_df = df[df["arm"] == arm_key]
        if arm_df.empty:
            continue

        # Find best model+classifier at full data
        full = arm_df[arm_df["data_fraction"] == 1.0]
        if full.empty:
            continue
        grouped = full.groupby(["model", "classifier"])["macro_f1"].mean()
        best_model, best_clf = grouped.idxmax()

        # Get results across fractions
        subset = arm_df[(arm_df["model"] == best_model) & (arm_df["classifier"] == best_clf)]
        curve = subset.groupby("data_fraction")["macro_f1"].agg(["mean", "std"]).reset_index()
        curve = curve.sort_values("data_fraction")

        ax.errorbar(
            curve["data_fraction"] * 100, curve["mean"], yerr=curve["std"],
            marker="o", label=f"{arm_label}\n({best_model}+{best_clf})",
            color=colors[arm_key], capsize=4, linewidth=2,
        )

    ax.set_xlabel("Training Data (%)")
    ax.set_ylabel("Macro-F1")
    ax.legend(loc="lower right")
    ax.set_xticks([10, 25, 50, 100])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_learning_curves.png", dpi=300)
    plt.close()
    print("Saved fig3_learning_curves.png")


def fig4_per_class_heatmap(df):
    """Heatmap of per-class F1 for top models."""
    e1 = df[df["data_fraction"] == 1.0]
    model_labels = []
    class_f1_matrix = []

    for arm in ["frozen_embedding", "handcrafted", "custom_cnn"]:
        arm_df = e1[e1["arm"] == arm]
        if arm_df.empty:
            continue
        grouped = arm_df.groupby(["model", "classifier"])["macro_f1"].mean()
        best_model, best_clf = grouped.idxmax()

        all_f1 = []
        classes = None
        for seed in cfg.SEEDS:
            frac_str = "1.00"
            if arm == "custom_cnn":
                fname = f"{best_model}_seed{seed}_frac{frac_str}.json"
            else:
                fname = f"{best_model}_{best_clf}_seed{seed}_frac{frac_str}.json"
            fpath = cfg.PER_CLASS_DIR / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                all_f1.append(data["f1_scores"])
                classes = data["classes"]

        if all_f1 and classes:
            mean_f1 = np.mean(all_f1, axis=0)
            class_f1_matrix.append(mean_f1)
            model_labels.append(f"{best_model}+{best_clf}")

    if not class_f1_matrix:
        print("No per-class data available for heatmap")
        return

    matrix = np.array(class_f1_matrix)
    fig, ax = plt.subplots(figsize=(16, max(4, len(model_labels) * 1.2)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, label="F1 Score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_per_class_heatmap.png", dpi=300)
    plt.close()
    print("Saved fig4_per_class_heatmap.png")


def fig5_pareto(df):
    """Scatter: macro-F1 vs extraction time per sample (from E5 efficiency results)."""
    timing_file = cfg.TIMING_DIR / "efficiency_results.json"
    if not timing_file.exists():
        print("Skipping fig5: no efficiency_results.json found. Run E5 first.")
        return

    with open(timing_file) as f:
        eff = json.load(f)

    # Build lookup: model -> per-sample time
    # For embedding/handcrafted arms: extraction time
    # For CNN arm: end-to-end inference time
    extract_time = {e["model"]: e["per_sample_s"] for e in eff.get("extraction", [])}
    cnn_infer_time = {e["model"]: e["per_sample_s"] for e in eff.get("cnn_inference", [])}
    all_times = {**extract_time, **cnn_infer_time}

    if not all_times:
        print("Skipping fig5: no timing data found.")
        return

    e1 = df[df["data_fraction"] == 1.0]
    arm_colors = {"frozen_embedding": "#2196F3", "handcrafted": "#FF9800", "custom_cnn": "#4CAF50"}
    arm_markers = {"frozen_embedding": "o", "handcrafted": "s", "custom_cnn": "^"}

    fig, ax = plt.subplots(figsize=(10, 7))
    for arm in ["frozen_embedding", "handcrafted", "custom_cnn"]:
        arm_df = e1[e1["arm"] == arm]
        if arm_df.empty:
            continue
        # Best macro-F1 per model: mean over seeds per (model, classifier) combo,
        # then take the max combo per model — matches the best-setup shown elsewhere.
        per_combo = arm_df.groupby(["model", "classifier"])["macro_f1"].mean()
        best_f1 = per_combo.groupby(level="model").max()
        for model_name, mean_f1 in best_f1.items():
            t = all_times.get(model_name)
            if t is None:
                continue
            ax.scatter(
                t, mean_f1,
                c=arm_colors[arm], marker=arm_markers[arm],
                s=90, alpha=0.85,
                label=arm.replace("_", " ").title() if model_name == best_f1.index[0] else "_nolegend_",
            )
            ax.annotate(
                model_name, (t, mean_f1),
                fontsize=7, alpha=0.85, textcoords="offset points", xytext=(5, 5),
            )

    ax.set_xlabel("Inference Time per Sample (s)")
    ax.set_ylabel("Macro-F1")
    # Deduplicate legend entries
    handles, labels_leg = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_pareto.png", dpi=300)
    plt.close()
    print("Saved fig5_pareto.png")


def fig6_all_models_ranked(df):
    """Horizontal bar chart: all models ranked by best macro-F1, coloured by arm."""
    e1 = df[df["data_fraction"] == 1.0]
    arm_colors = {"frozen_embedding": "#2196F3", "handcrafted": "#FF9800", "custom_cnn": "#4CAF50"}
    arm_labels = {"frozen_embedding": "Frozen Embedding", "handcrafted": "Handcrafted", "custom_cnn": "Custom CNN"}

    rows = []
    for arm in ["frozen_embedding", "handcrafted", "custom_cnn"]:
        arm_df = e1[e1["arm"] == arm]
        if arm_df.empty:
            continue
        grouped = arm_df.groupby(["model", "classifier"])["macro_f1"].agg(["mean", "std"])
        for model, model_group in grouped.groupby(level="model"):
            best_idx = model_group["mean"].idxmax()
            best = model_group.loc[best_idx]
            rows.append({
                "label": f"{best_idx[0]}\n(+{best_idx[1]})",
                "mean": best["mean"],
                "std": best["std"],
                "arm": arm,
            })

    rows = sorted(rows, key=lambda r: r["mean"])
    labels = [r["label"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    colors = [arm_colors[r["arm"]] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(rows))
    bars = ax.barh(y, means, xerr=stds, capsize=3, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0.5, 1.0)
    ax.axvline(x=means[-1], color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Value labels
    for bar, m in zip(bars, means):
        ax.text(m + 0.005, bar.get_y() + bar.get_height() / 2, f"{m:.3f}",
                va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=arm_colors[k], label=v) for k, v in arm_labels.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_all_models_ranked.png", dpi=300)
    plt.close()
    print("Saved fig6_all_models_ranked.png")


def fig7_classifier_sensitivity(df):
    """Grouped bar chart: LR vs SVM vs MLP macro-F1 for each embedding model."""
    e1 = df[(df["arm"] == "frozen_embedding") & (df["data_fraction"] == 1.0)]
    models = sorted(e1["model"].unique())
    classifiers = ["lr", "svm", "mlp"]
    clf_colors = {"lr": "#42A5F5", "svm": "#EF5350", "mlp": "#66BB6A"}
    clf_labels = {"lr": "LR (linear probe)", "svm": "SVM (RBF)", "mlp": "MLP"}

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, clf in enumerate(classifiers):
        means, stds = [], []
        for model in models:
            sub = e1[(e1["model"] == model) & (e1["classifier"] == clf)]["macro_f1"]
            means.append(sub.mean() if len(sub) > 0 else 0)
            stds.append(sub.std() if len(sub) > 0 else 0)
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=clf_labels[clf], color=clf_colors[clf], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.55, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_classifier_sensitivity.png", dpi=300)
    plt.close()
    print("Saved fig7_classifier_sensitivity.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    print(f"Loaded {len(df)} results")

    fig2_arm_comparison(df)
    fig3_learning_curves(df)
    fig4_per_class_heatmap(df)
    fig5_pareto(df)
    fig6_all_models_ranked(df)
    fig7_classifier_sensitivity(df)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
