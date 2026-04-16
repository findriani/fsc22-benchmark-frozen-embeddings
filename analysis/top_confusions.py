"""
Top-confusion analysis from saved confusion matrices.

For each top model, finds the class pairs that are most often confused
and reports them as a ranked table. Useful for error analysis in the paper.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

# ── Class groupings for FSC22 ─────────────────────────────────────────────────
# Broad semantic categories to summarise where errors fall.
# Adjust class names here if FSC22 uses different spellings.
ANTHROPOGENIC_CLASSES = {
    "chainsaw", "axe", "handsaw", "tree_falling",
    "car", "motorcycle", "truck", "engine",
    "gunshot", "explosion",
    "footstep", "voices", "dog_bark",
}
# Everything not in ANTHROPOGENIC is treated as NATURAL (biotic/abiotic)

def classify_group(class_name):
    return "anthropogenic" if class_name.lower() in ANTHROPOGENIC_CLASSES else "natural"


def load_mean_confusion_matrix(model, clf, seeds=None, data_fraction=1.0):
    """Average confusion matrices across seeds for a given model+classifier."""
    if seeds is None:
        seeds = cfg.SEEDS

    matrices = []
    for seed in seeds:
        stem = f"{model}_{clf}_seed{seed}_frac{data_fraction:.2f}"
        cm_path = cfg.CONFUSION_DIR / f"{stem}.npy"
        if cm_path.exists():
            matrices.append(np.load(cm_path))

    if not matrices:
        return None, None

    mean_cm = np.mean(matrices, axis=0)

    # Load class names from the first matching per-class JSON
    for seed in seeds:
        stem = f"{model}_{clf}_seed{seed}_frac{data_fraction:.2f}"
        json_path = cfg.PER_CLASS_DIR / f"{stem}.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            return mean_cm, data["classes"]

    return mean_cm, None


def top_confusions(cm, class_names, top_n=10):
    """Extract top-N most common off-diagonal confusion pairs.

    Returns a list of dicts: true_class, predicted_class, count, confusion_rate
    sorted by raw confusion count descending.
    """
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        row_total = cm[i].sum()
        if row_total == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            count = cm[i, j]
            if count > 0:
                pairs.append({
                    "true_class": class_names[i],
                    "predicted_as": class_names[j],
                    "avg_count": round(float(count), 2),
                    "confusion_rate": round(float(count / row_total), 3),
                })

    pairs.sort(key=lambda x: x["avg_count"], reverse=True)
    return pairs[:top_n]


def main():
    out_dir = cfg.RESULTS_DIR / "confusion_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.read_csv(cfg.RESULTS_CSV)
    e1 = df_results[df_results["data_fraction"] == 1.0]

    all_top_confusions = []

    for arm in ["frozen_embedding", "handcrafted", "custom_cnn"]:
        arm_df = e1[e1["arm"] == arm]
        if arm_df.empty:
            continue

        # Best model per arm
        grouped = arm_df.groupby(["model", "classifier"])["macro_f1"].mean()
        best_model, best_clf = grouped.idxmax()
        mean_f1 = grouped.max()

        cm, class_names = load_mean_confusion_matrix(best_model, best_clf)
        if cm is None or class_names is None:
            print(f"[{arm}] No confusion matrices found for {best_model}+{best_clf}")
            continue

        pairs = top_confusions(cm, class_names, top_n=10)

        print(f"\n=== {arm.upper()} — {best_model}+{best_clf} (macro-F1={mean_f1:.3f}) ===")
        print(f"{'True Class':<30} {'Predicted As':<30} {'Avg Count':>10} {'Rate':>8}")
        print("-" * 82)
        for p in pairs:
            print(f"{p['true_class']:<30} {p['predicted_as']:<30} {p['avg_count']:>10} {p['confusion_rate']:>8.3f}")

        # Save per-arm table
        df_pairs = pd.DataFrame(pairs)
        df_pairs["arm"] = arm
        df_pairs["model"] = best_model
        df_pairs["classifier"] = best_clf
        all_top_confusions.append(df_pairs)

        # Save mean confusion matrix as CSV for this model
        cm_df = pd.DataFrame(
            np.round(cm).astype(int),
            index=class_names,
            columns=class_names,
        )
        cm_df.to_csv(out_dir / f"cm_{best_model}_{best_clf}.csv")

    if all_top_confusions:
        combined = pd.concat(all_top_confusions, ignore_index=True)
        combined.to_csv(out_dir / "top_confusions_all_arms.csv", index=False)

    # ── Anthropogenic vs Natural summary ──────────────────────────────────────
    # For the best overall model, summarise F1 grouped by class category.
    print("\n=== ANTHROPOGENIC vs NATURAL CLASS GROUP SUMMARY ===")
    best_arm_df = e1.copy()
    if not best_arm_df.empty:
        grouped_all = best_arm_df.groupby(["model", "classifier"])["macro_f1"].mean()
        top_model, top_clf = grouped_all.idxmax()

        group_f1 = {"anthropogenic": [], "natural": []}
        for seed in cfg.SEEDS:
            stem = f"{top_model}_{top_clf}_seed{seed}_frac1.00"
            json_path = cfg.PER_CLASS_DIR / f"{stem}.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                data = json.load(f)
            for cls, f1 in zip(data["classes"], data["f1_scores"]):
                group = classify_group(cls)
                group_f1[group].append(f1)

        summary_rows = []
        for group, scores in group_f1.items():
            if scores:
                summary_rows.append({
                    "group": group,
                    "n_class_observations": len(scores),
                    "mean_f1": round(np.mean(scores), 4),
                    "std_f1": round(np.std(scores), 4),
                    "min_f1": round(np.min(scores), 4),
                    "max_f1": round(np.max(scores), 4),
                })
                print(f"  {group:>15}: mean F1={np.mean(scores):.3f} ± {np.std(scores):.3f} "
                      f"(min={np.min(scores):.3f}, max={np.max(scores):.3f})")

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df["model"] = top_model
            summary_df["classifier"] = top_clf
            summary_df.to_csv(out_dir / "anthropogenic_vs_natural_summary.csv", index=False)

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
