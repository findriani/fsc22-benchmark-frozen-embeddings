"""
Statistical significance tests for benchmark results.

Runs paired t-tests comparing arms and models.
Reports p-values and effect sizes (Cohen's d).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def cohens_d_paired(x, y):
    """Compute Cohen's d_z for paired samples (matches paired t-test).

    d_z = mean(x - y) / std(x - y, ddof=1)
    """
    diff = np.asarray(x) - np.asarray(y)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return np.mean(diff) / sd


def paired_comparison(df, model_a, clf_a, model_b, clf_b, metric="macro_f1"):
    """Run paired t-test between two model-classifier combos across seeds."""
    scores_a = df[(df["model"] == model_a) & (df["classifier"] == clf_a)].sort_values("seed")[metric].values
    scores_b = df[(df["model"] == model_b) & (df["classifier"] == clf_b)].sort_values("seed")[metric].values

    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return None

    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    d = cohens_d_paired(scores_a, scores_b)

    return {
        "model_a": f"{model_a}+{clf_a}",
        "model_b": f"{model_b}+{clf_b}",
        "mean_a": round(np.mean(scores_a), 4),
        "mean_b": round(np.mean(scores_b), 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "significant_005": p_value < 0.05,
        "cohens_d": round(d, 4),
    }


def main():
    df = pd.read_csv(cfg.RESULTS_CSV)
    e1 = df[df["data_fraction"] == 1.0].copy()
    print(f"Loaded {len(e1)} E1 results")

    # Find best model per arm
    best_per_arm = {}
    for arm in ["frozen_embedding", "handcrafted", "custom_cnn"]:
        arm_df = e1[e1["arm"] == arm]
        if arm_df.empty:
            continue
        grouped = arm_df.groupby(["model", "classifier"])["macro_f1"].mean()
        best_model, best_clf = grouped.idxmax()
        best_per_arm[arm] = (best_model, best_clf)
        print(f"Best {arm}: {best_model}+{best_clf} (mean F1={grouped.max():.4f})")

    # Pairwise comparisons between arms
    print("\n=== PAIRWISE COMPARISONS (Paired t-test, 5 seeds) ===")
    arms = list(best_per_arm.keys())
    results = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a_model, a_clf = best_per_arm[arms[i]]
            b_model, b_clf = best_per_arm[arms[j]]
            result = paired_comparison(e1, a_model, a_clf, b_model, b_clf)
            if result:
                result["comparison"] = f"{arms[i]} vs {arms[j]}"
                results.append(result)
                sig = "*" if result["significant_005"] else ""
                print(f"\n{arms[i]} vs {arms[j]}:")
                print(f"  {result['model_a']}: {result['mean_a']:.4f}")
                print(f"  {result['model_b']}: {result['mean_b']:.4f}")
                print(f"  t={result['t_statistic']:.4f}, p={result['p_value']:.4f}{sig}, d={result['cohens_d']:.4f}")

    # Compare all frozen embeddings against each other (best classifier per embedding)
    print("\n=== EMBEDDING COMPARISONS (Best classifier per embedding) ===")
    emb_best = {}
    emb_df = e1[e1["arm"] == "frozen_embedding"]
    if not emb_df.empty:
        for emb in emb_df["model"].unique():
            sub = emb_df[emb_df["model"] == emb]
            best_clf = sub.groupby("classifier")["macro_f1"].mean().idxmax()
            emb_best[emb] = best_clf

        emb_names = sorted(emb_best.keys())
        for i in range(len(emb_names)):
            for j in range(i + 1, len(emb_names)):
                result = paired_comparison(
                    e1, emb_names[i], emb_best[emb_names[i]],
                    emb_names[j], emb_best[emb_names[j]]
                )
                if result:
                    results.append(result)
                    sig = "*" if result["significant_005"] else ""
                    print(f"  {result['model_a']} vs {result['model_b']}: "
                          f"p={result['p_value']:.4f}{sig}, d={result['cohens_d']:.4f}")

    # Apply Benjamini-Hochberg FDR correction across all tests
    if results:
        results_df = pd.DataFrame(results)
        p_values = results_df["p_value"].values
        reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        results_df["p_value_fdr"] = p_corrected.round(4)
        results_df["significant_fdr_005"] = reject

        print("\n=== AFTER BENJAMINI-HOCHBERG FDR CORRECTION (alpha=0.05) ===")
        sig_before = results_df["significant_005"].sum()
        sig_after = results_df["significant_fdr_005"].sum()
        print(f"  Significant before correction: {sig_before} / {len(results_df)}")
        print(f"  Significant after FDR correction: {sig_after} / {len(results_df)}")
        for _, row in results_df.iterrows():
            marker = "*" if row["significant_fdr_005"] else ""
            print(f"  {row.get('comparison', row['model_a'] + ' vs ' + row['model_b'])}: "
                  f"p_raw={row['p_value']:.4f}, p_fdr={row['p_value_fdr']:.4f}{marker}")

        out_file = cfg.RESULTS_DIR / "statistical_tests.csv"
        results_df.to_csv(out_file, index=False)
        print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
