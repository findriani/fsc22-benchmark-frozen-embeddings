"""
Generate Table 8: Low-data regime results (all models x all fractions).

Shows macro-F1 (mean +/- std over 5 seeds) for each model's best classifier
at each training data fraction (10%, 25%, 50%, 100%).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def main():
    df = pd.read_csv(cfg.RESULTS_CSV)
    print(f"Loaded {len(df)} results")

    rows = []
    for arm in ["frozen_embedding", "handcrafted", "custom_cnn"]:
        adf = df[df["arm"] == arm]
        e1 = adf[adf["data_fraction"] == 1.0]

        if arm == "custom_cnn":
            combos = [(m, "end_to_end") for m in e1["model"].unique()]
        else:
            g = e1.groupby(["model", "classifier"])["macro_f1"].mean()
            best_clf = g.groupby(level="model").idxmax()
            combos = [best_clf[m] for m in best_clf.index]

        for model, clf in combos:
            sub = adf[(adf["model"] == model) & (adf["classifier"] == clf)]
            for frac in cfg.DATA_FRACTIONS:
                vals = sub[sub["data_fraction"] == frac]["macro_f1"]
                if len(vals) == 0:
                    continue
                rows.append({
                    "arm": arm,
                    "model": model,
                    "classifier": clf,
                    "data_fraction": frac,
                    "macro_f1_mean": round(vals.mean(), 4),
                    "macro_f1_std": round(vals.std(), 4),
                    "result": f"{vals.mean():.3f} +/- {vals.std():.3f}",
                })

    result_df = pd.DataFrame(rows)

    # Pivot for readable output
    pivot = result_df.pivot_table(
        index=["arm", "model", "classifier"],
        columns="data_fraction",
        values="result",
        aggfunc="first",
    )
    pivot = pivot[[0.10, 0.25, 0.50, 1.00]]
    pivot.columns = ["10%", "25%", "50%", "100%"]

    print("\n=== TABLE 8: Low-Data Regime (Macro-F1, best classifier per model) ===")
    print(pivot.to_string())

    out = cfg.RESULTS_DIR / "table8_lowdata.csv"
    result_df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # Also save the pivot version
    pivot_out = cfg.RESULTS_DIR / "table8_lowdata_pivot.csv"
    pivot.to_csv(pivot_out)
    print(f"Saved pivot to {pivot_out}")


if __name__ == "__main__":
    main()
