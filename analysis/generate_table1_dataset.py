"""
Generate Table 1: FSC22 class distribution (samples per class).

Reads from data/splits/metadata.json (must exist — run prepare_splits.py first).
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def main():
    meta_file = cfg.SPLITS_DIR / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {meta_file}. Run data/prepare_splits.py first."
        )

    with open(meta_file) as f:
        metadata = json.load(f)

    counts = Counter(metadata["labels"])
    rows = [{"class": cls, "samples": counts[cls]} for cls in sorted(counts.keys())]
    df = pd.DataFrame(rows)
    df = df.sort_values("samples", ascending=False).reset_index(drop=True)

    print(f"\n=== TABLE 1: FSC22 Class Distribution ({len(df)} classes, {df['samples'].sum()} total) ===")
    print(df.to_string(index=False))
    print(f"\nMin: {df['samples'].min()}, Max: {df['samples'].max()}, "
          f"Mean: {df['samples'].mean():.1f}, Median: {df['samples'].median():.1f}")

    out = cfg.RESULTS_DIR / "table1_dataset.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
