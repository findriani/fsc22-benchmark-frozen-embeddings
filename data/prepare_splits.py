"""
Prepare stratified train/val/test splits for FSC22.

Protocol: 5 REPEATED STRATIFIED TRAIN/VAL/TEST SPLITS (not 5-fold CV).
  - Each seed produces one independent 70/15/15 split of the full dataset.
  - The val set is used only for hyperparameter selection (grid search).
  - The test set is held out and never used during training or model selection.
  - Results are reported as mean ± std across the 5 seeds.

This is NOT k-fold cross-validation. Every split uses all data, but the
random partition differs per seed, giving variance estimates without the
nested-CV complexity required by k-fold when a val set is also needed.

Low-data subsets (10%, 25%, 50%) are stratified subsamples of the
training split only; val and test sets remain full-size in all cases.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def load_fsc22_metadata():
    """Load FSC22 metadata and return file paths + labels."""
    audio_dir = cfg.FSC22_AUDIO_DIR

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"FSC22 audio directory not found at {audio_dir}. "
            f"Set FSC22_AUDIO_DIR environment variable or place data in {audio_dir}"
        )

    # FSC22 is organized as: FSC22/<class_name>/<audio_file>.wav
    records = []
    for class_dir in sorted(audio_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for audio_file in sorted(class_dir.glob("*.wav")):
            records.append({
                "file_path": str(audio_file),
                "class_name": class_name,
                "filename": audio_file.name,
            })

    if not records:
        # Flat structure: all WAV files in one folder, labels from metadata CSV
        if cfg.FSC22_METADATA.exists():
            df = pd.read_csv(cfg.FSC22_METADATA)
            print(f"Loaded metadata CSV with {len(df)} entries")
            print(f"Columns: {list(df.columns)}")

            # FSC22 official CSV uses 'Dataset File Name' and 'Class Name'
            if "Dataset File Name" in df.columns and "Class Name" in df.columns:
                df = df.rename(columns={
                    "Dataset File Name": "filename",
                    "Class Name": "class_name",
                })
                df["file_path"] = df["filename"].apply(
                    lambda fn: str(audio_dir / fn)
                )
            elif "file_path" not in df.columns or "class_name" not in df.columns:
                raise ValueError(
                    f"Metadata CSV must have ('Dataset File Name', 'Class Name') "
                    f"or ('file_path', 'class_name') columns. "
                    f"Found: {list(df.columns)}"
                )

            # Verify files exist and drop missing
            before = len(df)
            df = df[df["file_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
            if len(df) < before:
                print(f"WARNING: {before - len(df)} files listed in CSV not found on disk (dropped)")

            print(f"Found {len(df)} audio files across {df['class_name'].nunique()} classes")
            return df

        raise FileNotFoundError(
            f"No audio files found in {audio_dir} and no metadata CSV at {cfg.FSC22_METADATA}. "
            f"Set FSC22_METADATA environment variable."
        )

    df = pd.DataFrame(records)
    print(f"Found {len(df)} audio files across {df['class_name'].nunique()} classes")
    print(f"Class distribution:\n{df['class_name'].value_counts().to_string()}")
    return df


def create_splits(df, seed):
    """Create stratified train/val/test split for a given seed."""
    labels = df["class_name"].values
    indices = np.arange(len(df))

    # First split: train+val vs test (85/15)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=labels
    )

    # Second split: train vs val (70/15 of total = ~82.4/17.6 of train_val)
    val_ratio = 0.15 / 0.85  # ~0.176
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, random_state=seed,
        stratify=labels[train_val_idx]
    )

    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def create_lowdata_subsets(train_indices, labels, seed, fractions):
    """Create stratified subsets of training data."""
    subsets = {}
    for frac in fractions:
        if frac >= 1.0:
            subsets[f"{frac:.2f}"] = train_indices
            continue
        n_keep = max(1, int(len(train_indices) * frac))
        train_labels = [labels[i] for i in train_indices]
        subset_idx, _ = train_test_split(
            train_indices, train_size=n_keep, random_state=seed,
            stratify=train_labels
        )
        subsets[f"{frac:.2f}"] = sorted(subset_idx)
    return subsets


def main():
    print("=" * 60)
    print("Preparing FSC22 splits")
    print("=" * 60)

    df = load_fsc22_metadata()
    labels = df["class_name"].values

    cfg.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Save metadata mapping (index → file_path, class)
    metadata = {
        "files": df["file_path"].tolist() if "file_path" in df.columns else [],
        "labels": df["class_name"].tolist(),
        "classes": sorted(df["class_name"].unique().tolist()),
        "n_samples": len(df),
        "n_classes": df["class_name"].nunique(),
    }
    with open(cfg.SPLITS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata['n_samples']} samples, {metadata['n_classes']} classes")

    for seed in cfg.SEEDS:
        split = create_splits(df, seed)
        print(f"\nSeed {seed}: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

        # Low-data subsets
        low_data = create_lowdata_subsets(
            split["train"], labels, seed, cfg.DATA_FRACTIONS
        )

        split_data = {
            "seed": seed,
            "split": split,
            "low_data_train": low_data,
        }

        split_file = cfg.SPLITS_DIR / f"split_seed{seed}.json"
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"  Saved to {split_file}")

        for frac_key, indices in low_data.items():
            print(f"  Low-data {frac_key}: {len(indices)} samples")

    print("\nDone!")


if __name__ == "__main__":
    main()
