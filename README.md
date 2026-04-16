# FSC22 Frozen Embedding Benchmark

A systematic benchmark comparing frozen general-purpose audio foundation model embeddings against handcrafted features and custom-trained CNNs on the **FSC22 Forest Sound Classification dataset** (27 classes).

---

## Research Overview

**Question:** Can frozen embeddings from large pretrained audio models outperform custom CNNs on forest sound classification — especially in low-data scenarios?

**Three-arm comparison:**
| Arm | Models | Classifier |
|---|---|---|
| Frozen Embeddings | PANNs CNN14, BEATs, AST, CLAP, OpenL3, YAMNet, VGGish | LR, SVM, MLP |
| Handcrafted Features | MFCC stats, Log-mel stats | SVM, XGBoost |
| Custom CNN | MobileNetV2, ResNet-18, EfficientNet-B0, DenseNet-121 | Softmax head |

**Experiments:**
- **E1** — Full dataset benchmark (5 seeds × 7 embeddings × 3 classifiers + CNN baselines)
- **E3** — Low-data regime (10%, 25%, 50%, 100% training data)
- **E5** — Efficiency analysis (extraction time, inference speed, model size)
- **E6a** — Augmentation ablation: ResNet-18 with precomputed pitch-shift + time-stretch bank (3× training data)
- **E6b** — Augmentation ablation: CLAP+SVM with Gaussian noise on raw waveform

---

## Directory Structure

```
fsc22_benchmark/
├── config.py                    # All settings: paths, models, hyperparams, seeds
├── reproduce.py                 # Master runner — runs all phases in sequence
├── requirements.txt
│
├── data/
│   ├── download_fsc22.py        # Download FSC22 from Kaggle (~2 GB)
│   ├── prepare_splits.py        # Create stratified 70/15/15 splits for all seeds
│   └── precompute_augmentations.py  # One-time: generate pitch-shift + time-stretch bank (E6a)
│
├── extraction/
│   ├── extract_all.py           # Dispatcher: calls the right extractor per model
│   └── extractors/
│       ├── panns.py             # PANNs CNN14 — 2048-dim (panns_inference)
│       ├── beats.py             # BEATs     — 768-dim  (Microsoft UniLM)
│       ├── ast_model.py         # AST       — 768-dim  (HuggingFace)
│       ├── clap_model.py        # CLAP      — 512-dim  (laion-clap)
│       ├── openl3_model.py      # OpenL3    — 6144-dim (openl3)
│       ├── yamnet.py            # YAMNet    — 1024-dim (TensorFlow Hub)
│       ├── vggish.py            # VGGish    — 128-dim  (torchvggish)
│       ├── mfcc.py              # MFCC stats — 320-dim (librosa)
│       └── logmel_stats.py      # Log-mel stats — 1024-dim (librosa)
│
├── experiments/
│   ├── run_embedding_clf.py     # Frozen embedding + sklearn classifier (E1, E3)
│   ├── run_handcrafted.py       # Handcrafted features + SVM/XGBoost (E1, E3)
│   ├── run_cnn_baseline.py      # CNN training from scratch (E1, E3)
│   ├── run_augmented_cnn.py     # E6a: ResNet-18 with precomputed augmentation bank
│   └── measure_efficiency.py   # E5: timing and parameter counts
│
└── analysis/
    ├── generate_tables.py       # Tables 3–6 as CSV
    ├── generate_figures.py      # Figures 2–7 as PNG
    ├── statistical_tests.py     # Paired t-tests + Cohen's d
    └── top_confusions.py        # Top confused class pairs per arm
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/findriani/fsc22-frozen-embedding-benchmark.git
cd fsc22-frozen-embedding-benchmark
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

BEATs is not pip-installable — clone the source and the extractor will find it automatically:

```bash
git clone https://github.com/microsoft/unilm.git /workspace/unilm
```

The extractor looks for BEATs at `/workspace/unilm/beats` by default.
If you clone elsewhere, set the environment variable:
```bash
export BEATS_SRC=/your/path/to/unilm/beats
```

**BEATs checkpoint:** The extractor needs the `BEATs_iter3+ (AS2M)` fine-tuned checkpoint.
The Microsoft Azure download URL is sometimes restricted (HTTP 409). If auto-download fails,
download manually from the [BEATs GitHub page](https://github.com/microsoft/unilm/tree/master/beats)
and point to it with:
```bash
export BEATS_CKPT_PATH=/path/to/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

### 3. Download FSC22 dataset

```bash
python data/download_fsc22.py
```

This downloads FSC22 (~2 GB) from **Kaggle** (`irmiot22/fsc22-dataset`) and extracts it automatically.
Requires a Kaggle API token — either set the environment variable or place `kaggle.json` in `~/.kaggle/`:

```bash
export KAGGLE_API_TOKEN=your_kaggle_api_token
```

If you already have the dataset locally, skip the download and set:

```bash
export FSC22_AUDIO_DIR=/path/to/folder/containing/wav/files
```

### 4. Prepare splits

```bash
python data/prepare_splits.py
```

Creates stratified 70/15/15 train/val/test splits for all 5 seeds under `data/splits/`.

---

## Running Experiments

### Run everything

```bash
python reproduce.py
```

### Run individual phases

```bash
python reproduce.py --phase extraction     # Extract all embeddings (GPU, ~2h)
python reproduce.py --phase embeddings     # E1 frozen embedding experiments (~30 min)
python reproduce.py --phase handcrafted    # E1 handcrafted experiments (~5 min)
python reproduce.py --phase cnn            # E1 CNN training (GPU, ~8-15h)
python reproduce.py --phase lowdata        # E3 low-data regime (~16-30h)
python reproduce.py --phase efficiency     # E5 efficiency measurements (~1h)
```

### Smoke test (verify pipeline end-to-end before a full run)

```bash
python reproduce.py --smoke-test
```

Runs 1 embedding model, 1 classifier, 1 seed, 2 CNN epochs. Completes in ~10 minutes. Run this on a new pod before committing to a 30–50 hour full run.

---

## Output Files

```
results/
├── all_results.csv              # All experiment results (one row per run)
├── augmented_cnn_results.csv    # E6a: ResNet-18 + augmentation bank results (5 seeds)
├── progress.json                # Resume checkpoint
├── REPORT.md                    # Comprehensive results report
├── per_class/                   # Per-class F1/precision/recall JSON + classification report TXT
├── confusion_matrices/          # Confusion matrix NPY per experiment
├── training_curves/             # Epoch-by-epoch loss/val_f1 JSON (CNN only)
├── timing/
│   └── efficiency_results.json  # E5 full efficiency report (extraction, classifier, CNN inference)
├── manifests/                   # Run manifests (Python/package versions, CUDA info)
└── figures/
    ├── fig1_pipeline.xml        # Draw.io source for pipeline diagram (Fig 1)
    ├── fig2_arm_comparison.png  # Best model per arm bar chart
    ├── fig3_learning_curves.png # Macro-F1 vs training data fraction (E3)
    ├── fig4_per_class_heatmap.png # Per-class F1 heatmap (top models)
    ├── fig5_pareto.png          # Accuracy vs extraction speed scatter
    ├── fig6_all_models_ranked.png # All 13 models ranked by macro-F1
    └── fig7_classifier_sensitivity.png # LR vs SVM vs MLP per embedding
```

`all_results.csv` columns: `exp_id`, `arm`, `model`, `classifier`, `seed`, `data_fraction`, `n_train`, `n_val`, `n_test`, `macro_f1`, `weighted_f1`, `accuracy`, `precision`, `recall`, `val_macro_f1`, `best_params`, `repr_size`, `n_params`, `time_s`

---

## Generating Paper Outputs

After all experiments are done, download the `results/` folder and run locally:

```bash
python analysis/generate_tables.py    # Tables 3-6 as CSV
python analysis/generate_figures.py   # Figures 2-7 as PNG
python analysis/statistical_tests.py  # Significance tests (paired t-test, Cohen's d, BH-FDR)
python analysis/top_confusions.py     # Top confused class pairs per arm
```

---

## Estimated Compute (Single GPU, e.g. A40 / RTX 3090)

| Phase | Time |
|---|---|
| Embedding extraction (7 models) | ~1-2 hours |
| E1 embedding experiments (CPU) | ~1.5–2 hours |
| E1 CNN training (4 archs × 5 seeds) | ~8-15 hours |
| E3 low-data (all models × 4 fracs) | ~16-30 hours |
| E5 efficiency | ~1 hour |
| **Total** | **~28-50 hours** |

Embeddings are extracted once and cached as `.npz` files, so E3 reuses them without re-running the heavy models.

