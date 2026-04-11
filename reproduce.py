"""
FSC22 Benchmark Reproduction Script

Runs all experiments described in the paper in sequence.

Usage:
    python reproduce.py                        # run all phases
    python reproduce.py --phase extraction     # single phase
    python reproduce.py --smoke-test           # quick end-to-end check

Phases:
    extraction    Extract embeddings from all seven encoders (GPU recommended)
    embeddings    Frozen embedding + classifier experiments
    handcrafted   Handcrafted feature experiments
    cnn           CNN baseline training (GPU required)
    lowdata       Data-efficiency analysis across four training fractions
    augmentation  Augmentation fairness check (CNN and CLAP sides)
    efficiency    Extraction and inference timing measurements
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nFailed: {description}")
        sys.exit(result.returncode)


def phase_extraction():
    run(
        [sys.executable, "extraction/extract_all.py"],
        "Extracting embeddings from all seven encoders",
    )


def phase_embeddings(smoke_test=False):
    import config as cfg
    models = list(cfg.EMBEDDING_MODELS.keys())
    classifiers = list(cfg.CLASSIFIERS.keys())
    seeds = [cfg.SEEDS[0]] if smoke_test else cfg.SEEDS
    fractions = [1.0]

    if smoke_test:
        models = [models[0]]
        classifiers = [classifiers[0]]

    for model in models:
        for clf in classifiers:
            for seed in seeds:
                for frac in fractions:
                    run(
                        [
                            sys.executable, "experiments/run_embedding_clf.py",
                            "--model", model,
                            "--classifier", clf,
                            "--seed", str(seed),
                            "--data-fraction", str(frac),
                        ],
                        f"Embedding: {model} + {clf}, seed={seed}, frac={frac}",
                    )


def phase_lowdata():
    import config as cfg
    fractions = cfg.DATA_FRACTIONS

    for model in cfg.EMBEDDING_MODELS:
        for clf in cfg.CLASSIFIERS:
            for seed in cfg.SEEDS:
                for frac in fractions:
                    if frac == 1.0:
                        continue  # already run in phase_embeddings
                    run(
                        [
                            sys.executable, "experiments/run_embedding_clf.py",
                            "--model", model,
                            "--classifier", clf,
                            "--seed", str(seed),
                            "--data-fraction", str(frac),
                        ],
                        f"Low-data: {model} + {clf}, seed={seed}, frac={frac}",
                    )

    import config as cfg
    for arch in cfg.CNN_ARCHITECTURES:
        for seed in cfg.SEEDS:
            for frac in fractions:
                if frac == 1.0:
                    continue
                run(
                    [
                        sys.executable, "experiments/run_cnn_baseline.py",
                        "--arch", arch,
                        "--seed", str(seed),
                        "--data-fraction", str(frac),
                    ],
                    f"Low-data CNN: {arch}, seed={seed}, frac={frac}",
                )


def phase_handcrafted():
    import config as cfg
    for feature in cfg.HANDCRAFTED_FEATURES:
        for seed in cfg.SEEDS:
            for frac in cfg.DATA_FRACTIONS:
                run(
                    [
                        sys.executable, "experiments/run_handcrafted.py",
                        "--feature", feature,
                        "--seed", str(seed),
                        "--data-fraction", str(frac),
                    ],
                    f"Handcrafted: {feature}, seed={seed}, frac={frac}",
                )


def phase_cnn(smoke_test=False):
    import config as cfg
    archs = list(cfg.CNN_ARCHITECTURES.keys())
    seeds = [cfg.SEEDS[0]] if smoke_test else cfg.SEEDS

    if smoke_test:
        archs = [archs[0]]

    for arch in archs:
        for seed in seeds:
            run(
                [
                    sys.executable, "experiments/run_cnn_baseline.py",
                    "--arch", arch,
                    "--seed", str(seed),
                    "--data-fraction", "1.0",
                ],
                f"CNN: {arch}, seed={seed}",
            )


def phase_augmentation():
    import config as cfg
    for seed in cfg.SEEDS:
        run(
            [
                sys.executable, "experiments/run_augmented_cnn.py",
                "--seed", str(seed),
            ],
            f"Augmented CNN (ResNet-18 + pitch/time-stretch), seed={seed}",
        )
    for seed in cfg.SEEDS:
        run(
            [
                sys.executable, "experiments/run_augmented_embedding.py",
                "--seed", str(seed),
            ],
            f"Augmented embedding (CLAP + Gaussian noise), seed={seed}",
        )


def phase_efficiency():
    run(
        [sys.executable, "experiments/measure_efficiency.py"],
        "Measuring extraction and inference efficiency",
    )


PHASES = {
    "extraction": phase_extraction,
    "embeddings": phase_embeddings,
    "handcrafted": phase_handcrafted,
    "cnn": phase_cnn,
    "lowdata": phase_lowdata,
    "augmentation": phase_augmentation,
    "efficiency": phase_efficiency,
}

ALL_PHASES = [
    "extraction",
    "embeddings",
    "handcrafted",
    "cnn",
    "lowdata",
    "augmentation",
    "efficiency",
]


def main():
    parser = argparse.ArgumentParser(description="Reproduce FSC22 benchmark experiments")
    parser.add_argument(
        "--phase",
        choices=list(PHASES.keys()),
        help="Run a single phase. Omit to run all phases in order.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick end-to-end check: one model, one classifier, one seed, two CNN epochs.",
    )
    args = parser.parse_args()

    if args.phase:
        fn = PHASES[args.phase]
        import inspect
        if "smoke_test" in inspect.signature(fn).parameters:
            fn(smoke_test=args.smoke_test)
        else:
            fn()
    else:
        for name in ALL_PHASES:
            fn = PHASES[name]
            import inspect
            if "smoke_test" in inspect.signature(fn).parameters:
                fn(smoke_test=args.smoke_test)
            else:
                fn()

    print("\nAll phases complete. Run analysis scripts to generate figures and tables:")
    print("  python analysis/generate_figures.py")
    print("  python analysis/generate_tables.py")
    print("  python analysis/statistical_tests.py")


if __name__ == "__main__":
    main()
