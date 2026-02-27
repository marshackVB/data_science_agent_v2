"""Run a minimal training pass to validate the pipeline before deployment."""

import argparse
import sys

from train import run_training

MODEL_TYPES = ["xgboost", "lightgbm", "catboost"]


def main():
    parser = argparse.ArgumentParser(description="Smoke test training pipeline before deployment")
    parser.add_argument(
        "--model",
        choices=MODEL_TYPES + ["all"],
        default="all",
        help="Model type to test (default: all)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        help="Fraction of data to sample (default: 0.1)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of Optuna trials per model (default: 3)",
    )
    args = parser.parse_args()

    models_to_run = MODEL_TYPES if args.model == "all" else [args.model]
    failures = []

    for model_type in models_to_run:
        print(f"\n--- Smoke test: {model_type} ---")
        try:
            run_training(
                sample_fraction=args.sample_fraction,
                n_trials=args.n_trials,
                model_type=model_type,
            )
            print(f"✓ {model_type} passed")
        except Exception as e:
            print(f"✗ {model_type} failed: {e}")
            failures.append((model_type, str(e)))

    if failures:
        print(f"\n{len(failures)} model(s) failed: {[m for m, _ in failures]}")
        sys.exit(1)

    print(f"\nAll {len(models_to_run)} model(s) passed smoke test.")


if __name__ == "__main__":
    main()
