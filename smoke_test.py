"""Run a minimal training pass to validate the pipeline before deployment."""

from train import run_training

if __name__ == "__main__":
    run_training(
        sample_fraction=0.1,
        n_trials=3,
    )
