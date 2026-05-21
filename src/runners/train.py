"""Training runner (CON-runner-cli).

    python -m src.runners.train --method <rf_hog|plain_cnn|stn_cnn>

Thin ``argparse`` orchestration: look up the method via the registry, load and
normalize GTSRB, fit, save the model, plot the training summary. The runner
owns normalization for ALL splits so the method's HOG extraction always sees
``[0,1]`` floats (Pitfall 2). Uses only the stdlib ``argparse`` parser — fancy
third-party CLI frameworks are disallowed (DEC-005 / PROJECT_PLAN.md §9).
"""

import argparse
import os

from src.common.data_loader import load_gtsrb
from src.common.evaluate import plot_training_curves
from src.common.preprocess import normalize
from src.methods.registry import get_method


def main():
    ap = argparse.ArgumentParser(
        description="Train a signbench method on GTSRB."
    )
    # choices=[...] validates --method before any work runs (V5 input validation).
    ap.add_argument(
        "--method",
        required=True,
        choices=["rf_hog", "plain_cnn", "stn_cnn"],
    )
    ap.add_argument("--data-dir", default="data/gtsrb")
    args = ap.parse_args()

    model = get_method(args.method)
    (Xtr, ytr), (Xva, yva), _ = load_gtsrb(args.data_dir)
    # Runner owns normalization for ALL splits (Pitfall 2).
    Xtr, Xva = normalize(Xtr), normalize(Xva)

    history = model.fit(Xtr, ytr, Xva, yva)

    os.makedirs("models", exist_ok=True)
    # Use the MODEL_EXT class attribute so the runner stays method-agnostic.
    model.save(f"models/{args.method}{model.MODEL_EXT}")

    os.makedirs(f"analysis/{args.method}", exist_ok=True)
    # Slice history to only the [0,1]-valued keys before plotting — passing
    # n_features (1764) / n_estimators (500) would clip the fixed [0,1] y-axis
    # of the single-point bar chart (Pitfall 4).
    curve = {k: history[k] for k in ("train_acc", "val_acc") if k in history}
    plot_training_curves(curve, f"analysis/{args.method}/training_curves.png")

    print(f"train: {args.method} done — history={history}")


if __name__ == "__main__":
    main()
