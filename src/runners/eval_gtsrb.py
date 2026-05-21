"""GTSRB evaluation runner (CON-runner-cli).

    python -m src.runners.eval_gtsrb --method <rf_hog|plain_cnn|stn_cnn>

Thin ``argparse`` orchestration: load a trained model, predict on the GTSRB
test split, write ``metrics.json`` plus the confusion-matrix PNG and
classification-report text. The runner ALWAYS runs to completion and exits 0 —
it does NOT enforce the >=90% accuracy gate (the phase verifier reads
``metrics.json`` separately). Uses only the stdlib ``argparse`` parser.
"""

import argparse
import json
import os

from src.common.data_loader import load_gtsrb
from src.common.evaluate import (
    classification_report_text,
    compute_metrics,
    plot_confusion_matrix,
)
from src.common.preprocess import normalize
from src.methods.registry import get_method


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a signbench method on the GTSRB test split."
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
    # Use the MODEL_EXT class attribute so the runner stays method-agnostic.
    model.load(f"models/{args.method}{model.MODEL_EXT}")

    _, _, (Xte, yte) = load_gtsrb(args.data_dir)
    Xte = normalize(Xte)
    preds, _conf = model.predict(Xte)

    out = f"analysis/{args.method}"
    # analysis/ is gitignored — create the dir BEFORE any write (Pitfall 3).
    os.makedirs(out, exist_ok=True)

    metrics = compute_metrics(yte, preds)
    # Provenance fields per the research metrics.json schema (Phase 8 report.py
    # aggregates across methods without parsing directory paths).
    metrics["method"] = args.method
    metrics["split"] = "test"
    metrics["n_samples"] = int(len(yte))
    with open(f"{out}/metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    plot_confusion_matrix(yte, preds, f"{out}/confusion_matrix.png")
    with open(f"{out}/classification_report.txt", "w") as fp:
        fp.write(classification_report_text(yte, preds))

    # NO accuracy assertion, NO non-zero exit on a low score — always exit 0.
    print(f"eval-gtsrb: {args.method} done — accuracy={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
