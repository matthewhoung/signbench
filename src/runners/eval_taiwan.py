"""Taiwan domain-transfer evaluation runner (CON-runner-cli).

    python -m src.runners.eval_taiwan [--data-dir data/taiwan]

Unlike ``eval_gtsrb`` (one method per invocation, selected by a method flag),
this runner evaluates ALL THREE trained methods (``rf_hog``, ``plain_cnn``,
``stn_cnn``) in a single invocation — it takes no per-method selection flag. The
point of the phase is a cross-method domain-transfer comparison on real Taiwan
traffic-sign photos.

For each method it loads the trained model, runs ``predict`` on the Taiwan
images, and records one row per (image × method). The three artifacts written to
``analysis/taiwan/`` are:

- ``predictions.csv``       — columns filename, method, predicted_class,
                              confidence, ground_truth, correct
                              (CON-taiwan-predictions-schema); n_images × 3 rows.
- ``visualization_grid.png`` — the existing ``evaluate.plot_taiwan_grid`` helper
                              renders images-on-rows × methods-on-columns,
                              cells colour-coded green/red.
- ``summary.txt``           — one quotable one-liner per method.

Domain-transfer accuracy is EXPECTED to be poor (the trained models never saw
Taiwan signs; OOD rows have ground_truth ``-1`` so no model can be ``correct`` on
them). A low score is the finding, not a failure — the runner ALWAYS exits 0 and
the phase-completion gate is artifact existence + schema correctness, NOT an
accuracy threshold (T-07-04).
"""

import argparse
import csv
import os
from collections import Counter

from src.common.class_names import GTSRB_CLASSES
from src.common.evaluate import plot_taiwan_grid
from src.common.preprocess import normalize
from src.common.taiwan_loader import load_taiwan
from src.methods.registry import get_method

# Fixed method order — this is both the grid COLUMN order and the summary.txt
# LINE order. predictions[i] inner rows follow this same order.
METHODS = ["rf_hog", "plain_cnn", "stn_cnn"]

OUT_DIR = "analysis/taiwan"
PREDICTIONS_CSV = f"{OUT_DIR}/predictions.csv"
GRID_PNG = f"{OUT_DIR}/visualization_grid.png"
SUMMARY_TXT = f"{OUT_DIR}/summary.txt"

CSV_HEADER = [
    "filename",
    "method",
    "predicted_class",
    "confidence",
    "ground_truth",
    "correct",
]


def _class_name(class_id: int) -> str:
    """Human-readable name for a GTSRB class id (falls back to the bare id)."""
    return GTSRB_CLASSES.get(int(class_id), f"class {int(class_id)}")


def _summary_line(method: str, y, per_method_preds, per_method_confs) -> str:
    """Build the one-line domain-transfer summary for a single method.

    Splits the Taiwan images into universal (ground_truth != -1) and OOD
    (ground_truth == -1) and reports:
      "<method>: X/N universal-class signs correct,
       Y/M OOD signs predicted as <distinct class names>"
    """
    universal_idx = [i for i in range(len(y)) if int(y[i]) != -1]
    ood_idx = [i for i in range(len(y)) if int(y[i]) == -1]

    n_universal = len(universal_idx)
    correct_universal = sum(
        1 for i in universal_idx if int(per_method_preds[i]) == int(y[i])
    )

    n_ood = len(ood_idx)
    if n_ood:
        ood_preds = [int(per_method_preds[i]) for i in ood_idx]
        # Most-common-first ordering of the distinct predicted classes, so the
        # line reads "predicted as <dominant guess>, <next>, ...".
        ranked = [cid for cid, _ in Counter(ood_preds).most_common()]
        ood_names = ", ".join(_class_name(cid) for cid in ranked)
        ood_confs = [float(per_method_confs[i]) for i in ood_idx]
        mean_conf = sum(ood_confs) / len(ood_confs)
        ood_part = (
            f"{n_ood}/{n_ood} OOD signs predicted as {ood_names} "
            f"(mean confidence {mean_conf:.2f})"
        )
    else:
        ood_part = "0/0 OOD signs"

    return (
        f"{method}: {correct_universal}/{n_universal} universal-class signs "
        f"correct, {ood_part}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate all three trained signbench methods on the Taiwan "
            "traffic-sign photos (domain-transfer probe)."
        )
    )
    # No per-method selection flag — this runner always evaluates all three.
    ap.add_argument("--data-dir", default="data/taiwan")
    args = ap.parse_args()

    # 1. Load the Taiwan images: uint8 (N,32,32,3) X, int y with -1 for OOD.
    X, y, filenames = load_taiwan(args.data_dir)
    n_images = len(filenames)

    # 2. analysis/taiwan/ is gitignored — create it BEFORE any write.
    os.makedirs(OUT_DIR, exist_ok=True)

    # 3. Normalize once — shared float input for every method (mirrors eval_gtsrb).
    Xn = normalize(X)

    # 4. Run each method; collect rows + per-method predictions for the grid.
    rows: list[list] = []
    # predictions_by_method[method] -> list of predicted class ids, parallel to X.
    predictions_by_method: dict[str, list[int]] = {}
    confidences_by_method: dict[str, list[float]] = {}

    for method_name in METHODS:
        model = get_method(method_name)
        # MODEL_EXT keeps the path resolution method-agnostic (matches eval_gtsrb).
        model.load(f"models/{method_name}{model.MODEL_EXT}")
        preds, confs = model.predict(Xn)

        method_preds: list[int] = []
        method_confs: list[float] = []
        for i in range(n_images):
            pred_class = int(preds[i])
            confidence = float(confs[i])
            ground_truth = int(y[i])
            correct = pred_class == ground_truth  # always False for OOD (gt=-1)
            rows.append(
                [
                    filenames[i],
                    method_name,
                    pred_class,
                    confidence,
                    ground_truth,
                    correct,
                ]
            )
            method_preds.append(pred_class)
            method_confs.append(confidence)

        predictions_by_method[method_name] = method_preds
        confidences_by_method[method_name] = method_confs
        print(f"eval-taiwan: {method_name} done — {n_images} images predicted")

    # 5. Write predictions.csv — one row per (image × method) = n_images × 3.
    with open(PREDICTIONS_CSV, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    # 6. Build grid inputs and call the EXISTING plot_taiwan_grid helper.
    #    images: uint8 thumbnails (the helper does ax.imshow).
    #    predictions[i][j]: image i, method j — inner order follows METHODS.
    images = [X[i] for i in range(n_images)]
    predictions = [
        [predictions_by_method[m][i] for m in METHODS] for i in range(n_images)
    ]
    ground_truths = [int(y[i]) for i in range(n_images)]
    plot_taiwan_grid(images, predictions, ground_truths, filenames, METHODS, GRID_PNG)

    # 7. Write summary.txt — exactly one line per method, in METHODS order.
    summary_lines = [
        _summary_line(
            m, y, predictions_by_method[m], confidences_by_method[m]
        )
        for m in METHODS
    ]
    with open(SUMMARY_TXT, "w") as fp:
        fp.write("\n".join(summary_lines) + "\n")

    # 8. Completion message. NO accuracy assertion, NO non-zero exit — always 0.
    print(f"eval-taiwan: wrote {PREDICTIONS_CSV}, {GRID_PNG}, {SUMMARY_TXT}")
    for line in summary_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
