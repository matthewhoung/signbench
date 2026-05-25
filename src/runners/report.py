"""Cross-method comparison runner (CON-runner-cli).

    python -m src.runners.report [--analysis-dir analysis]

Unlike ``train`` / ``eval_gtsrb`` (one method per invocation, selected by a
``--method`` flag), this runner AGGREGATES all three trained methods
(``rf_hog``, ``plain_cnn``, ``stn_cnn``) in a single invocation — it takes no
per-method selection flag (DEC-002 — fixed 3-method roster). It is the final
stage of ``make all`` and produces the four ``analysis/comparison/`` artifacts
the rubric's 方法比較 (method comparison) section quotes directly:

- ``accuracy_comparison.png``  — bar chart of GTSRB test accuracy, all 3 methods
- ``loss_comparison.png``      — train/val loss curves, CNN methods ONLY
                                 (rf_hog has no epochs — DEC / criterion #2)
- ``per_class_accuracy.png``   — heatmap, 3 methods x 43 GTSRB classes
- ``final_table.md``           — drop-in markdown comparison table, 3 methods
                                 plus the Taiwan domain-transfer columns

The Taiwan columns of ``final_table.md`` are computed DIRECTLY from the
structured ``analysis/taiwan/predictions.csv`` (deterministic, schema-stable) —
NOT by parsing the free-text ``analysis/taiwan/summary.txt`` (its
variable-length comma-separated OOD class list makes naive parsing fragile).

Uses only the stdlib ``argparse`` parser — fancy third-party CLI frameworks are
disallowed (DEC-005 / PROJECT_PLAN.md §9). The runner ALWAYS runs to completion
and exits 0 (no accuracy gate — consistent with eval_gtsrb / eval_taiwan).
"""

# --- Headless backend selection. MUST happen BEFORE any pyplot import. ---
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Stdlib + third-party imports. ---
import argparse
import csv
import json
import os

from src.common.class_names import GTSRB_CLASSES

# Fixed 3-method roster (DEC-002) — also the row order of every artifact.
METHODS = ["rf_hog", "plain_cnn", "stn_cnn"]
# CNN methods only — rf_hog has no epoch-by-epoch curve, so it is excluded
# from loss_comparison.png (criterion #2).
CNN_METHODS = ["plain_cnn", "stn_cnn"]

# Class ids span the full GTSRB label space 0..42.
CLASS_IDS = list(range(43))


def _load_metrics(analysis_dir: str) -> dict:
    """Load each method's metrics.json. Returns {method: metrics_dict}.

    A missing/unreadable metrics.json yields an empty dict for that method so
    downstream code degrades gracefully rather than crashing.
    """
    metrics: dict[str, dict] = {}
    for method in METHODS:
        path = os.path.join(analysis_dir, method, "metrics.json")
        try:
            with open(path) as fp:
                metrics[method] = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"report: WARNING — could not read {path} ({exc})")
            metrics[method] = {}
    return metrics


def plot_accuracy_comparison(metrics: dict, out_path: str) -> None:
    """Bar chart of GTSRB test accuracy, one bar per method (criterion #2)."""
    accs = [float(metrics.get(m, {}).get("accuracy", 0.0)) for m in METHODS]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(METHODS, accs, color=["#8c8c8c", "#4c72b0", "#c44e52"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("GTSRB test accuracy")
    ax.set_title("GTSRB test accuracy by method")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(acc + 0.02, 0.97),
            f"{acc:.4f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"report: wrote {out_path}")


def plot_loss_comparison(analysis_dir: str, out_path: str) -> None:
    """Train/val loss curves over epochs for the CNN methods only.

    rf_hog is excluded — it has no epoch-keyed history (criterion #2). A missing
    training_history.json is skipped with a printed warning rather than crashing
    (a partial run should not hard-fail; the final make-all run produces them).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = 0
    for method in CNN_METHODS:
        path = os.path.join(analysis_dir, method, "training_history.json")
        try:
            with open(path) as fp:
                history = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"report: WARNING — skipping {method} loss curve ({exc})")
            continue
        if "loss" in history:
            ax.plot(history["loss"], label=f"{method} train")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label=f"{method} val", linestyle="--")
        drawn += 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training/validation loss — CNN methods")
    if drawn:
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "no training history available",
                ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"report: wrote {out_path}")


def plot_per_class_accuracy(analysis_dir: str, out_path: str) -> None:
    """Heatmap of per-class accuracy: 3 methods (rows) x 43 classes (cols).

    Reads each method's per_class.json. A missing file yields a zero row with a
    printed warning rather than crashing.
    """
    import numpy as np
    import seaborn as sns

    matrix = []
    for method in METHODS:
        path = os.path.join(analysis_dir, method, "per_class.json")
        try:
            with open(path) as fp:
                per_class = json.load(fp).get("per_class", {})
        except (OSError, json.JSONDecodeError) as exc:
            print(f"report: WARNING — {method} per_class.json missing ({exc})")
            per_class = {}
        matrix.append([float(per_class.get(str(cid), 0.0)) for cid in CLASS_IDS])

    data = np.asarray(matrix)
    # Short axis labels: "id name" keeps the 43-column axis legible.
    col_labels = [f"{cid} {GTSRB_CLASSES.get(cid, '')}" for cid in CLASS_IDS]

    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(
        data,
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        xticklabels=col_labels,
        yticklabels=METHODS,
        cbar=True,
        annot=False,
        ax=ax,
    )
    ax.set_xlabel("GTSRB class")
    ax.set_ylabel("Method")
    ax.set_title("Per-class accuracy: methods x 43 GTSRB classes")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"report: wrote {out_path}")


def _taiwan_columns(analysis_dir: str) -> dict:
    """Compute the per-method Taiwan final_table.md columns from predictions.csv.

    Returns {method: {"universal": "X/N", "ood_conf": float | None}}.

    The structured CSV (columns: filename, method, predicted_class, confidence,
    ground_truth, correct) is the source — summary.txt is NOT parsed (T-08-04).
    All field coercions are defensive: a malformed cell is skipped rather than
    crashing the whole table; the OOD-mean guards against a zero-row divide.
    """
    # Initialise counters per method so a method with zero CSV rows still
    # appears in the table.
    universal_total = {m: 0 for m in METHODS}
    universal_correct = {m: 0 for m in METHODS}
    ood_confidences: dict[str, list[float]] = {m: [] for m in METHODS}

    path = os.path.join(analysis_dir, "taiwan", "predictions.csv")
    try:
        with open(path, newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                method = row.get("method", "")
                if method not in METHODS:
                    continue
                # Coerce ground_truth defensively; skip a malformed row.
                try:
                    gt = int(row.get("ground_truth", ""))
                except (TypeError, ValueError):
                    continue
                if gt != -1:
                    # Universal-class sign.
                    universal_total[method] += 1
                    if str(row.get("correct", "")).strip().lower() == "true":
                        universal_correct[method] += 1
                else:
                    # OOD sign — collect its confidence for the mean.
                    try:
                        ood_confidences[method].append(
                            float(row.get("confidence", ""))
                        )
                    except (TypeError, ValueError):
                        continue
    except OSError as exc:
        print(f"report: WARNING — could not read {path} ({exc})")

    columns: dict[str, dict] = {}
    for method in METHODS:
        total = universal_total[method]
        correct = universal_correct[method]
        confs = ood_confidences[method]
        ood_conf = sum(confs) / len(confs) if confs else None
        columns[method] = {
            "universal": f"{correct}/{total}",
            "ood_conf": ood_conf,
        }
    return columns


def write_final_table(analysis_dir: str, metrics: dict, out_path: str) -> None:
    """Write the drop-in markdown comparison table (criterion #4).

    Columns: Method | GTSRB test accuracy | Precision | Recall | F1 |
             Taiwan universal correct | Taiwan OOD mean confidence.
    One row per method in METHODS order. GTSRB columns come from metrics.json;
    the two Taiwan columns from predictions.csv (via _taiwan_columns).
    """
    taiwan = _taiwan_columns(analysis_dir)

    header = (
        "| Method | GTSRB test accuracy | Precision | Recall | F1 "
        "| Taiwan universal correct | Taiwan OOD mean confidence |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- |"

    rows = []
    for method in METHODS:
        m = metrics.get(method, {})
        acc = float(m.get("accuracy", 0.0))
        prec = float(m.get("precision", 0.0))
        rec = float(m.get("recall", 0.0))
        f1 = float(m.get("f1", 0.0))
        tw = taiwan.get(method, {})
        universal = tw.get("universal", "0/0")
        ood_conf = tw.get("ood_conf")
        ood_str = "n/a" if ood_conf is None else f"{ood_conf:.3f}"
        rows.append(
            f"| {method} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} "
            f"| {universal} | {ood_str} |"
        )

    lines = [
        "## Cross-method comparison",
        "",
        "GTSRB test-split metrics for all three methods, plus the Taiwan "
        "domain-transfer probe (universal-class signs correct / total; mean "
        "softmax confidence on out-of-distribution Taiwan-unique signs).",
        "",
        header,
        separator,
        *rows,
        "",
    ]
    with open(out_path, "w") as fp:
        fp.write("\n".join(lines))
    print(f"report: wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate all three trained signbench methods into the four "
            "analysis/comparison/ cross-method artifacts."
        )
    )
    # No per-method selection flag — this runner always aggregates all three.
    ap.add_argument("--analysis-dir", default="analysis")
    args = ap.parse_args()

    analysis_dir = args.analysis_dir
    comparison_dir = os.path.join(analysis_dir, "comparison")
    # analysis/comparison/ does not exist yet — create it BEFORE any write.
    os.makedirs(comparison_dir, exist_ok=True)

    metrics = _load_metrics(analysis_dir)

    accuracy_png = os.path.join(comparison_dir, "accuracy_comparison.png")
    loss_png = os.path.join(comparison_dir, "loss_comparison.png")
    per_class_png = os.path.join(comparison_dir, "per_class_accuracy.png")
    final_table_md = os.path.join(comparison_dir, "final_table.md")

    plot_accuracy_comparison(metrics, accuracy_png)
    plot_loss_comparison(analysis_dir, loss_png)
    plot_per_class_accuracy(analysis_dir, per_class_png)
    write_final_table(analysis_dir, metrics, final_table_md)

    # NO accuracy assertion, NO non-zero exit — always exit 0.
    print(
        "report: done — wrote "
        f"{accuracy_png}, {loss_png}, {per_class_png}, {final_table_md}"
    )


if __name__ == "__main__":
    main()
