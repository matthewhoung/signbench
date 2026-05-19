"""Evaluation suite (CON-evaluate). Metrics + plots used by every Phase 4–8 runner. Headless-safe (matplotlib Agg backend).

Five functions per PROJECT_PLAN.md §4 and ``.planning/intel/constraints.md``
(CON-evaluate):

- ``compute_metrics(y_true, y_pred) -> dict``
- ``plot_training_curves(history, out_path) -> None``
- ``plot_confusion_matrix(y_true, y_pred, out_path) -> None``
- ``plot_taiwan_grid(images, predictions, ground_truths, filenames, methods, out_path) -> None``
- ``classification_report_text(y_true, y_pred) -> str``

The module selects matplotlib's non-interactive ``Agg`` backend at import time —
BEFORE the first ``pyplot`` import — so importing this module on a no-display
host (WSL2, CI, agent shells) succeeds and every plotting function writes PNGs
without a window server. Every plot function calls ``plt.close(...)`` after
``savefig`` to free figures (Phase 4–8 runners can produce many plots per run,
so leaks would matter — T-02-06).
"""

# --- Headless backend selection. MUST happen BEFORE any pyplot import. ---
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Stdlib + third-party imports. ---
import os

import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.common.class_names import GTSRB_CLASSES

# Apply seaborn's default theme once at import time so heatmaps look consistent
# across every runner. Safe to call repeatedly but cheaper to do once here.
sns.set_theme()


def compute_metrics(y_true, y_pred) -> dict:
    """Compute accuracy / precision / recall / F1 per CON-evaluate.

    Returns a dict with exactly the keys ``{"accuracy", "precision", "recall",
    "f1"}``, all values plain Python floats in ``[0.0, 1.0]``. Precision /
    recall / F1 use ``average="weighted"`` (handles GTSRB's 43-class multi-class
    case cleanly without a macro-vs-micro debate at this layer — per-class
    detail lives in ``classification_report_text``). ``zero_division=0``
    suppresses sklearn's degenerate-split warnings.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def _ensure_parent_dir(out_path: str) -> None:
    """Create the parent directory of ``out_path`` if it doesn't already exist.

    Tolerates ``out_path`` with no parent component (e.g. ``"plot.png"``).
    """
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _is_epoch_keyed(history: dict) -> bool:
    """Return True if ``history`` looks like an epoch-keyed dict (CNN case).

    Epoch-keyed: at least one value is a list/array with len > 1.
    Single-point (RF case): values are scalars or length-1 arrays.
    """
    for value in history.values():
        if hasattr(value, "__len__") and len(value) > 1:
            return True
    return False


def plot_training_curves(history, out_path) -> None:
    """Plot training curves (epoch-keyed or single-point) to ``out_path``.

    Two branches:

    - **Multi-epoch** (CNN case): ``history`` has lists/arrays of length > 1
      under keys like ``"accuracy"`` / ``"val_accuracy"`` / ``"loss"`` /
      ``"val_loss"``. Renders two side-by-side subplots (accuracy left, loss
      right) with validation overlays where present.
    - **Single-point** (RF case): ``history`` has scalar values, e.g.
      ``{"accuracy": 0.93}``. Renders a single subplot as a bar chart with the
      y-axis fixed to ``[0, 1]``.

    Writes a non-zero-byte PNG to ``out_path``. Creates the parent directory if
    it doesn't already exist. Closes the figure after ``savefig`` (T-02-06).
    """
    _ensure_parent_dir(out_path)

    if _is_epoch_keyed(history):
        # Multi-epoch CNN case: accuracy on the left, loss on the right.
        fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

        if "accuracy" in history:
            ax_acc.plot(history["accuracy"], label="train")
        if "val_accuracy" in history:
            ax_acc.plot(history["val_accuracy"], label="val")
        ax_acc.set_title("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0.0, 1.0)
        ax_acc.legend(loc="lower right")

        if "loss" in history:
            ax_loss.plot(history["loss"], label="train")
        if "val_loss" in history:
            ax_loss.plot(history["val_loss"], label="val")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return

    # Single-point RF case: bar chart of each scalar metric.
    fig, ax = plt.subplots(figsize=(6, 5))
    keys = list(history.keys())
    values = [float(history[k]) for k in keys]
    ax.bar(keys, values, color="steelblue")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Training summary (single-point)")
    ax.set_ylabel("Value")
    for i, v in enumerate(values):
        ax.text(i, min(v + 0.02, 0.98), f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, out_path) -> None:
    """Render the confusion matrix as a heatmap to ``out_path``.

    Tick labels: when all unique class ids fall within ``GTSRB_CLASSES`` (i.e.
    ``0..42``), use the human-readable names; otherwise fall back to the
    integer ids. x-tick labels are rotated 90° for legibility on the 43-class
    GTSRB axis.

    Writes a non-zero-byte PNG. Creates the parent directory if needed. Closes
    all figures after ``savefig`` (T-02-06).
    """
    _ensure_parent_dir(out_path)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_ids = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))

    if all(int(cid) in GTSRB_CLASSES for cid in unique_ids):
        labels = [GTSRB_CLASSES[int(cid)] for cid in unique_ids]
    else:
        labels = [str(int(cid)) for cid in unique_ids]

    cm = confusion_matrix(y_true, y_pred, labels=unique_ids)

    # Scale the figure with the number of classes so 43-class GTSRB stays legible.
    side = max(6, int(0.4 * len(unique_ids)) + 4)
    fig, ax = plt.subplots(figsize=(side, side))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close("all")


def plot_taiwan_grid(
    images, predictions, ground_truths, filenames, methods, out_path
) -> None:
    """Render the cross-method Taiwan visualization grid to ``out_path``.

    Layout: rows = images (one per Taiwan photo), columns = methods. Each cell
    shows the image thumbnail with the method's prediction overlaid as the
    title. Title is colored green when ``predictions[i][j] == ground_truths[i]``
    and red otherwise. Leftmost column carries the filename as the row label.

    Writes a non-zero-byte PNG. Creates the parent directory if needed. Closes
    the figure after ``savefig`` (T-02-06). Handles the 1-row / 1-column
    degenerate cases via ``np.atleast_2d`` on the ``axes`` array.
    """
    _ensure_parent_dir(out_path)

    n_imgs = len(images)
    n_methods = len(methods)
    fig, axes = plt.subplots(
        n_imgs, n_methods, figsize=(3 * n_methods, 3 * n_imgs), squeeze=False
    )
    # squeeze=False already gives a 2-D array; np.atleast_2d kept as belt-and-suspenders
    # in case future matplotlib versions change the squeeze default.
    axes = np.atleast_2d(axes)

    for i in range(n_imgs):
        for j in range(n_methods):
            ax = axes[i, j]
            ax.imshow(images[i])
            ax.set_xticks([])
            ax.set_yticks([])

            pred = predictions[i][j]
            gt = ground_truths[i]

            pred_name = ""
            if isinstance(pred, (int, np.integer)) and int(pred) in GTSRB_CLASSES:
                pred_name = f" ({GTSRB_CLASSES[int(pred)]})"
            text = f"{methods[j]}\npred: {pred}{pred_name}\ngt: {gt}"
            color = "green" if pred == gt else "red"
            ax.set_title(text, color=color, fontsize=8)

        # Leftmost column carries the filename label for the row.
        axes[i, 0].set_ylabel(filenames[i], rotation=0, ha="right", labelpad=40)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def classification_report_text(y_true, y_pred) -> str:
    """Return sklearn's classification_report as a plain string.

    ``zero_division=0`` suppresses the noisy warning that fires on classes with
    zero support (common in the Taiwan eval where only a small subset of GTSRB
    classes is represented).
    """
    return classification_report(y_true, y_pred, zero_division=0)
