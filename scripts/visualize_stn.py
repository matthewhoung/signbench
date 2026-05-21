"""STN spatial-rectification visualization — ROADMAP Phase 6 success criterion #7.

Loads the trained ``models/stn_cnn.keras``, builds a sub-model that exposes the
named ``"stn"`` layer's intermediate output, runs a handful of GTSRB test images
through it, and renders a 2-row matplotlib grid (top = input, bottom = STN
output) to ``analysis/stn_cnn/stn_visualization.png``.

The sub-model trick — ``keras.Model(inputs=model.input,
outputs=model.get_layer("stn").output)`` — shares the trained weights and
exposes the spatial transformer's rectified image tensor. It works on a model
reloaded from ``.keras`` because ``STNLayer`` is named ``"stn"`` in the
functional graph and registered via ``register_keras_serializable``.

Run as::

    uv run python -m scripts.visualize_stn

(``scripts/__init__.py`` already exists from Phase 2.) DEC-005 forbids new
Makefile targets — this is a documented manual post-``make train-stn`` step,
not a wired target.
"""

import os

import numpy as np
import keras
import matplotlib
matplotlib.use("Agg")                        # headless (WSL2) — like evaluate.py
import matplotlib.pyplot as plt

from src.common.data_loader import load_gtsrb
from src.common.preprocess import normalize
# Importing the model module registers STNLayer via register_keras_serializable
# so load_model can resolve the custom layer.
import src.methods.stn_cnn.model  # noqa: F401


def main():
    """Render the STN before/after grid to analysis/stn_cnn/stn_visualization.png."""
    model = keras.models.load_model("models/stn_cnn.keras")
    viz = keras.Model(inputs=model.input, outputs=model.get_layer("stn").output)

    _, _, (Xte, _) = load_gtsrb("data/gtsrb")
    sample = normalize(Xte[:8])
    rectified = viz.predict(sample, verbose=0)
    # The STN output may slightly exceed [0,1] (bilinear sampling round-off);
    # clip for clean imshow rendering.
    rectified = np.clip(rectified, 0.0, 1.0)

    n = len(sample)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5))
    for i in range(n):
        axes[0, i].imshow(sample[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(rectified[i])
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("input", rotation=0, ha="right")
    axes[1, 0].set_ylabel("STN out", rotation=0, ha="right")
    fig.suptitle("STN spatial rectification — input (top) vs STN output (bottom)")

    # analysis/ is gitignored — never exists on a fresh checkout (Pitfall 10).
    os.makedirs("analysis/stn_cnn", exist_ok=True)
    plt.tight_layout()
    plt.savefig("analysis/stn_cnn/stn_visualization.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote analysis/stn_cnn/stn_visualization.png")


if __name__ == "__main__":
    main()
