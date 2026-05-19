"""GTSRB pickle loader (CON-data-loader).

Implements ``load_gtsrb(data_dir)`` per PROJECT_PLAN.md §4. The Udacity-distributed
GTSRB pickles (``train.p``, ``valid.p``, ``test.p``) were produced under Python 2
and must be opened with ``encoding='latin1'`` for compatibility. Phase 3 lands the
actual data files on disk; Phase 2 only validates the loader contract with
synthetic pickles.

This module is intentionally orthogonal to ``preprocess`` — the runner orchestrates
preprocessing so all methods share a single pipeline (DEC-001 cross-method
comparability).
"""

import pickle
from pathlib import Path

import numpy as np


def load_gtsrb(
    data_dir: str,
) -> "tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]":
    """Load the three GTSRB pickle files from ``data_dir``.

    Reads ``<data_dir>/train.p``, ``<data_dir>/valid.p``, ``<data_dir>/test.p``
    (note: the validation file is ``valid.p``, not ``val.p``, per the Udacity
    distribution and PROJECT_PLAN.md §3 folder layout) and returns the documented
    3-tuple-of-2-tuples shape:

        ((X_train, y_train), (X_val, y_val), (X_test, y_test))

    Each pickle is a Python-2-format dict with keys ``'features'`` (uint8
    ``(N, 32, 32, 3)`` ndarray) and ``'labels'`` (int ``(N,)`` ndarray). The
    Python-2 origin is why we open with ``encoding='latin1'``.

    Raises ``FileNotFoundError`` with the offending path in the message if any
    of the three pickle files is missing.
    """
    base = Path(data_dir)
    splits = (("train.p", "train"), ("valid.p", "valid"), ("test.p", "test"))
    payloads: dict[str, dict] = {}

    for filename, label in splits:
        path = base / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected GTSRB pickle not found: {path}. "
                f"Run `make data` (Phase 3) to provision GTSRB data."
            )
        with path.open("rb") as fp:
            payloads[label] = pickle.load(fp, encoding="latin1")

    def _split(name: str) -> "tuple[np.ndarray, np.ndarray]":
        data = payloads[name]
        X = np.asarray(data["features"])
        y = np.asarray(data["labels"])
        return X, y

    return _split("train"), _split("valid"), _split("test")
