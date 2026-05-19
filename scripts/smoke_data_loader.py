"""Phase-2 smoke test for the GTSRB data loader (ROADMAP success criterion #6).

Loads ``data/gtsrb/train.p`` and prints shape + class distribution. Tolerates
Phase-3 not having landed the data yet — prints a skip message and exits 0
instead of failing, so the Phase 2 gate can close before Phase 3 ships.

Run as::

    uv run python -m scripts.smoke_data_loader

ROADMAP success criterion #6 quotes verbatim "wired ready, not yet run on data".
That's exactly the state this script is designed to satisfy: it's runnable on
disk, exits 0 when data is absent (the current Phase-2 reality), and will
exercise the load + print path organically once Phase 3 lands the pickles.
"""

import os
import sys

import numpy as np

from src.common.data_loader import load_gtsrb


def main() -> int:
    """Run the smoke test. Returns 0 in both data-absent and data-present paths."""
    data_dir = "data/gtsrb"
    required = ("train.p", "valid.p", "test.p")
    missing = [name for name in required if not os.path.isfile(os.path.join(data_dir, name))]
    if missing:
        first_missing = missing[0]
        print(
            f"Phase 3 has not landed {data_dir}/{first_missing} yet — "
            "smoke test skipped (will be exercised once `make data` runs)."
        )
        return 0

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_gtsrb(data_dir)
    except FileNotFoundError as e:
        # Defensive: covers the edge case where data/gtsrb/ exists but a pickle
        # disappears between the pre-check above and the load call.
        print(f"Smoke test skipped: {e}")
        return 0

    print(f"train: X={X_train.shape}, y={y_train.shape}")
    print(f"valid: X={X_val.shape}, y={y_val.shape}")
    print(f"test:  X={X_test.shape}, y={y_test.shape}")

    ids, counts = np.unique(y_train, return_counts=True)
    print(
        f"train class distribution: {len(ids)} classes; "
        f"min count = {int(counts.min())}, "
        f"max count = {int(counts.max())}, "
        f"mean count = {counts.mean():.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
