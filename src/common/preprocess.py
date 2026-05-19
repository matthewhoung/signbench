"""Shared preprocessing helpers (CON-preprocess).

``normalize(X)`` and ``resize_to_32(image)`` are identical for ALL methods.
Augmentation is OFF by default per DEC-005 — no augmentation function is exported
at module top level (the cleanest compliance: no entry point exists at all).
"""

import cv2
import numpy as np


def normalize(X: np.ndarray) -> np.ndarray:
    """Rescale pixel values to [0.0, 1.0] float32. Identical pipeline for ALL methods (DEC-001 cross-method comparability)."""
    return X.astype(np.float32) / 255.0


def resize_to_32(image: np.ndarray) -> np.ndarray:
    """Resize an arbitrary HxWx3 image to (32, 32, 3). Channel order is preserved."""
    return cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
