"""Taiwan traffic-sign loader (CON-taiwan-loader).

Implements ``load_taiwan(data_dir)`` per PROJECT_PLAN.md §4. Reads the
user-authored ``labels.csv``, loads each referenced photo, and applies the SAME
spatial preprocessing as the GTSRB training pipeline so the trained models see
inputs in a consistent shape.

Mirrors ``data_loader.load_gtsrb`` in its division of labour: this module
returns **uint8** ``(N, 32, 32, 3)`` images — pixel-value normalization is the
runner's job (``eval_taiwan`` calls ``preprocess.normalize``), keeping a single
shared normalization pipeline across all methods (DEC-001 cross-method
comparability).

Unlike GTSRB pickles (which arrive pre-shaped 32×32×3), Taiwan photos arrive at
arbitrary input sizes (~800–1500 px), so each image is passed through
``preprocess.resize_to_32``. OpenCV reads BGR; the channel order is converted to
RGB so it matches the GTSRB pipeline the models were trained on.

The ``OOD`` literal in ``labels.csv`` (Taiwan-unique signs with no GTSRB class)
maps to the integer sentinel ``-1`` in ``y``. No model can ever predict ``-1``,
so OOD rows are intentionally never ``correct`` downstream — the domain-shift
story is *what* the models predict for those signs and *how confidently*.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.common.preprocess import resize_to_32

# Integer sentinel for out-of-distribution (Taiwan-unique) signs that have no
# GTSRB class id. Chosen as -1 because GTSRB ids span 0..42.
OOD_SENTINEL = -1


def load_taiwan(data_dir: str) -> "tuple[np.ndarray, np.ndarray, list[str]]":
    """Load the Taiwan traffic-sign dataset from ``data_dir``.

    Reads ``<data_dir>/labels.csv`` (columns ``filename``,
    ``gtsrb_class_id_or_OOD``, ``notes`` per CON-taiwan-labels-schema), loads
    each image from ``<data_dir>/images/<filename>``, converts BGR→RGB, and
    resizes to 32×32×3 via ``preprocess.resize_to_32``.

    Returns the documented 3-tuple:

        (X, y, filenames)

    - ``X``  — uint8 ``(N, 32, 32, 3)`` ndarray, NOT normalized (the runner
      calls ``preprocess.normalize``, mirroring ``load_gtsrb`` + ``eval_gtsrb``).
    - ``y``  — int ``(N,)`` ndarray; an integer class id for universal signs,
      the sentinel ``-1`` for ``OOD`` rows.
    - ``filenames`` — ``list[str]`` of the image filenames in original
      labels.csv row order, parallel to ``X`` and ``y``.

    Row order from ``labels.csv`` is preserved across all three return values.

    Raises ``FileNotFoundError`` with the offending path in the message if
    ``labels.csv`` is missing, or if a referenced image file does not exist or
    cannot be decoded.
    """
    base = Path(data_dir)
    labels_path = base / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Expected Taiwan labels file not found: {labels_path}. "
            f"The user must create data/taiwan/labels.csv with columns "
            f"filename,gtsrb_class_id_or_OOD,notes (CON-taiwan-labels-schema)."
        )

    df = pd.read_csv(labels_path)

    images: list[np.ndarray] = []
    labels: list[int] = []
    filenames: list[str] = []

    # Iterate rows IN FILE ORDER so X / y / filenames stay parallel.
    for _, row in df.iterrows():
        filename = str(row["filename"]).strip()
        image_path = base / "images" / filename
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Taiwan image referenced by labels.csv not found: {image_path}."
            )

        # cv2.imread returns None on an unreadable/corrupt file (T-07-03).
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(
                f"Taiwan image could not be decoded by OpenCV: {image_path}. "
                f"File may be corrupt or in an unsupported format."
            )

        # BGR→RGB so channel order matches the GTSRB training pipeline.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Taiwan photos arrive at arbitrary size — bring them to 32×32×3.
        resized = resize_to_32(rgb)
        images.append(resized.astype(np.uint8))

        # Map gtsrb_class_id_or_OOD: the literal "OOD" -> -1, else int.
        raw = str(row["gtsrb_class_id_or_OOD"]).strip()
        if raw.upper() == "OOD":
            labels.append(OOD_SENTINEL)
        else:
            labels.append(int(raw))

        filenames.append(filename)

    X = np.stack(images).astype(np.uint8)
    y = np.array(labels, dtype=int)
    return X, y, filenames
