"""RF + HOG ``ModelStrategy`` implementation.

``RFHOGModel`` extracts grayscale HOG features from 32x32 GTSRB images and
classifies them with a ``RandomForestClassifier``. It is the first concrete
``ModelStrategy`` and validates the whole train -> save -> eval -> report
pipeline before any deep-learning effort.

The verified config (``config.py``) measured 90.31% GTSRB test accuracy.
HOG extraction lives in the SHARED private helper ``_extract_hog`` so the exact
same transform is guaranteed at both ``fit`` and ``predict`` time — train and
inference cannot drift (04-RESEARCH.md Pitfall 2).
"""

import joblib
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.common.interfaces import ModelStrategy
from src.methods.rf_hog.config import HOG_PARAMS, RF_PARAMS


class RFHOGModel(ModelStrategy):
    """RandomForest-on-HOG classifier for 43-class GTSRB traffic signs.

    Implements the five-method ``ModelStrategy`` ABC surface. Expects images
    already normalized to ``[0,1]`` float32 ``(N,32,32,3)`` — the runner owns
    normalization for ALL splits.
    """

    # File-extension declaration the runner reads to build ``models/<name>.pkl``.
    # This is an additive NON-ABSTRACT class attribute, NOT a sixth abstract
    # method — CON-modelstrategy-abc locks the five abstract methods, not class
    # attributes (04-RESEARCH.md Assumption A1). CNN methods will set ".keras".
    MODEL_EXT = ".pkl"

    def __init__(self):
        self.rf = None                       # set by fit() or load()
        self.hog_params = dict(HOG_PARAMS)

    def name(self) -> str:
        """Identifier used in filenames and reports."""
        return "rf_hog"

    def _extract_hog(self, X) -> np.ndarray:
        """Extract HOG features for a batch of images.

        X: ``(N,32,32,3)`` float32 in ``[0,1]`` (already normalized by the
        runner). Returns an ``(N,1764)`` float32 array. SHARED by ``fit`` and
        ``predict`` so the transform is identical at train and inference time.
        """
        feats = []
        for img in tqdm(X, desc="HOG"):
            gray = rgb2gray(img)             # (32,32) float
            feats.append(hog(gray, feature_vector=True, **self.hog_params))
        return np.asarray(feats, dtype=np.float32)   # (N,1764)

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        """Train the RF on HOG features; return a single-point history dict."""
        F_train = self._extract_hog(X_train)
        self.rf = RandomForestClassifier(**RF_PARAMS)
        self.rf.fit(F_train, y_train)
        train_acc = accuracy_score(y_train, self.rf.predict(F_train))
        F_val = self._extract_hog(X_val)
        val_acc = accuracy_score(y_val, self.rf.predict(F_val))
        return {
            # train_acc / val_acc are scalars in [0,1] — safe for
            # plot_training_curves' fixed [0,1] y-axis.
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            # n_features / n_estimators are kept for the Phase 8 report; the
            # runner must NOT pass them to plot_training_curves (Pitfall 4).
            "n_features": int(F_train.shape[1]),
            "n_estimators": int(RF_PARAMS["n_estimators"]),
        }

    def predict(self, X) -> "tuple[np.ndarray, np.ndarray]":
        """Return ``(class_ids, confidences)`` — both 1-D arrays of length N."""
        F = self._extract_hog(X)
        proba = self.rf.predict_proba(F)             # (N, 43)
        idx = np.argmax(proba, axis=1)
        # Map proba column index -> class id via classes_, NOT bare argmax.
        # GTSRB's contiguous 0..42 classes make argmax coincidentally correct,
        # but classes_ indexing is the robust pattern and costs nothing.
        predictions = self.rf.classes_[idx]
        confidences = np.max(proba, axis=1)          # max-proba == confidence, [0,1]
        return predictions, confidences

    def save(self, path: str) -> None:
        """Serialize the trained RF + HOG params to ``path``.

        ``compress=3`` is ESSENTIAL — an uncompressed 500-tree forest on
        1764-dim features is 1.9 GB; compressed it is ~129 MB with no accuracy
        change (Pitfall 5).
        """
        joblib.dump(
            {"rf": self.rf, "hog_params": self.hog_params},
            path,
            compress=3,
        )

    def load(self, path: str) -> None:
        """Restore a trained model from ``path`` (mutates self)."""
        bundle = joblib.load(path)
        self.rf = bundle["rf"]
        self.hog_params = bundle["hog_params"]
