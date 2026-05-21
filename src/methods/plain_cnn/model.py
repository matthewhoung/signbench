"""Plain CNN ``ModelStrategy`` implementation.

``PlainCNNModel`` is a vanilla 3-conv-block Keras 3 CNN
(Conv -> BatchNormalization -> ReLU -> MaxPooling, three times -> Flatten ->
Dense(256, ReLU) -> Dropout(0.5) -> Dense(43, softmax)). It is the second
concrete ``ModelStrategy`` and an **ablation baseline**: its role is to cleanly
beat the RF baseline (90.32%) while *underperforming* the hero STN+CNN, so
STN's contribution stays quantifiable.

By design it contains **no STN, no learned colour transform, no 1x1 conv, and
no custom ``Layer`` subclass** — it is a plain ``keras.Sequential`` of stock
layers only (success criterion #2, verifiable by inspection). The spatial
transformer and the learned colour module belong to Phase 6's ``stn_cnn``.

The verified config (``config.py``) measured 96.79-97.28% GTSRB test accuracy
across three seeds on this machine's RTX 4060 (05-RESEARCH.md).
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from src.common.interfaces import ModelStrategy
from src.methods.plain_cnn.config import ARCH_PARAMS, TRAIN_PARAMS

# GPU memory growth — run once at module import, before any model is built.
# By default TF grabs ALL visible VRAM on first use; the desktop already uses
# ~2.5 GB of the 8 GB. set_memory_growth makes TF allocate incrementally so it
# coexists with the desktop. Must be called before any GPU memory is allocated;
# the try/except RuntimeError guards the already-initialised case (harmless).
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # context already initialised — harmless


class PlainCNNModel(ModelStrategy):
    """Vanilla 3-conv-block CNN for 43-class GTSRB traffic-sign classification.

    Implements the five-method ``ModelStrategy`` ABC surface. Expects images
    already normalized to ``[0,1]`` float32 ``(N,32,32,3)`` — the runner owns
    normalization for ALL splits. No STN, no learned colour transform — an
    ablation baseline whose role is to underperform the hero STN+CNN.
    """

    # File-extension declaration the runner reads to build ``models/plain_cnn.keras``.
    # Additive NON-ABSTRACT class attribute, NOT a sixth abstract method —
    # CON-modelstrategy-abc locks the five abstract methods, not class
    # attributes (Phase 4 Assumption A1). RFHOGModel sets ".pkl".
    MODEL_EXT = ".keras"

    def __init__(self):
        self.model = None                    # set by fit() or load()

    def name(self) -> str:
        """Identifier used in filenames and reports."""
        return "plain_cnn"

    def _build_model(self) -> keras.Sequential:
        """Build the 3-conv-block CNN as a plain ``keras.Sequential``.

        Each conv block is exactly Conv2D -> BatchNormalization ->
        Activation("relu") -> MaxPooling2D, in that order — BatchNorm sits
        BETWEEN the conv and the activation (the canonical Ioffe-Szegedy
        ordering). The activation is a standalone ``layers.Activation`` layer
        (NOT a ``Conv2D(activation=...)`` shortcut) so the Conv -> BN -> ReLU
        order is literally visible in the layer list (ablation criterion #2).
        Only stock Keras layers — no custom Layer subclass, no 1x1 conv, no
        STN / spatial-transformer / colour-transform layer.
        """
        f1, f2, f3 = ARCH_PARAMS["conv_filters"]      # (32, 64, 128)
        k = ARCH_PARAMS["kernel_size"]                # 3
        model = keras.Sequential([
            keras.Input(shape=ARCH_PARAMS["input_shape"]),    # (32,32,3)
            # --- Conv block 1 ---
            layers.Conv2D(f1, k, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=2),
            # --- Conv block 2 ---
            layers.Conv2D(f2, k, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=2),
            # --- Conv block 3 ---
            layers.Conv2D(f3, k, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=2),
            # --- Classifier head ---
            layers.Flatten(),
            layers.Dense(ARCH_PARAMS["dense_units"], activation="relu"),     # 256
            layers.Dropout(ARCH_PARAMS["dropout_rate"]),                     # 0.5
            layers.Dense(ARCH_PARAMS["num_classes"], activation="softmax"),  # 43
        ])
        return model

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        """Train for a FIXED 30-epoch schedule; return the epoch-keyed history.

        Returns ``dict(history.history)`` — the Keras ``History.history`` with
        the four epoch-keyed lists ``accuracy`` / ``loss`` / ``val_accuracy`` /
        ``val_loss`` — consumed directly by ``evaluate.plot_training_curves``'s
        epoch-keyed branch. NO EarlyStopping callback (see config.py docstring).
        """
        keras.utils.set_random_seed(TRAIN_PARAMS["seed"])     # 42
        self.model = self._build_model()

        # GPU-visibility check — print once at training start. The runner cannot
        # FIX a missing GPU (the linker path is locked once `import tensorflow`
        # runs), but it can detect and warn loudly so a silent CPU fallback does
        # not pass unnoticed.
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[plain_cnn] training on GPU: {gpus[0].name}")
        else:
            print(
                "[plain_cnn] WARNING: no GPU visible — training on CPU "
                "(much slower). Check LD_LIBRARY_PATH includes the "
                "nvidia/*/lib dirs (the Makefile sets this; running the "
                "runner directly may not)."
            )

        self.model.compile(
            optimizer=keras.optimizers.Adam(TRAIN_PARAMS["learning_rate"]),  # 1e-3
            loss="sparse_categorical_crossentropy",     # integer labels 0..42
            metrics=["accuracy"],
        )
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TRAIN_PARAMS["epochs"],            # 30, FIXED — no early stopping
            batch_size=TRAIN_PARAMS["batch_size"],    # 128
            verbose=2,
        )
        # History.history == {accuracy, loss, val_accuracy, val_loss}, each a
        # list of 30 floats. dict(...) returns a plain dict, not the History
        # object's attribute.
        return dict(history.history)

    def predict(self, X) -> "tuple[np.ndarray, np.ndarray]":
        """Return ``(class_ids, confidences)`` — both 1-D arrays of length N.

        Unlike RFHOGModel there is no ``classes_`` remap: a softmax column
        index IS the GTSRB class id (classes 0..42 contiguous, 43 softmax
        units in that order). The CNN consumes the normalized ``(N,32,32,3)``
        float array directly — no HOG, no feature extraction.
        """
        proba = self.model.predict(X, batch_size=256, verbose=0)   # (N, 43) softmax
        predictions = np.argmax(proba, axis=1)                     # class ids 0..42
        confidences = np.max(proba, axis=1)                        # max softmax prob, [0,1]
        return predictions, confidences

    def save(self, path: str) -> None:
        """Serialize to the Keras 3 native ``.keras`` format.

        The ``.keras`` extension selects the native format automatically — no
        ``save_format=`` argument, no ``.h5`` (Keras 3, 05-RESEARCH Pitfall 7).
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """Restore a trained model from a ``.keras`` file (mutates self)."""
        self.model = keras.models.load_model(path)
