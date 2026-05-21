"""STN+CNN hero ``ModelStrategy`` implementation.

``STNCNNModel`` is the project's hero method and core technical claim:

    Input(32,32,3)
      -> learned colour transform (1x1 conv 10ch -> VLReLU -> 1x1 conv 3ch -> VLReLU)
      -> STNLayer  (spatial transformer — localization net + grid + bilinear sampler)
      -> deep CNN classifier (the reference's 11-conv feature extractor + classifier)
      -> Dropout(0.6) -> Dense(43, softmax)

It is the third and final concrete ``ModelStrategy`` and mirrors
``PlainCNNModel`` exactly — same five-method ABC surface, same ``MODEL_EXT``,
same module-import GPU memory-growth loop, same ``predict``/``save``/``load``
shape. The only differences are a deeper ``_build_model``, a
``ReduceLROnPlateau`` callback, and ``epochs=40``.

The functional API (``keras.Model(inputs, outputs)``) is used — not
``Sequential`` — because the graph carries a custom layer and needs a reliable
``get_layer("stn")`` for the STN visualization (success criterion #7).

The empirically-verified config (``config.py``) measured 99.13% GTSRB test
accuracy on this machine's RTX 4060 (06-RESEARCH.md, 2026-05-21), clearing the
>=98% HERO GATE.
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.regularizers import l2

from src.common.interfaces import ModelStrategy
from src.methods.stn_cnn.stn_layer import STNLayer
from src.methods.stn_cnn.color_transform import color_transform_layers
from src.methods.stn_cnn.config import ARCH_PARAMS, TRAIN_PARAMS

# GPU memory growth — run once at module import, before any model is built.
# By default TF grabs ALL visible VRAM on first use; set_memory_growth makes TF
# allocate incrementally so it coexists with the desktop. Mirrors plain_cnn.
# The try/except RuntimeError guards the already-initialised case (harmless).
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # context already initialised — harmless


class STNCNNModel(ModelStrategy):
    """Hero model: input -> learned colour transform -> STN -> deep CNN
    classifier -> softmax(43). Implements the five-method ``ModelStrategy`` ABC.

    Expects images already normalized to ``[0,1]`` float32 ``(N,32,32,3)`` — the
    runner owns normalization for ALL splits. The reference's leading
    ``Lambda(x/127.5-1)`` is intentionally dropped (the runner's ``normalize()``
    replaces it; the leading ``BatchNormalization()`` then standardizes).
    """

    # File-extension declaration the runner reads to build ``models/stn_cnn.keras``.
    # Additive NON-ABSTRACT class attribute, NOT a sixth abstract method —
    # CON-modelstrategy-abc locks the five abstract methods, not class attributes
    # (Phase 4 Assumption A1). Consistent with plain_cnn's ".keras".
    MODEL_EXT = ".keras"

    def __init__(self):
        self.model = None                    # set by fit() or load()

    def name(self) -> str:
        """Identifier used in filenames and reports."""
        return "stn_cnn"

    def _build_model(self) -> keras.Model:
        """Build the STN+CNN as a functional ``keras.Model``.

        Wiring: Input -> learned colour transform -> STNLayer(name="stn") ->
        the reference's deep 11-conv classifier -> Dropout(0.6) ->
        Dense(43, softmax). The STN layer is NAMED ``"stn"`` so the
        visualization sub-model can resolve ``get_layer("stn")`` (criterion #7).

        The classifier depth is the reference's full feature extractor — depth
        is load-bearing (06-RESEARCH Pitfall 5: a shallow classifier overfits
        and tops out at 96.17%, missing the gate).
        """
        L2 = ARCH_PARAMS["l2_reg"]                            # 5e-4
        inp = keras.Input(shape=ARCH_PARAMS["input_shape"])   # (32,32,3)
        x = inp
        # learned colour transform (stock layers)
        for layer in color_transform_layers(
            l2_reg=L2, vlrelu_slope=ARCH_PARAMS["vlrelu_slope"]
        ):
            x = layer(x)
        # spatial transformer (custom layer) — NAMED "stn" for the visualization
        x = STNLayer(output_size=ARCH_PARAMS["stn_output_size"], name="stn")(x)

        # deep CNN classifier — the reference feature extractor + classifier
        def conv(f):
            return layers.Conv2D(
                f, 5, padding="same", activation="relu",
                kernel_regularizer=l2(L2),
            )

        for f in ARCH_PARAMS["feature_filters"]:              # 16,32,64,96,128,192
            x = conv(f)(x)
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)                         # -> 16x16
        x = conv(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)                         # -> 8x8
        x = conv(128)(x)
        x = layers.BatchNormalization()(x)
        x = conv(64)(x)
        x = layers.MaxPooling2D(8)(x)                         # -> 1x1
        x = layers.Flatten()(x)
        x = layers.Dropout(ARCH_PARAMS["dropout_rate"])(x)    # 0.6
        out = layers.Dense(
            ARCH_PARAMS["num_classes"], activation="softmax"  # 43
        )(x)
        return keras.Model(inp, out, name="stn_cnn")

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        """Train for a FIXED 40-epoch schedule; return the epoch-keyed history.

        Returns ``dict(history.history)`` — ``{accuracy, loss, val_accuracy,
        val_loss, learning_rate}``, each a list of 40 floats. The runner routes
        any list-valued history through ``plot_training_curves``'s epoch-keyed
        branch; ``plot_training_curves`` reads only the four acc/loss keys and
        ignores ``learning_rate`` (Phase 5 fix — zero runner change needed).

        A ``ReduceLROnPlateau`` callback is the modern Keras 3 replacement for
        the reference's Adam ``decay=0.01`` — ONE fixed LR schedule, not a
        sweep (DEC-005). No EarlyStopping (consistent with Phase 5).
        """
        keras.utils.set_random_seed(TRAIN_PARAMS["seed"])     # 42
        self.model = self._build_model()

        # GPU-visibility check — print once at training start. The runner cannot
        # FIX a missing GPU (the linker path is locked once `import tensorflow`
        # runs), but it can detect and warn loudly so a silent CPU fallback —
        # the STN+CNN on CPU is impractically slow — does not pass unnoticed.
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[stn_cnn] training on GPU: {gpus[0].name}")
        else:
            print(
                "[stn_cnn] WARNING: no GPU visible — training on CPU "
                "(the STN+CNN on CPU is VERY slow). Check LD_LIBRARY_PATH "
                "includes the nvidia/*/lib dirs (the Makefile sets this; "
                "running the runner directly may not)."
            )

        self.model.compile(
            optimizer=keras.optimizers.Adam(TRAIN_PARAMS["learning_rate"]),  # 1e-3
            loss="sparse_categorical_crossentropy",     # integer labels 0..42
            metrics=["accuracy"],
        )
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy", factor=0.5, patience=5,
                min_lr=1e-5, verbose=1,
            )
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TRAIN_PARAMS["epochs"],            # 40, FIXED — no early stopping
            batch_size=TRAIN_PARAMS["batch_size"],    # 128
            callbacks=callbacks,
            verbose=2,
        )
        return dict(history.history)

    def predict(self, X) -> "tuple[np.ndarray, np.ndarray]":
        """Return ``(class_ids, confidences)`` — both 1-D arrays of length N.

        A softmax column index IS the GTSRB class id (classes 0..42 contiguous,
        43 softmax units in that order). Identical to ``PlainCNNModel.predict``.
        """
        proba = self.model.predict(X, batch_size=256, verbose=0)   # (N, 43) softmax
        predictions = np.argmax(proba, axis=1)                     # class ids 0..42
        confidences = np.max(proba, axis=1)                        # max softmax prob
        return predictions, confidences

    def save(self, path: str) -> None:
        """Serialize to the Keras 3 native ``.keras`` format.

        The ``.keras`` extension selects the native format automatically. The
        custom ``STNLayer`` is captured because it is decorated with
        ``register_keras_serializable`` and explicitly builds its child loc-net.
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """Restore a trained model from a ``.keras`` file (mutates self).

        No ``custom_objects=`` argument is needed: ``STNLayer`` is decorated with
        ``@keras.saving.register_keras_serializable(package="signbench")``, so
        ``load_model`` resolves the custom class globally (06-RESEARCH Pitfall 3).
        """
        self.model = keras.models.load_model(path)
