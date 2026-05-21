"""Spatial Transformer custom layer — modern TF 2.21 / Keras 3.14 port.

``STNLayer`` is a faithful port of the *math* in
``reference/GTSRB_Keras_STN/spatial_transformer.py`` (DEC-004 — read-only
reference; the architecture is mined, the TF 1.x / old-Keras API is rewritten).
It has the three classic STN components:

  1. a localization network (a small CNN regressing 6 affine parameters),
  2. a grid generator (``_meshgrid`` -> homogeneous coords -> affine matmul),
  3. a bilinear sampler (``_bilinear_sample`` via ``tf.gather_nd``).

Two non-negotiable correctness details, both empirically verified
(06-RESEARCH.md, 2026-05-21):

* **Identity init** — the localization net's final ``Dense(6)`` uses a zero
  kernel and a ``[1,0,0,0,1,0]`` bias, so at init the layer is a pass-through
  (verified max-abs-error ~3e-6 vs the input). Without it the STN destroys the
  image and training collapses (Pitfall 2).
* **Serialization fix** — the localization ``Sequential`` is created in
  ``__init__`` (NOT lazily in ``build``) AND explicitly built inside ``build``
  via ``self.loc.build(input_shape)``. Combined with the
  ``register_keras_serializable`` decorator, this lets ``keras.models.load_model``
  round-trip the layer (Pitfall 3). Without it ``load_model`` raises
  "Layer was never built".

No TF 1.x / old-Keras API anywhere: no ``keras.engine.topology``, no
``keras.layers.core``, no ``tf.Session``, no ``predict_classes``, no ``np_utils``.
"""

import tensorflow as tf
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="signbench")
class STNLayer(layers.Layer):
    """Spatial Transformer: localization net -> 6 affine params -> sampling grid
    -> bilinear sampler. One STN only (DEC-005). Identity-initialized so it
    starts as a pass-through."""

    def __init__(self, output_size=(32, 32), **kwargs):
        super().__init__(**kwargs)
        self.output_size = tuple(output_size)
        # Localization net built in __init__ (NOT lazily) — the serialization
        # landmine fix. Final Dense(6) is identity-initialized.
        self.loc = keras.Sequential(
            [
                layers.Conv2D(16, 7, padding="valid", activation="elu"),
                layers.MaxPooling2D(2),
                layers.Conv2D(32, 5, padding="valid", activation="elu"),
                layers.MaxPooling2D(2),
                layers.Conv2D(64, 3, padding="valid", activation="elu"),
                layers.MaxPooling2D(2),
                layers.Flatten(),
                layers.Dense(128, activation="elu"),
                layers.Dense(64, activation="elu"),
                layers.Dense(
                    6,
                    kernel_initializer="zeros",
                    bias_initializer=keras.initializers.Constant(
                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    ),
                ),
            ],
            name="locnet",
        )

    def build(self, input_shape):
        # CRITICAL: explicitly build the child so its weights exist before
        # serialization. Without this, load_model() raises "Layer was never built".
        self.loc.build(input_shape)
        super().build(input_shape)

    def _meshgrid(self, height, width):
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.linspace(-1.0, 1.0, height)
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        ones = tf.ones_like(xx)
        return tf.stack([xx, yy, ones], axis=0)  # (3, H*W)

    def _bilinear_sample(self, img, x, y):
        B = tf.shape(img)[0]
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]
        Hf = tf.cast(H, tf.float32)
        Wf = tf.cast(W, tf.float32)
        # normalized [-1,1] -> pixel [0, dim-1]
        x = 0.5 * (x + 1.0) * (Wf - 1.0)
        y = 0.5 * (y + 1.0) * (Hf - 1.0)
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1
        x0c = tf.clip_by_value(x0, 0, W - 1)
        x1c = tf.clip_by_value(x1, 0, W - 1)
        y0c = tf.clip_by_value(y0, 0, H - 1)
        y1c = tf.clip_by_value(y1, 0, H - 1)

        def gather(xc, yc):
            # gather_nd index order is (row, col) == (y, x); batch_dims=1
            return tf.gather_nd(img, tf.stack([yc, xc], axis=-1), batch_dims=1)

        Ia = gather(x0c, y0c)
        Ib = gather(x0c, y1c)
        Ic = gather(x1c, y0c)
        Id = gather(x1c, y1c)
        x0f = tf.cast(x0, tf.float32)
        x1f = tf.cast(x1, tf.float32)
        y0f = tf.cast(y0, tf.float32)
        y1f = tf.cast(y1, tf.float32)
        # bilinear weights from the UNCLIPPED x,y — keeps gradients clean at borders
        wa = tf.expand_dims((x1f - x) * (y1f - y), -1)
        wb = tf.expand_dims((x1f - x) * (y - y0f), -1)
        wc = tf.expand_dims((x - x0f) * (y1f - y), -1)
        wd = tf.expand_dims((x - x0f) * (y - y0f), -1)
        return wa * Ia + wb * Ib + wc * Ic + wd * Id  # (B, N, C)

    def call(self, inputs):
        theta = tf.reshape(self.loc(inputs), (-1, 2, 3))   # (B, 2, 3)
        B = tf.shape(inputs)[0]
        Ho, Wo = self.output_size
        grid = self._meshgrid(Ho, Wo)                       # (3, Ho*Wo)
        grid = tf.tile(tf.expand_dims(grid, 0), [B, 1, 1])  # (B, 3, Ho*Wo)
        src = tf.matmul(theta, grid)                        # (B, 2, Ho*Wo)
        xs = tf.reshape(src[:, 0, :], (B, Ho * Wo))
        ys = tf.reshape(src[:, 1, :], (B, Ho * Wo))
        sampled = self._bilinear_sample(inputs, xs, ys)     # (B, Ho*Wo, C)
        return tf.reshape(sampled, (B, Ho, Wo, tf.shape(inputs)[-1]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size[0], self.output_size[1], input_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"output_size": self.output_size})
        return cfg
