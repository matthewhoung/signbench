"""RGB -> 10ch (1x1 conv) -> 3ch (1x1 conv) learned colour transform with VLReLU.

Per capstone_report.md "Learned color transformation" and conv_model.py:44-49
(reference repo — DEC-004 read-only; the math is ported, the API rewritten).

All stock Keras layers — no custom ``Layer`` subclass. ``keras.layers.LeakyReLU``
IS Very-Leaky-ReLU; using a stock layer means the colour transform serializes to
``.keras`` for free (no extra custom object to register).

VLReLU slope is 0.33 — the canonical value from Mishkin et al. 2016 (the paper
``capstone_report.md`` cites), and the value used in the empirically-verified
99.13% run. The reference *code* used ``LeakyReLU(alpha=0.5)``; success criterion
#1 says "per the original paper", hence 0.33.
"""

from keras import layers
from keras.regularizers import l2


def color_transform_layers(l2_reg=5e-4, vlrelu_slope=0.33):
    """Return the learned-colour-transform layer stack.

    RGB -> 1x1 conv 10ch -> VLReLU -> 1x1 conv 3ch -> VLReLU, with
    ``BatchNormalization`` interleaved. Stock layers only.

    NOTE — Keras 3.14 renamed the LeakyReLU argument ``alpha`` -> ``negative_slope``.
    Using the old ``alpha=`` would raise or be silently ignored (06-RESEARCH
    Pitfall 4).
    """
    return [
        layers.BatchNormalization(),
        layers.Conv2D(10, 1, padding="same", kernel_regularizer=l2(l2_reg)),
        layers.LeakyReLU(negative_slope=vlrelu_slope),   # VLReLU
        layers.BatchNormalization(),
        layers.Conv2D(3, 1, padding="same", kernel_regularizer=l2(l2_reg)),
        layers.LeakyReLU(negative_slope=vlrelu_slope),   # VLReLU
        layers.BatchNormalization(),
    ]
