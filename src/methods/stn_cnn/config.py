"""STN+CNN hero hyperparameters — the one fixed, empirically-verified config.

Measured 99.13% GTSRB test accuracy on this machine's RTX 4060 (06-RESEARCH.md,
2026-05-21) — clears the >=98% HERO GATE. LOCKED — DEC-005 forbids hyperparameter
sweeps and multiple STN layers.

NOTES on DEC-005 compliance:
- ReduceLROnPlateau is ONE fixed LR *schedule* (the modern replacement for the
  reference's Adam decay=0.01, which Keras 3 dropped) — NOT a sweep.
- l2_reg 5e-4 is the reference's 0.05 rescaled for normalize()'d [0,1] inputs
  (the reference applied l2 on raw [0,255] pixels) — a scale-correction of one
  fixed value, not tuning.
- Exactly ONE STN layer.
"""

ARCH_PARAMS = {
    "input_shape": (32, 32, 3),
    "stn_output_size": (32, 32),
    "feature_filters": (16, 32, 64, 96, 128, 192),  # reference feature extractor
    "l2_reg": 5e-4,                # rescaled from reference 0.05 for [0,1] inputs
    "vlrelu_slope": 0.33,          # Very-Leaky-ReLU slope (Mishkin et al. 2016)
    "dropout_rate": 0.6,           # reference value (conv_model.py:73)
    "num_classes": 43,
}

TRAIN_PARAMS = {
    "optimizer": "adam",
    "learning_rate": 1e-3,         # reference value (train_keras.py:33)
    "loss": "sparse_categorical_crossentropy",
    "batch_size": 128,             # reference value
    "epochs": 40,                  # FIXED — no EarlyStopping
    "seed": 42,                    # mirrors Phases 4-5
    # ReduceLROnPlateau is configured in model.py: factor=0.5, patience=5,
    # min_lr=1e-5, monitor="val_accuracy".
}
