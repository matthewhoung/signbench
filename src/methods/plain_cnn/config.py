"""Plain CNN hyperparameters — the one fixed, empirically-verified config.

These two dicts are the entirety of the Plain CNN method's tunable surface.
They were not guessed: training on all 34799 GTSRB train images and testing
on all 12630 test images on this WSL2 machine's RTX 4060 GPU measured
**96.79-97.28% test accuracy across three seeds (42/123/7)** with exactly the
values below (05-RESEARCH.md, 2026-05-21).

This config is LOCKED, not a starting point. DEC-005 forbids hyperparameter
sweeps — do NOT add a second config variant, an LR schedule, or any tuning.

``dropout_rate`` 0.5 is a standard *regularization* layer (textbook vanilla-CNN
practice, predates and is independent of data augmentation). It is NOT data
augmentation and is therefore in-scope under DEC-005 — a plan/verifier check
must not mistake it for a DEC-005 violation.

The 30-epoch FIXED schedule with NO early stopping is deliberate: short-patience
early stopping was empirically observed to stop a seed at 95.65% (premature
val-accuracy plateau), uncomfortably close to the >=95% gate. A fixed 30-epoch
run lands every seed safely in the 96.8-97.3% band and is deterministic in
duration. Do NOT add an EarlyStopping config.
"""

# CNN architecture parameters — input shape, the three conv-block filter counts,
# kernel size, classifier-head width, dropout rate, and the GTSRB class count.
ARCH_PARAMS = {
    "input_shape": (32, 32, 3),
    "conv_filters": (32, 64, 128),   # the 3 conv blocks
    "kernel_size": 3,                # 3x3 convs, padding="same"
    "dense_units": 256,              # classifier-head hidden width
    "dropout_rate": 0.5,             # regularizer before the final layer (NOT augmentation)
    "num_classes": 43,               # GTSRB classes 0..42
}

# Training recipe parameters. epochs is a FIXED schedule — there is intentionally
# NO early-stopping key (see module docstring).
TRAIN_PARAMS = {
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "loss": "sparse_categorical_crossentropy",  # integer labels 0..42 — no one-hot
    "batch_size": 128,
    "epochs": 30,                    # FIXED schedule — NO early stopping
    "seed": 42,                      # mirrors Phase 4 random_state=42
}
