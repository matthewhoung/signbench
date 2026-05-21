"""RF + HOG hyperparameters — the one empirically-verified, LOCKED config.

These two dicts are the entirety of the RF+HOG method's tunable surface.
They were not guessed: training on all 34799 GTSRB train images and testing
on all 12630 test images on this WSL2 CPU machine measured **90.31% test
accuracy** with exactly the values below (04-RESEARCH.md, 2026-05-21).

This config is LOCKED, not a starting point. DEC-005 forbids hyperparameter
sweeps. Five plausible "improvements" were empirically tested and all failed
or did not help (more trees, more orientations, CLAHE preproc, color-histogram
concat, ``class_weight='balanced'``) — do NOT add any of them.

HOG is applied to a GRAYSCALE image (``skimage.color.rgb2gray`` output) with
NO ``channel_axis`` argument. The resulting feature vector is 1764 floats per
32x32 image. ``random_state=42`` makes the 90.31% deterministic and exactly
reproducible on every run.
"""

# HOG descriptor parameters. orientations / pixels_per_cell / cells_per_block
# are the configurable "HOG cell/block/orientations" knobs (ROADMAP criterion 1)
# — named, editable module constants.
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (4, 4),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
}

# RandomForestClassifier parameters. n_estimators / max_depth are the
# configurable "RF n_estimators/max_depth" knobs (ROADMAP criterion 1).
# class_weight is intentionally ABSENT (= None): research verified
# class_weight='balanced' scored WORSE (90.02% vs 90.31%).
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,        # fully-grown trees; capping to 20 drops below the gate
    "max_features": "sqrt",   # 'log2' and 0.1 both scored worse
    "n_jobs": -1,             # use all CPU cores
    "random_state": 42,       # determinism — the 90.31% reproduces exactly
}
