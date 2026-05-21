"""Method registry (CON-method-registry).

``get_method(name)`` maps a method name to a fresh ``ModelStrategy`` instance.
Runners call this so they never hardcode concrete method classes.

Imports are LAZY (per-name, inside ``get_method``) — NOT a module-level
``{name: Class}`` dict. A top-level dict would force
``from src.methods.plain_cnn.model import PlainCNNModel`` at registry import
time, but ``plain_cnn``/``stn_cnn`` are still empty stubs (Phases 5-6) — that
``ImportError`` would break ``train.py``/``eval_gtsrb.py`` for ``rf_hog`` too.
Lazy imports also avoid pulling in TensorFlow when running the CPU-only RF
method.
"""

from src.common.interfaces import ModelStrategy


def get_method(name: str) -> ModelStrategy:
    """Return a fresh ``ModelStrategy`` instance for ``name``.

    Raises ``ValueError`` for an unknown method name.
    """
    if name == "rf_hog":
        from src.methods.rf_hog.model import RFHOGModel
        return RFHOGModel()

    if name == "plain_cnn":
        # Lazy import — pulling in plain_cnn.model triggers the heavy
        # TensorFlow import, which must happen ONLY when plain_cnn is
        # actually requested (so rf_hog runs stay TF-free and fast).
        from src.methods.plain_cnn.model import PlainCNNModel
        return PlainCNNModel()

    # The stn_cnn branch is intentionally still ABSENT — src/methods/stn_cnn/
    # model.py is a bare stub; importing it would raise. It arrives in Phase 6.

    raise ValueError(
        f"Unknown method '{name}'. Available: rf_hog, plain_cnn "
        f"(stn_cnn arrives in Phase 6)."
    )
