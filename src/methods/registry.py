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

    # Phases 5-6 extend the ladder here:
    # if name == "plain_cnn":
    #     from src.methods.plain_cnn.model import PlainCNNModel
    #     return PlainCNNModel()
    # if name == "stn_cnn":
    #     from src.methods.stn_cnn.model import STNCNNModel
    #     return STNCNNModel()

    raise ValueError(
        f"Unknown method '{name}'. Available: rf_hog "
        f"(plain_cnn, stn_cnn arrive in Phases 5-6)."
    )
