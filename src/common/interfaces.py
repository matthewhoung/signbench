"""ModelStrategy ABC (CON-modelstrategy-abc) — the uniform method contract every src/methods/* module implements.

The five-method surface (``name``, ``fit``, ``predict``, ``save``, ``load``) is the
locked contract from PROJECT_PLAN.md §4 and intel/constraints.md (CON-modelstrategy-abc).
Runners depend only on this surface; methods are interchangeable behind it.

The return type of ``predict`` is annotated as a forward-reference string
``"tuple[np.ndarray, np.ndarray]"`` to avoid forcing ``numpy`` evaluation at class-body
construction time while still keeping the annotation legible to tooling.
"""

from abc import ABC, abstractmethod

import numpy as np  # noqa: F401  (imported for the forward-reference annotation)


class ModelStrategy(ABC):
    """Uniform contract every method module under ``src/methods/`` implements.

    Concrete subclasses MUST override all five abstract methods below;
    instantiation otherwise raises ``TypeError``.
    """

    @abstractmethod
    def name(self) -> str:
        """Identifier used in filenames and reports (e.g. 'rf_hog')."""
        ...

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        """Train and return training-curve dict (epoch-keyed for CNNs, single-point for RF)."""
        ...

    @abstractmethod
    def predict(self, X) -> "tuple[np.ndarray, np.ndarray]":
        """Return (class_ids, confidences); both are 1-D numpy arrays of length N."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize the trained model to disk at ``path``."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore a trained model from disk at ``path`` (mutates self)."""
        ...
