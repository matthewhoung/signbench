"""src.common — shared infrastructure modules.

Consumers import from submodules explicitly (e.g.
``from src.common.interfaces import ModelStrategy``). The package ``__init__``
deliberately does NOT re-export submodule names: this keeps module boundaries
explicit and avoids circular-import surprises when method modules (Phase 4+)
reference common contracts.
"""
