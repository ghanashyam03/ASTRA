from __future__ import annotations

from typing import Any

from astra.data.storage import TrajectoryStore
from astra.physics.kernel import PhysicsKernel

# Global state (module-level singletons, not thread-safe — single worker)
_kernel: PhysicsKernel | None = None
_store: TrajectoryStore | None = None
_jobs: dict[str, dict[str, Any]] = {}   # in-memory job tracking


def get_kernel() -> PhysicsKernel:
    global _kernel
    if _kernel is None:
        raise RuntimeError("Physics kernel not initialized")
    return _kernel


def get_store() -> TrajectoryStore:
    global _store
    if _store is None:
        raise RuntimeError("Storage not initialized")
    return _store
