"""ASTRA FastAPI application. Zero physics logic in this file."""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from astra.api.middleware.logging import RequestLoggingMiddleware
from astra.api.routes import health, missions, physics, trajectories
from astra.data.storage import TrajectoryStore
from astra.physics.kernel import PhysicsKernel

logger = logging.getLogger(__name__)

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _kernel, _store
    logger.info("Starting ASTRA — loading SPICE kernels...")
    try:
        _kernel = PhysicsKernel().load()
        logger.info("SPICE kernels loaded.")
    except Exception as e:
        logger.warning(f"SPICE kernels not loaded: {e}. Physics endpoints will be unavailable.")
        _kernel = PhysicsKernel()  # no kernels — degraded mode
    
    db_path = getattr(app.state, "db_path", "data/astra.duckdb")
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _store = TrajectoryStore(db_path)
    
    logger.info(f"ASTRA ready with database {db_path}.")
    yield
    if _store:
        _store.close()
    logger.info("ASTRA shutdown complete.")


app = FastAPI(
    title="ASTRA",
    description="Autonomous Space Trajectory Reasoning Architecture",
    version="0.1.0",
    lifespan=lifespan,
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

app.include_router(health.router)
app.include_router(missions.router)
app.include_router(trajectories.router)
app.include_router(physics.router)
