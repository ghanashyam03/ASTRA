"""ASTRA FastAPI application. Zero physics logic in this file."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from astra.api import dependencies as deps
from astra.api.middleware.logging import RequestLoggingMiddleware
from astra.api.routes.health import router as health_router
from astra.api.routes.missions import router as missions_router
from astra.api.routes.physics import router as physics_router
from astra.api.routes.trajectories import router as trajectories_router
from astra.data.storage import TrajectoryStore
from astra.physics.kernel import PhysicsKernel

logger = logging.getLogger(__name__)


# Re-expose dependencies for backward compatibility (e.g. imports from astra.api.app)
get_kernel = deps.get_kernel
get_store = deps.get_store
_jobs = deps._jobs


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting ASTRA — loading SPICE kernels...")
    try:
        deps._kernel = PhysicsKernel().load()
        logger.info("SPICE kernels loaded.")
    except Exception as e:
        logger.warning(f"SPICE kernels not loaded: {e}. Physics endpoints will be unavailable.")
        deps._kernel = PhysicsKernel()  # no kernels — degraded mode

    db_path = getattr(app.state, "db_path", "data/astra.duckdb")
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    deps._store = TrajectoryStore(db_path)

    logger.info(f"ASTRA ready with database {db_path}.")
    yield
    if deps._store:
        deps._store.close()
    logger.info("ASTRA shutdown complete.")


app = FastAPI(
    title="ASTRA",
    description="Autonomous Space Trajectory Reasoning Architecture",
    version="0.1.0",
    lifespan=lifespan,
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

app.include_router(health_router)
app.include_router(missions_router)
app.include_router(trajectories_router)
app.include_router(physics_router)
