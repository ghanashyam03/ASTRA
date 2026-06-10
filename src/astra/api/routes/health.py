"""
Health routes for the ASTRA API.

Contains:
- GET /v1/health: Check application health status and SPICE loading.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from astra.api.dependencies import get_kernel
from astra.api.schemas.responses import HealthResponse
from astra.physics.kernel import PhysicsKernel

router = APIRouter()


@router.get("/v1/health", response_model=HealthResponse)
async def health(kernel: PhysicsKernel = Depends(get_kernel)) -> HealthResponse:
    """Check application health status and SPICE loading."""
    return HealthResponse(
        status="ok",
        service="ASTRA",
        version="0.1.0",
        spice_loaded=kernel is not None and kernel._kernels_loaded,
    )


@router.get("/v1/metrics")
async def metrics() -> dict[str, Any]:
    """Return runtime metrics: uptime, job counts, cache hit rate, storage stats."""
    import time
    from typing import Any
    from astra.api.app import get_kernel, get_store, _jobs
    
    result: dict[str, Any] = {
        "service": "ASTRA",
        "version": "0.1.0",
    }
    
    try:
        kernel = get_kernel()
        if kernel._kernels_loaded and kernel.ephemeris.cache is not None:
            result["cache"] = kernel.ephemeris.cache.stats.to_dict()
            result["cache"]["n_entries"] = len(kernel.ephemeris.cache)
        else:
            result["cache"] = {"status": "not_loaded"}
    except Exception:
        result["cache"] = {"status": "unavailable"}
    
    try:
        store = get_store()
        traj_count = store.conn.execute(
            "SELECT COUNT(*) FROM trajectories"
        ).fetchone()[0]
        run_count = store.conn.execute(
            "SELECT COUNT(*) FROM optimization_runs"
        ).fetchone()[0]
        result["storage"] = {
            "trajectories": traj_count,
            "optimization_runs": run_count,
        }
    except Exception:
        result["storage"] = {"status": "unavailable"}
    
    try:
        result["active_jobs"] = len([
            v for v in _jobs.values()
            if v.get("status") in ("queued", "running")
        ])
    except Exception:
        pass
    
    return result
