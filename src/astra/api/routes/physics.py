"""
Physics routes for the ASTRA API.

Contains:
- GET /v1/bodies/{body_name}/state: Get state vectors (position/velocity) of a celestial body.
- POST /v1/windows/porkchop: Generate transfer delta-v scan (porkchop plot) data.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from astra.api.dependencies import get_kernel
from astra.api.schemas.requests import PorkchopRequest
from astra.api.schemas.responses import BodyStateResponse

router = APIRouter(tags=["physics"])


@router.get("/v1/bodies/{body_name}/state", response_model=BodyStateResponse)
async def get_body_state(
    body_name: str, epoch_j2000: float = 0.0
) -> BodyStateResponse:
    """Retrieve position, velocity, and orbital parameters of a celestial body at a
    specific epoch.
    """
    kernel = get_kernel()
    if not kernel._kernels_loaded:
        raise HTTPException(status_code=503, detail="SPICE kernels not loaded")
    
    from astra.state.orbital_state import CelestialBody
    try:
        body = CelestialBody[body_name.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown body: {body_name}")
        
    try:
        state = kernel.get_body_state(body, epoch_j2000)
        return BodyStateResponse(**state.to_dict())
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/v1/windows/porkchop")
async def porkchop(request: PorkchopRequest) -> dict[str, Any]:
    """Calculate and return porkchop grid scan data for the mission launch window."""
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_string
    from astra.optimization.engine import compute_porkchop
    from astra.visualization.heatmap import build_porkchop_plot
    
    kernel = get_kernel()
    dsl = parse_mission_string(request.mission_yaml)
    mission = compile_mission(
        dsl,
        kernel.ephemeris if kernel._kernels_loaded else None
    )
    dep_epochs, tof_days, dv_grid = compute_porkchop(
        mission, kernel,
        n_dep=min(request.n_dep, 100),
        n_tof=min(request.n_tof, 100),
    )
    plot_data = build_porkchop_plot(
        dep_epochs, tof_days, dv_grid,
        ephemeris=kernel.ephemeris if kernel._kernels_loaded else None,
    )
    return plot_data.to_dict()
