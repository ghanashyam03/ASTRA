"""Health routes for the ASTRA API.

Contains:
- GET /v1/health: Check application health status and SPICE loading.
"""

from fastapi import APIRouter, Depends

from astra.api.schemas.responses import HealthResponse
from astra.api.app import get_kernel
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
