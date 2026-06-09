"""Trajectories routes for the ASTRA API.

Contains:
- GET /v1/trajectories/{trajectory_id}: Get saved trajectory data.
- GET /v1/trajectories/{trajectory_id}/explanation: Get explainability breakdown for a trajectory.
"""

from typing import Any
from fastapi import APIRouter, HTTPException

from astra.api.app import get_store

router = APIRouter(tags=["trajectories"])


@router.get("/v1/trajectories/{trajectory_id}")
async def get_trajectory(trajectory_id: str) -> dict[str, Any]:
    """Retrieve saved trajectory coordinates, maneuvers, and metadata by ID."""
    store = get_store()
    result = store.get_trajectory(trajectory_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    return result


@router.get("/v1/trajectories/{trajectory_id}/explanation")
async def get_explanation(trajectory_id: str) -> dict[str, Any]:
    """Retrieve explanation trace and constraints evaluation report by trajectory ID."""
    store = get_store()
    result = store.get_trajectory(trajectory_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    explanation = result.get("explanation")
    if not isinstance(explanation, dict):
        raise HTTPException(status_code=404, detail="No explanation for this trajectory")
    return explanation
