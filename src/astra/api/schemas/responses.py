from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    spice_loaded: bool


class JobSubmittedResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "failed"]


class BodyStateResponse(BaseModel):
    epoch_j2000: float
    position_km: list[float]
    velocity_km_s: list[float]
    frame: str
    central_body: str
    r_km: float
    v_km_s: float
    sma_km: float
    eccentricity: float


class ErrorResponse(BaseModel):
    detail: str
    error_type: str | None = None


class MissionLegResponse(BaseModel):
    phase: str
    origin: str
    destination: str
    departure_epoch_j2000: float
    arrival_epoch_j2000: float
    delta_v_km_s: float
    tof_days: float
    metadata: dict[str, Any]


class OptimizationInfo(BaseModel):
    n_trials: int
    n_feasible: int
    wall_time_s: float


class MissionSummaryResponse(BaseModel):
    mission_id: str
    status: str
    route: str
    total_delta_v_km_s: float
    total_duration_days: float
    legs: list[MissionLegResponse]
    optimization: OptimizationInfo

