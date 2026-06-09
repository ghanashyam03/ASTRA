from __future__ import annotations

from typing import Literal
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
