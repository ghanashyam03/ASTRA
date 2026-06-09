from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OptimizeRequest(BaseModel):
    mission_yaml: str
    model_config = ConfigDict(json_schema_extra={
        "example": {"mission_yaml": "version: '1.0'\nmission_id: test\n..."}
    })


class PorkchopRequest(BaseModel):
    mission_yaml: str
    n_dep: int = Field(default=50, ge=5, le=100)
    n_tof: int = Field(default=50, ge=5, le=100)


class MCTSRequest(BaseModel):
    mission_yaml: str
    flyby_candidates: list[str] = ["VENUS", "EARTH"]
    n_iterations: int = Field(default=500, ge=50, le=5000)
    dv_budget: float = Field(default=15.0, gt=0, le=50.0)
