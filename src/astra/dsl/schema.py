"""Pydantic v2 schema for the ASTRA Mission Definition Language."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class PropulsionTypeSchema(StrEnum):
    CHEMICAL = "chemical"
    ELECTRIC = "electric"
    HYBRID = "hybrid"

class PropulsionSchema(BaseModel):
    type: PropulsionTypeSchema = PropulsionTypeSchema.CHEMICAL
    isp_seconds: Annotated[float, Field(gt=0, le=10000)] = 450.0
    thrust_newtons: float = 0.0   # 0 = impulsive model

class SpacecraftSchema(BaseModel):
    name: str
    dry_mass_kg: Annotated[float, Field(gt=0)]
    fuel_mass_kg: Annotated[float, Field(gt=0)]
    propulsion: PropulsionSchema

    @model_validator(mode="after")
    def validate_mass(self) -> SpacecraftSchema:
        total = self.dry_mass_kg + self.fuel_mass_kg
        if total <= 0:
            raise ValueError("Total mass must be positive")
        return self

class OrbitSchema(BaseModel):
    type: str = "circular"
    altitude_km: float | None = None      # for circular
    periapsis_km: float | None = None     # for elliptical
    apoapsis_km: float | None = None      # for elliptical

    @model_validator(mode="after")
    def validate_orbit(self) -> OrbitSchema:
        if self.type == "elliptical":
            if self.periapsis_km is None or self.apoapsis_km is None:
                raise ValueError(
                    "Both periapsis_km and apoapsis_km must be provided for elliptical orbits"
                )
            if self.apoapsis_km <= self.periapsis_km:
                raise ValueError("apoapsis must be greater than periapsis radius from center")
        elif self.type == "circular":
            if self.altitude_km is None:
                raise ValueError("altitude_km must be provided for circular orbits")
        return self

class BodyOrbitSchema(BaseModel):
    body: str
    orbit: OrbitSchema | None = None

class LaunchWindowSchema(BaseModel):
    start: datetime
    end: datetime
    resolution_days: Annotated[float, Field(gt=0)] = 1.0
    tof_min_days: Annotated[float, Field(gt=0)] = 30.0
    tof_max_days: Annotated[float, Field(gt=0)] = 400.0

    @model_validator(mode="after")
    def validate_window(self) -> LaunchWindowSchema:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if self.tof_max_days <= self.tof_min_days:
            raise ValueError("tof_max_days must exceed tof_min_days")
        return self

class ConstraintType(StrEnum):
    MAX_DELTA_V = "max_delta_v"
    MAX_DURATION = "max_duration"
    MIN_PERIAPSIS = "min_periapsis"
    MAX_C3 = "max_c3"

class ConstraintSchema(BaseModel):
    type: ConstraintType
    value_km_s: float | None = None
    value_days: float | None = None
    value_km: float | None = None
    body: str | None = None
    hard: bool = True

class ObjectiveDirection(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class ObjectiveMetric(StrEnum):
    DELTA_V_TOTAL = "delta_v_total"
    TIME_OF_FLIGHT = "time_of_flight"
    ARRIVAL_MASS = "arrival_mass"

class ObjectiveSchema(BaseModel):
    metric: ObjectiveMetric
    direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE
    weight: Annotated[float, Field(gt=0)] = 1.0

class PhysicsModelType(StrEnum):
    TWO_BODY = "two_body"
    PATCHED_CONICS = "patched_conics"
    N_BODY = "n_body"

class PhysicsSchema(BaseModel):
    models: list[PhysicsModelType] = [PhysicsModelType.PATCHED_CONICS]
    ephemeris: str = "DE440"
    n_body_perturbations: list[str] = []

class OptimizationStrategySchema(StrEnum):
    BAYESIAN = "bayesian"
    GRADIENT = "gradient"
    EVOLUTIONARY = "evolutionary"
    HYBRID = "hybrid"

class OptimizationBudgetSchema(BaseModel):
    max_evaluations: Annotated[int, Field(gt=0)] = 5000
    time_limit_seconds: Annotated[float, Field(gt=0)] = 300.0

class OptimizationSchema(BaseModel):
    strategy: OptimizationStrategySchema = OptimizationStrategySchema.BAYESIAN
    budget: OptimizationBudgetSchema = OptimizationBudgetSchema()
    seed: int = 42
    neural_acceleration: bool = False

class TrajectorySchema(BaseModel):
    origin: BodyOrbitSchema
    destination: BodyOrbitSchema

class MissionDSL(BaseModel):
    """Root model for ASTRA Mission Definition Language."""
    version: str = "1.0"
    mission_id: str
    spacecraft: SpacecraftSchema
    trajectory: TrajectorySchema
    launch_window: LaunchWindowSchema
    constraints: list[ConstraintSchema] = []
    objectives: list[ObjectiveSchema]
    physics: PhysicsSchema = PhysicsSchema()
    optimization: OptimizationSchema = OptimizationSchema()
