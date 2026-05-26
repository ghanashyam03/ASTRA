"""ASTRA Mission Definition Language (DSL) module.
Provides schemas, parsers, and compilers for specifying space missions.
"""
from __future__ import annotations

from astra.dsl.compiler import (
    CompiledConstraint,
    CompiledMission,
    CompiledObjective,
    compile_mission,
)
from astra.dsl.parser import (
    parse_mission_file,
    parse_mission_string,
)
from astra.dsl.schema import (
    BodyOrbitSchema,
    ConstraintSchema,
    ConstraintType,
    LaunchWindowSchema,
    MissionDSL,
    ObjectiveDirection,
    ObjectiveMetric,
    ObjectiveSchema,
    OptimizationBudgetSchema,
    OptimizationSchema,
    OptimizationStrategySchema,
    OrbitSchema,
    PhysicsModelType,
    PhysicsSchema,
    PropulsionSchema,
    PropulsionTypeSchema,
    SpacecraftSchema,
    TrajectorySchema,
)

__all__ = [
    "MissionDSL",
    "SpacecraftSchema",
    "PropulsionSchema",
    "PropulsionTypeSchema",
    "OrbitSchema",
    "BodyOrbitSchema",
    "LaunchWindowSchema",
    "ConstraintSchema",
    "ConstraintType",
    "ObjectiveSchema",
    "ObjectiveMetric",
    "ObjectiveDirection",
    "PhysicsSchema",
    "PhysicsModelType",
    "OptimizationSchema",
    "OptimizationStrategySchema",
    "OptimizationBudgetSchema",
    "TrajectorySchema",
    "parse_mission_file",
    "parse_mission_string",
    "CompiledMission",
    "CompiledConstraint",
    "CompiledObjective",
    "compile_mission",
]
