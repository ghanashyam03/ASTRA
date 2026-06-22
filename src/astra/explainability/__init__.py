"""ASTRA explainability engine.
Provides structural mathematical decompositions, constraint analysis, and window rationales.
"""

from __future__ import annotations

from astra.explainability.constraint_analysis import (
    ConstraintAnalysis,
    ConstraintStatus,
    analyze_constraints,
)
from astra.explainability.deltav_decomp import (
    DeltaVComponent,
    DeltaVDecomposition,
    decompose_delta_v,
)
from astra.explainability.engine import ExplanationTrace, explain
from astra.explainability.pareto_analysis import ParetoAnalysis, ParetoPoint, analyze_pareto
from astra.explainability.window_rationale import WindowRationale, build_window_rationale

__all__ = [
    "DeltaVComponent",
    "DeltaVDecomposition",
    "decompose_delta_v",
    "WindowRationale",
    "build_window_rationale",
    "ConstraintStatus",
    "ConstraintAnalysis",
    "analyze_constraints",
    "ParetoPoint",
    "ParetoAnalysis",
    "analyze_pareto",
    "ExplanationTrace",
    "explain",
]
