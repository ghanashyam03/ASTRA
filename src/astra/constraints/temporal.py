from __future__ import annotations

from dataclasses import dataclass
from astra.state.trajectory import Trajectory


@dataclass
class TemporalConstraintResult:
    constraint_type: str
    limit_days: float
    actual_days: float
    satisfied: bool
    margin_days: float


def check_max_duration(
    trajectory: Trajectory,
    max_duration_days: float,
) -> TemporalConstraintResult:
    """Checks that trajectory duration does not exceed the maximum limit."""
    actual = trajectory.duration_days
    satisfied = actual <= max_duration_days
    margin = max_duration_days - actual

    return TemporalConstraintResult(
        constraint_type="max_duration",
        limit_days=max_duration_days,
        actual_days=actual,
        satisfied=satisfied,
        margin_days=margin,
    )


def check_launch_window(
    departure_epoch_j2000: float,
    window_start_j2000: float,
    window_end_j2000: float,
) -> TemporalConstraintResult:
    """Checks that the departure epoch is within the launch window range."""
    actual_days = (departure_epoch_j2000 - window_start_j2000) / 86400.0
    window_days = (window_end_j2000 - window_start_j2000) / 86400.0
    satisfied = window_start_j2000 <= departure_epoch_j2000 <= window_end_j2000
    margin = min(
        (departure_epoch_j2000 - window_start_j2000) / 86400.0,
        (window_end_j2000 - departure_epoch_j2000) / 86400.0,
    )

    return TemporalConstraintResult(
        constraint_type="launch_window",
        limit_days=window_days,
        actual_days=actual_days,
        satisfied=satisfied,
        margin_days=margin,
    )
