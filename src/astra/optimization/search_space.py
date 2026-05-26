from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchSpace:
    """Defines the bounded parameter space for trajectory optimization."""
    departure_start: float    # J2000 seconds
    departure_end: float      # J2000 seconds
    tof_min: float            # seconds
    tof_max: float            # seconds

    def __post_init__(self) -> None:
        assert self.departure_end > self.departure_start
        assert self.tof_max > self.tof_min

    @property
    def departure_span_days(self) -> float:
        return (self.departure_end - self.departure_start) / 86400.0

    @property
    def tof_span_days(self) -> float:
        return (self.tof_max - self.tof_min) / 86400.0
