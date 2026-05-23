"""Main physics kernel — unified interface to all physics computations."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from astra.physics.ephemeris import EphemerisEngine
from astra.physics.lambert import lambert_izzo
from astra.physics.propagator import propagate_two_body
from astra.state.orbital_state import CelestialBody, OrbitalState


class PhysicsKernel:
    """Unified physics interface. Owns the ephemeris engine and
    provides all trajectory computation primitives."""

    def __init__(self, kernel_dir: Path | str = "data/spice_kernels") -> None:
        self.ephemeris = EphemerisEngine(Path(kernel_dir))
        self._kernels_loaded = False

    def load(self) -> PhysicsKernel:
        """Load SPICE kernels. Call once at startup."""
        self.ephemeris.load_kernels()
        self._kernels_loaded = True
        return self

    def get_body_state(
        self, body: CelestialBody, epoch_j2000: float
    ) -> OrbitalState:
        return self.ephemeris.get_body_state(body, epoch_j2000)

    def lambert_solve(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        tof_seconds: float,
        mu: float,
        retrograde: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Solve Lambert problem. Returns (v1, v2, converged)."""
        return lambert_izzo(r1, r2, tof_seconds, mu, retrograde)

    def propagate(
        self, state: OrbitalState, dt_seconds: float
    ) -> OrbitalState:
        return propagate_two_body(state, dt_seconds)

    def compute_delta_v(
        self, state: OrbitalState, target_velocity: np.ndarray
    ) -> float:
        """Magnitude of velocity change needed [km/s]."""
        return float(np.linalg.norm(target_velocity - state.velocity))

    def epoch_from_date(self, iso_date: str) -> float:
        return self.ephemeris.epoch_from_date(iso_date)

    def date_from_epoch(self, epoch: float) -> str:
        return self.ephemeris.date_from_epoch(epoch)
