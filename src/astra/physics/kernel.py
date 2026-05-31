"""Main physics kernel — unified interface to all physics computations.
Coordinates orbital propagation, ephemeris queries, and Lambert boundary value solvers.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from astra.physics.ephemeris import EphemerisEngine, EphemerisTarget
from astra.physics.lambert import lambert_izzo
from astra.physics.propagator import Integrator, propagate_two_body
from astra.state.orbital_state import CelestialBody, OrbitalState

if TYPE_CHECKING:
    from astra.data.cache import EphemerisCache

class PhysicsKernel:
    """Unified physics interface. Owns the ephemeris engine and
    provides all trajectory computation primitives."""

    def __init__(
        self,
        kernel_dir: Path | str = "data/spice_kernels",
        cache: EphemerisCache | None = None,
    ) -> None:
        from astra.data.cache import EphemerisCache as _Cache
        if cache is None:
            cache = _Cache(max_entries=50_000)
        self.ephemeris = EphemerisEngine(
            Path(kernel_dir),
            cache=cache,
        )
        self._kernels_loaded = False

    def load(self) -> PhysicsKernel:
        """Load SPICE kernels. Call once at startup."""
        self.ephemeris.load_kernels()
        self._kernels_loaded = True
        return self

    def get_body_state(
        self,
        target: CelestialBody | EphemerisTarget,
        epoch_j2000: float,
        observer: CelestialBody | EphemerisTarget | str = "SUN",
        frame: str = "ECLIPJ2000",
        central_body: CelestialBody | None = None,
    ) -> OrbitalState:
        """Query state vectors relative to an observer using precise ephemerides."""
        return self.ephemeris.get_body_state(
            target, epoch_j2000, observer=observer, frame=frame, central_body=central_body
        )

    def lambert_solve(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        tof_seconds: float,
        mu: float,
        retrograde: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Solve Lambert BVP using Householder/Halley iterative formulation.
        
        Raises LambertSingularityError or LambertConvergenceError on failures.
        """
        return lambert_izzo(r1, r2, tof_seconds, mu, retrograde)

    def propagate(
        self,
        state: OrbitalState,
        dt_seconds: float,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        integrator: Integrator | None = None,
    ) -> OrbitalState:
        """Numerically propagate Keplerian state using configurable integrators and
        collision safety checks.
        """
        return propagate_two_body(
            state, dt_seconds, rtol=rtol, atol=atol, integrator=integrator
        )

    def compute_delta_v(
        self, state: OrbitalState, target_velocity: np.ndarray
    ) -> float:
        """Magnitude of velocity change needed [km/s]."""
        return float(np.linalg.norm(target_velocity - state.velocity))

    def epoch_from_date(self, iso_date: str) -> float:
        """Convert ISO date string to J2000 seconds via SPICE."""
        return self.ephemeris.epoch_from_date(iso_date)

    def date_from_epoch(self, epoch: float) -> str:
        """Convert J2000 seconds to ISO date string via SPICE."""
        return self.ephemeris.date_from_epoch(epoch)
