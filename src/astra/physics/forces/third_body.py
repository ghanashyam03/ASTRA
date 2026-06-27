"""Third-body gravitational perturbation from a planet's time-varying
ephemeris position, for use in heliocentric propagation. Implements the
existing ForceModel interface — composes with PointMassGravity(mu_sun) via
the existing build_ode() machinery with zero changes to the propagator.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import numpy as np

from astra.physics.forces.gravity import ForceModel
from astra.state.orbital_state import GM, CelestialBody

if TYPE_CHECKING:
    from astra.physics.kernel import PhysicsKernel


class EphemerisThirdBodyPerturbation(ForceModel):
    """Gravitational acceleration from a perturbing body, looked up from the
    kernel's ephemeris at the CURRENT simulation time, for use in a
    heliocentric propagation where the perturbing body's position is not
    fixed.

    TODO: Conduct a follow-up study analyzing how these boundary state
    deviations (frozen vs. moving) translate to actual C3 departure energy
    and total mission Delta-V cost discrepancies inside multi-leg trajectory conics.
    """

    def __init__(
        self, kernel: PhysicsKernel, perturbing_body: str, reference_epoch_j2000: float
    ) -> None:
        """reference_epoch_j2000 is the real epoch corresponding to t=0 in the
        propagator's local time — acceleration(state_vec, t) will query the
        kernel at reference_epoch_j2000 + t.
        """
        self.kernel = kernel
        self.body = CelestialBody[perturbing_body.upper()]
        self.mu = GM[perturbing_body.upper()]
        self.reference_epoch = reference_epoch_j2000

    def acceleration(self, state_vec: np.ndarray, t: float) -> np.ndarray:
        """state_vec is the spacecraft's HELIOCENTRIC [x,y,z,vx,vy,vz]."""
        r_sc_helio = state_vec[:3]
        planet_state = self.kernel.get_body_state(self.body, self.reference_epoch + t)
        r_planet_helio = planet_state.position
        r_sc_to_planet = r_sc_helio - r_planet_helio
        dist = float(np.linalg.norm(r_sc_to_planet))
        if dist < 1e-6:
            return np.zeros(3, dtype=np.float64)
        accel = -self.mu * r_sc_to_planet / (dist**3)
        return typing.cast(np.ndarray, accel)
