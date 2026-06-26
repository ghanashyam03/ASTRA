"""Screening-level estimate of how much a planet moves along its own orbit
during the time a spacecraft spends inside its sphere of influence, relative
to the SOI radius itself. A simple, robust, low-risk metric for flagging
which bodies' flyby physics deserve closer (Prompt 45) investigation.

TODO: Perform high-fidelity numerical propagation audits to quantify actual
trajectory reconstruction error, as a large ratio does not guarantee large
errors, and a small ratio does not guarantee exact patched-conics matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from astra.physics.kernel import PhysicsKernel
from astra.physics.soi import compute_soi_radius
from astra.state.orbital_state import CelestialBody


@dataclass
class DisplacementRatioResult:
    body: str
    v_inf_km_s: float
    soi_radius_km: float
    crossing_duration_s: float
    planet_orbital_speed_km_s: float
    planet_displacement_km: float
    ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "body": self.body,
            "v_inf_km_s": round(self.v_inf_km_s, 3),
            "soi_radius_km": round(self.soi_radius_km, 0),
            "crossing_duration_days": round(self.crossing_duration_s / 86400.0, 2),
            "planet_orbital_speed_km_s": round(self.planet_orbital_speed_km_s, 3),
            "planet_displacement_km": round(self.planet_displacement_km, 0),
            "ratio": round(self.ratio, 3),
        }


def soi_crossing_displacement_ratio(
    body: str,
    v_inf_km_s: float,
    encounter_epoch_j2000: float,
    kernel: PhysicsKernel,
) -> DisplacementRatioResult:
    """Compute the displacement ratio for one (body, v_inf, epoch) case."""
    cb = CelestialBody[body.upper()]
    r_soi = compute_soi_radius(body)
    t_cross = 2.0 * r_soi / v_inf_km_s
    v_planet_vec = kernel.get_body_state(cb, encounter_epoch_j2000).velocity
    v_planet = float(np.linalg.norm(v_planet_vec))
    d_planet = v_planet * t_cross
    return DisplacementRatioResult(
        body=body.upper(),
        v_inf_km_s=v_inf_km_s,
        soi_radius_km=r_soi,
        crossing_duration_s=t_cross,
        planet_orbital_speed_km_s=v_planet,
        planet_displacement_km=d_planet,
        ratio=d_planet / r_soi,
    )
