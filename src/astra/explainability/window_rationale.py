from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from astra.state.orbital_state import CelestialBody
from astra.state.trajectory import Trajectory

if TYPE_CHECKING:
    from astra.physics.ephemeris import EphemerisEngine


@dataclass
class WindowRationale:
    selected_departure_epoch: float
    selected_tof_days: float
    selected_dv_km_s: float
    departure_date_utc: str
    arrival_date_utc: str
    c3_km2_s2: float          # departure energy
    synodic_period_days: float
    planet_angle_deg: float   # phase angle at departure
    rationale_points: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "departure_date_utc": self.departure_date_utc,
            "arrival_date_utc": self.arrival_date_utc,
            "tof_days": round(self.selected_tof_days, 1),
            "delta_v_km_s": round(self.selected_dv_km_s, 4),
            "c3_km2_s2": round(self.c3_km2_s2, 4),
            "synodic_period_days": round(self.synodic_period_days, 1),
            "planet_phase_angle_deg": round(self.planet_angle_deg, 2),
            "rationale": self.rationale_points,
        }

def compute_synodic_period(body1: CelestialBody, body2: CelestialBody) -> float:
    """Synodic period in days between two planets orbiting the Sun."""
    # Approximate semi-major axes from Kepler's 3rd law using known periods
    PERIOD_DAYS: dict[str, float] = {
        "MERCURY": 87.97, "VENUS": 224.70, "EARTH": 365.25,
        "MARS": 686.97, "JUPITER": 4332.6, "SATURN": 10759.2,
        "URANUS": 30688.5, "NEPTUNE": 60182.0,
    }
    T1 = PERIOD_DAYS.get(body1.value, 365.25)
    T2 = PERIOD_DAYS.get(body2.value, 686.97)
    if T1 == T2:
        return float("inf")
    return abs(1.0 / (1.0 / T1 - 1.0 / T2))

def compute_c3(v_spacecraft_helio: np.ndarray, v_body_helio: np.ndarray) -> float:
    """C3 = |v_inf|² = |v_sc - v_body|² [km²/s²]."""
    v_inf = v_spacecraft_helio - v_body_helio
    return float(np.dot(v_inf, v_inf))

def build_window_rationale(
    trajectory: Trajectory,
    origin: CelestialBody,
    destination: CelestialBody,
    ephemeris: EphemerisEngine,
) -> WindowRationale:
    """Generate window selection rationale from trajectory data."""
    dep_epoch = trajectory.departure_epoch
    arr_epoch = trajectory.arrival_epoch
    tof_days = trajectory.duration_days
    dv_total = trajectory.delta_v_total

    # Planet positions and velocities at departure
    origin_state = ephemeris.get_body_state(origin, dep_epoch)
    dest_state = ephemeris.get_body_state(destination, dep_epoch)

    # Departure spacecraft velocity
    if trajectory.maneuvers:
        v_sc_dep = origin_state.velocity + trajectory.maneuvers[0].delta_v
    else:
        v_sc_dep = origin_state.velocity

    c3 = None
    if trajectory.metadata and "c3_km2_s2" in trajectory.metadata:
        c3 = trajectory.metadata["c3_km2_s2"]
    else:
        c3 = compute_c3(v_sc_dep, origin_state.velocity)
    synodic = compute_synodic_period(origin, destination)

    # Phase angle between origin and destination at departure
    r1 = origin_state.position
    r2 = dest_state.position
    cos_angle = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    phase_deg = math.degrees(math.acos(cos_angle))

    # Convert epochs to dates
    try:
        dep_date = ephemeris.date_from_epoch(dep_epoch)
        arr_date = ephemeris.date_from_epoch(arr_epoch)
    except Exception:
        dep_date = f"J2000+{dep_epoch/86400:.0f}d"
        arr_date = f"J2000+{arr_epoch/86400:.0f}d"

    # Build rationale from computed values
    rationale = [
        f"Departure {dep_date}: {origin.value}-{destination.value} phase angle is "
        f"{phase_deg:.1f}°, near optimal for {tof_days:.0f}-day transfer.",
        f"C3 = {c3:.2f} km²/s² (launch energy). Lower C3 means less fuel on departure burn.",
        f"Total Δv = {dv_total:.3f} km/s over {tof_days:.1f} days.",
        f"Synodic period is {synodic:.0f} days — next opportunity "
        f"~{synodic:.0f} days after this window.",
        f"TOF of {tof_days:.0f} days places arrival at {arr_date}.",
    ]

    return WindowRationale(
        selected_departure_epoch=dep_epoch,
        selected_tof_days=tof_days,
        selected_dv_km_s=dv_total,
        departure_date_utc=dep_date,
        arrival_date_utc=arr_date,
        c3_km2_s2=c3,
        synodic_period_days=synodic,
        planet_angle_deg=phase_deg,
        rationale_points=rationale,
    )
