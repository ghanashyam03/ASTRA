"""Proves the existing constraint engine generalizes to N maneuvers without
modification — written as a verification test, not a feature implementation,
since check_propellant_budget already sums trajectory.delta_v_total generically."""

import numpy as np

from astra.constraints.physical import check_max_delta_v
from astra.constraints.propellant import check_propellant_budget
from astra.state.orbital_state import CelestialBody, OrbitalState
from astra.state.spacecraft import PropulsionSystem, PropulsionType, Spacecraft
from astra.state.trajectory import Maneuver, Trajectory


def _make_n_maneuver_trajectory(n_maneuvers: int, dv_each: float) -> Trajectory:
    states = [
        OrbitalState(
            epoch=float(i) * 86400.0,
            position=np.zeros(3),
            velocity=np.zeros(3),
            central_body=CelestialBody.SUN,
        )
        for i in range(n_maneuvers + 1)
    ]
    labels = ["TMI"] + [f"FLY_BODY{i}" for i in range(n_maneuvers - 2)] + ["MOI"]
    maneuvers = [
        Maneuver(
            epoch=float(i) * 86400.0,
            delta_v=np.array([dv_each, 0.0, 0.0]),
            label=labels[i] if i < len(labels) else f"M{i}",
        )
        for i in range(n_maneuvers)
    ]
    return Trajectory(states=states, maneuvers=maneuvers)


def test_two_maneuver_trajectory_baseline() -> None:
    traj = _make_n_maneuver_trajectory(2, 2.0)
    assert abs(traj.delta_v_total - 4.0) < 1e-9


def test_four_maneuver_trajectory_identical_math() -> None:
    """TMI + FLY + DSM + MOI = 4 maneuvers. Same summation logic must apply."""
    traj = _make_n_maneuver_trajectory(4, 1.0)
    assert abs(traj.delta_v_total - 4.0) < 1e-9
    result = check_max_delta_v(traj, max_dv_km_s=10.0)
    assert result.satisfied


def test_propellant_budget_n_maneuvers() -> None:
    prop = PropulsionSystem(PropulsionType.CHEMICAL, 450.0, 22000.0, 2400.0)
    sc = Spacecraft("TestCraft", 1800.0, prop)
    traj_2 = _make_n_maneuver_trajectory(2, 2.5)
    traj_4 = _make_n_maneuver_trajectory(4, 1.25)  # same total Δv, different N
    result_2 = check_propellant_budget(traj_2, sc)
    result_4 = check_propellant_budget(traj_4, sc)
    # Same total Δv must yield identical propellant requirement regardless of
    # how many maneuvers it's split across — proves N-maneuver-agnostic math.
    assert abs(result_2.required_propellant_kg - result_4.required_propellant_kg) < 1e-6
    assert result_2.satisfied == result_4.satisfied
