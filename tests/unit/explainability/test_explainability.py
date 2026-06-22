import numpy as np

from astra.dsl.compiler import CompiledConstraint
from astra.dsl.schema import ConstraintType
from astra.state.orbital_state import CelestialBody, OrbitalState
from astra.state.trajectory import Maneuver, Trajectory


def make_test_trajectory() -> Trajectory:
    s0 = OrbitalState(
        epoch=0.0,
        position=np.array([1.496e8, 0.0, 0.0]),
        velocity=np.array([0.0, 29.78, 0.0]),
        central_body=CelestialBody.SUN,
    )
    s1 = OrbitalState(
        epoch=200 * 86400.0,
        position=np.array([0.0, 2.279e8, 0.0]),
        velocity=np.array([-24.1, 0.0, 0.0]),
        central_body=CelestialBody.SUN,
    )
    m1 = Maneuver(epoch=0.0, delta_v=np.array([0.0, 3.62, 0.0]), label="TMI")
    m2 = Maneuver(epoch=200 * 86400.0, delta_v=np.array([2.10, 0.0, 0.0]), label="MOI")
    return Trajectory(states=[s0, s1], maneuvers=[m1, m2])


def test_delta_v_decomposition() -> None:
    from astra.explainability.deltav_decomp import decompose_delta_v

    traj = make_test_trajectory()
    decomp = decompose_delta_v(traj)
    assert abs(decomp.total_km_s - traj.delta_v_total) < 1e-9
    assert len(decomp.components) == 2
    assert decomp.components[0].label == "TMI"
    assert decomp.components[1].label == "MOI"
    fractions = [c.fraction_of_total for c in decomp.components]
    assert abs(sum(fractions) - 1.0) < 1e-9
    assert abs(decomp.margin_km_s - decomp.total_km_s * 0.03) < 1e-9


def test_constraint_analysis_satisfied() -> None:
    from astra.explainability.constraint_analysis import analyze_constraints

    traj = make_test_trajectory()
    constraints = [
        CompiledConstraint(type=ConstraintType.MAX_DELTA_V, limit=8.0, hard=True),
        CompiledConstraint(type=ConstraintType.MAX_DURATION, limit=250 * 86400.0, hard=True),
    ]
    analysis = analyze_constraints(traj, constraints)
    assert analysis.all_satisfied
    # Both maneuvers sum to 3.62 + 2.10 = 5.72 km/s < 8.0 → satisfied
    dv_status = next(s for s in analysis.statuses if s.type == ConstraintType.MAX_DELTA_V)
    assert dv_status.satisfied
    assert dv_status.margin_pct > 0


def test_constraint_analysis_violated() -> None:
    from astra.explainability.constraint_analysis import analyze_constraints

    traj = make_test_trajectory()
    constraints = [
        CompiledConstraint(type=ConstraintType.MAX_DELTA_V, limit=4.0, hard=True),
    ]
    analysis = analyze_constraints(traj, constraints)
    assert not analysis.all_satisfied


def test_pareto_analysis() -> None:
    from astra.explainability.pareto_analysis import analyze_pareto

    trajs = []
    for dv, days in [(4.0, 200), (5.0, 170), (6.5, 140), (8.0, 120)]:
        s0 = OrbitalState(
            epoch=0.0,
            position=np.array([1.496e8, 0.0, 0.0]),
            velocity=np.array([0.0, 29.78, 0.0]),
            central_body=CelestialBody.SUN,
        )
        s1 = OrbitalState(
            epoch=days * 86400.0,
            position=np.array([0.0, 2.279e8, 0.0]),
            velocity=np.array([-24.0, 0.0, 0.0]),
            central_body=CelestialBody.SUN,
        )
        half = dv / 2
        m1 = Maneuver(epoch=0.0, delta_v=np.array([0.0, half, 0.0]))
        m2 = Maneuver(epoch=days * 86400.0, delta_v=np.array([half, 0.0, 0.0]))
        trajs.append(Trajectory(states=[s0, s1], maneuvers=[m1, m2]))

    pa = analyze_pareto(trajs)
    assert pa.fuel_optimal.delta_v_total == 4.0
    assert pa.time_optimal.duration_days == 120.0
    assert pa.tradeoff_km_s_per_day > 0


def test_full_explanation_trace() -> None:
    from astra.dsl.compiler import CompiledConstraint, CompiledMission, CompiledObjective
    from astra.explainability.engine import explain
    from astra.state.spacecraft import PropulsionSystem, PropulsionType, Spacecraft

    traj = make_test_trajectory()
    prop = PropulsionSystem(PropulsionType.CHEMICAL, 450.0, 22000.0, 2400.0)
    sc = Spacecraft("TestCraft", 1800.0, prop)
    mission = CompiledMission(
        mission_id="test_explain",
        spacecraft=sc,
        origin_body=CelestialBody.EARTH,
        destination_body=CelestialBody.MARS,
        departure_epoch_start=0.0,
        departure_epoch_end=86400.0 * 365,
        tof_min_seconds=120 * 86400.0,
        tof_max_seconds=280 * 86400.0,
        tof_step_seconds=86400.0,
        constraints=[CompiledConstraint(ConstraintType.MAX_DELTA_V, 8.0, True)],
        objectives=[CompiledObjective("delta_v_total", "minimize", 1.0)],
        seed=42,
        max_evaluations=1000,
    )
    trace = explain(traj, mission, pareto_front=[traj])
    assert trace.mission_id == "test_explain"
    d = trace.to_dict()
    assert "delta_v_decomposition" in d
    assert "constraint_analysis" in d
    assert d["delta_v_decomposition"]["total_km_s"] > 0
