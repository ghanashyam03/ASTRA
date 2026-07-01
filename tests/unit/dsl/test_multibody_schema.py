import pytest
from pydantic import ValidationError

from astra.dsl.compiler import compile_mission
from astra.dsl.parser import parse_mission_string
from astra.dsl.schema import FlybyBodySchema, MultiBodyTrajectorySchema


def test_flyby_body_requires_burn_budget_if_allowed() -> None:
    with pytest.raises(ValidationError):
        FlybyBodySchema(body="VENUS", powered_burn_allowed=True, max_powered_burn_km_s=0.0)


def test_flyby_body_valid_with_budget() -> None:
    fb = FlybyBodySchema(body="VENUS", powered_burn_allowed=True, max_powered_burn_km_s=0.5)
    assert fb.max_powered_burn_km_s == 0.5


def test_multibody_dsm_budget_defaults_to_zero() -> None:
    """No implicit correction budget — must be explicitly declared."""
    from astra.dsl.schema import BodyOrbitSchema

    mbt = MultiBodyTrajectorySchema(
        origin=BodyOrbitSchema(body="Earth"),
        destination=BodyOrbitSchema(body="Mars"),
    )
    assert mbt.dsm_budget_km_s == 0.0


def test_periapsis_range_validation() -> None:
    with pytest.raises(ValidationError):
        FlybyBodySchema(body="VENUS", min_periapsis_alt_km=5000.0, max_periapsis_alt_km=1000.0)


# Precedence and compiler verification tests
YAML_WITH_BOTH_TRAJECTORIES = """
version: "1.0"
mission_id: test_precedence
spacecraft:
  name: TestCraft
  dry_mass_kg: 1000.0
  fuel_mass_kg: 500.0
  propulsion:
    type: chemical
    isp_seconds: 310.0
trajectory:
  origin:
    body: Earth
  destination:
    body: Mars
multi_body_trajectory:
  origin:
    body: Venus
  destination:
    body: Jupiter
  dsm_budget_km_s: 1.5
  max_revs_per_leg: 2
  flyby_sequence:
    - body: Mars
      min_periapsis_alt_km: 400.0
      max_periapsis_alt_km: 10000.0
      powered_burn_allowed: true
      max_powered_burn_km_s: 0.8
launch_window:
  start: "2030-01-01T00:00:00Z"
  end: "2031-01-01T00:00:00Z"
  tof_min_days: 90
  tof_max_days: 300
objectives:
  - metric: delta_v_total
    direction: minimize
"""


def test_precedence_behavior() -> None:
    """Show compile output uses multi_body_trajectory fields when both are specified."""
    dsl = parse_mission_string(YAML_WITH_BOTH_TRAJECTORIES)
    assert dsl.trajectory is not None
    assert dsl.multi_body_trajectory is not None

    compiled = compile_mission(dsl)
    # Origin and destination should match multi_body_trajectory, not trajectory
    from astra.state.orbital_state import CelestialBody

    assert compiled.origin_body == CelestialBody.VENUS
    assert compiled.destination_body == CelestialBody.JUPITER

    # Extra fields should be populated correctly
    assert compiled.dsm_budget_km_s == 1.5
    assert compiled.max_revs_per_leg == 2
    assert len(compiled.flyby_sequence) == 1
    assert compiled.flyby_sequence[0] == {
        "body": "MARS",
        "min_alt_km": 400.0,
        "max_alt_km": 10000.0,
        "powered_allowed": True,
        "max_powered_km_s": 0.8,
        "max_revs": 2,
    }
