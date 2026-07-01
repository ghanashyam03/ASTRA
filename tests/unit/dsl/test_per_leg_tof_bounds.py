"""Tests for per-leg TOF bounds in DSL schema and compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from astra.dsl.compiler import compile_mission
from astra.dsl.parser import parse_mission_string, parse_mission_yaml
from astra.dsl.schema import FlybyBodySchema, MissionDSL


@pytest.fixture
def galileo_yaml_path() -> str:
    return "data/benchmarks/galileo_veega_1989.yaml"


@pytest.fixture
def simple_mission_dsl() -> MissionDSL:
    yaml_text = """
version: "1.0"
mission_id: simple_test_mission
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
    body: Earth
  destination:
    body: Mars
  max_revs_per_leg: 0
  flyby_sequence:
    - body: Venus
      min_periapsis_alt_km: 300.0
      max_periapsis_alt_km: 20000.0
      powered_burn_allowed: false
launch_window:
  start: "2030-01-01T00:00:00Z"
  end: "2031-01-01T00:00:00Z"
  resolution_days: 1
  tof_min_days: 90
  tof_max_days: 300
objectives:
  - metric: delta_v_total
    direction: minimize
"""
    return parse_mission_string(yaml_text)


def test_flyby_body_schema_accepts_per_leg_tof() -> None:
    fb = FlybyBodySchema(body="VENUS", tof_min_days=90.0, tof_max_days=200.0, max_revs=1)
    assert fb.tof_min_days == 90.0
    assert fb.tof_max_days == 200.0
    assert fb.max_revs == 1


def test_flyby_body_schema_rejects_bad_tof_bounds() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="tof_max_days"):
        FlybyBodySchema(body="VENUS", tof_min_days=200.0, tof_max_days=100.0)


def test_compiler_propagates_per_leg_tof_bounds(galileo_yaml_path: Path) -> None:
    """Galileo YAML per-leg bounds must appear in compiled mission."""
    dsl = parse_mission_yaml(galileo_yaml_path)
    mission = compile_mission(dsl)
    # 3 flyby legs + 1 final leg = 4 entries
    assert len(mission.leg_tof_bounds) == 4
    # Earth→Earth leg (index 2) must have tof_max >= 600 days
    assert mission.leg_tof_bounds[2][1] >= 600 * 86400.0


def test_compiler_global_tof_fallback(
    simple_mission_dsl: MissionDSL,
) -> None:
    """FlybyBodySchema with no per-leg override must use LaunchWindowSchema global."""
    mission = compile_mission(simple_mission_dsl)
    if mission.leg_tof_bounds:
        for i, (tmin, tmax) in enumerate(mission.leg_tof_bounds):
            assert tmin == mission.tof_min_seconds, f"leg {i} tof_min mismatch"
            assert tmax == mission.tof_max_seconds, f"leg {i} tof_max mismatch"
