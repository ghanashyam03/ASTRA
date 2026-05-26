import pytest
from pydantic import ValidationError

VALID_YAML_TEXT = """
version: "1.0"
mission_id: test_mission
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
launch_window:
  start: "2030-01-01T00:00:00Z"
  end: "2031-01-01T00:00:00Z"
  tof_min_days: 90
  tof_max_days: 300
objectives:
  - metric: delta_v_total
    direction: minimize
"""

def test_valid_yaml_parses() -> None:
    from astra.dsl.parser import parse_mission_string
    m = parse_mission_string(VALID_YAML_TEXT)
    assert m.mission_id == "test_mission"
    assert m.spacecraft.name == "TestCraft"

def test_invalid_mass_raises() -> None:
    bad = VALID_YAML_TEXT.replace("dry_mass_kg: 1000.0", "dry_mass_kg: -5.0")
    with pytest.raises(ValidationError):
        from astra.dsl.parser import parse_mission_string
        parse_mission_string(bad)

def test_invalid_window_raises() -> None:
    bad = VALID_YAML_TEXT.replace("end: \"2031-01-01", "end: \"2029-01-01")
    with pytest.raises(ValidationError):
        from astra.dsl.parser import parse_mission_string
        parse_mission_string(bad)

def test_reference_mission_parses() -> None:
    from astra.dsl.parser import parse_mission_file
    m = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    assert m.mission_id == "earth_mars_2031_fuel_optimal"
    assert m.spacecraft.fuel_mass_kg == 2400.0
    assert len(m.constraints) == 2
    assert len(m.objectives) == 2

def test_compiler_produces_mission() -> None:
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_string
    m = parse_mission_string(VALID_YAML_TEXT)
    compiled = compile_mission(m)
    assert compiled.mission_id == "test_mission"
    assert compiled.spacecraft.dry_mass_kg == 1000.0
    # TOF bounds in seconds
    assert compiled.tof_min_seconds == 90 * 86400.0
    assert compiled.tof_max_seconds == 300 * 86400.0
