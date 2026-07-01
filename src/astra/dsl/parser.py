"""Parse YAML or JSON mission specs into MissionDSL objects."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from astra.dsl.schema import MissionDSL


def parse_mission_file(path: str | Path) -> MissionDSL:
    """Parse YAML or JSON file → MissionDSL. Raises on invalid schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mission file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .yaml or .json")
    return MissionDSL.model_validate(data)


def parse_mission_string(text: str, fmt: str = "yaml") -> MissionDSL:
    """Parse YAML or JSON string → MissionDSL."""
    if fmt == "yaml":
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return MissionDSL.model_validate(data)


def parse_mission_yaml(path: str | Path) -> MissionDSL:
    """Parse YAML file → MissionDSL. Alias for parse_mission_file."""
    return parse_mission_file(path)
