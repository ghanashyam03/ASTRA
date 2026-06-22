"""Single-flyby historical validation — proves the Prompt 28 feasibility gate
accepts real, individually-completed planetary flybys. No chaining — each flyby
is evaluated in isolation using only its own approach v_infinity and periapsis.

These tests are deliberately tolerant: patched-conics two-body flyby physics
cannot reproduce historical Δv to high precision without real orbit determination
data. The purpose is to confirm PHYSICAL PLAUSIBILITY (the gate accepts a flyby
that we know, historically, actually happened) — not numerical precision.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from astra.physics.flyby import (
    check_flyby_feasibility,
    compute_flyby_turn_angle,
    impact_parameter_from_periapsis,
)
from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody


def _load_params(filename: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(f"data/benchmarks/{filename}").read_text()))


def _validate_historical_flyby(
    params: dict[str, Any], assumed_turn_deg_range: tuple[float, float]
) -> None:
    """Shared validation: a real, completed flyby must always be is_achievable_*
    True for SOME modest turn angle within the given plausible range — proving the
    gate doesn't falsely reject real, historically-flown geometry.
    """
    body = params["flyby_body"]
    v_inf = params["v_inf_approach_km_s"]
    R_body = PHYSICAL_RADIUS[CelestialBody[body.upper()]]
    r_p = R_body + params["periapsis_altitude_km"]

    turn_rad = compute_flyby_turn_angle(v_inf, r_p, body)
    turn_deg = math.degrees(turn_rad)

    assert assumed_turn_deg_range[0] <= turn_deg <= assumed_turn_deg_range[1], (
        f"{params['mission']} {body} flyby: computed turn {turn_deg:.2f}° outside "
        f"plausible range {assumed_turn_deg_range} — verify v_inf/altitude citation"
    )

    feas = check_flyby_feasibility(v_inf, turn_rad, body)
    assert feas.is_achievable_unpowered, (
        f"{params['mission']} {body} flyby must be achievable unpowered — "
        f"this geometry historically happened. Gate rejection: {feas.rejection_reason}"
    )

    b = impact_parameter_from_periapsis(
        r_p,
        v_inf,
        __import__("astra.state.orbital_state", fromlist=["GM"]).GM[body.upper()],
    )
    print(
        f"\n{params['mission']} {body} flyby ({params['date']}): "
        f"turn={turn_deg:.2f}°, periapsis={r_p:.0f} km, impact param={b:.0f} km"
    )


def test_galileo_venus_1990() -> None:
    params = _load_params("galileo_venus_1990_params.json")
    # Plausible range is intentionally wide — verify against citation, narrow if confident
    _validate_historical_flyby(params, assumed_turn_deg_range=(10.0, 90.0))


def test_galileo_earth_1990() -> None:
    params = _load_params("galileo_earth1_1990_params.json")
    _validate_historical_flyby(params, assumed_turn_deg_range=(10.0, 120.0))


def test_messenger_venus_2006() -> None:
    params = _load_params("messenger_venus1_params.json")
    _validate_historical_flyby(params, assumed_turn_deg_range=(5.0, 90.0))


def test_gate_distinguishes_real_from_impossible() -> None:
    """Sanity check: the SAME body, SAME v_inf as a real flyby, but with an
    artificially impossible turn angle (the historical turn + 130°) must be
    rejected — proving the gate has real discriminating power, not just
    always returning True."""
    params = _load_params("galileo_venus_1990_params.json")
    body = params["flyby_body"]
    v_inf = params["v_inf_approach_km_s"]
    R_body = PHYSICAL_RADIUS[CelestialBody[body.upper()]]
    r_p = R_body + params["periapsis_altitude_km"]
    real_turn = compute_flyby_turn_angle(v_inf, r_p, body)

    impossible_turn = real_turn + math.radians(130.0)
    feas = check_flyby_feasibility(v_inf, impossible_turn, body)
    assert (
        feas.is_achievable_at_all is False
    ), "An artificially inflated turn angle must be rejected by the gate"
    print(
        f"\nReal turn {math.degrees(real_turn):.1f}° accepted. "
        f"Inflated turn {math.degrees(impossible_turn):.1f}° correctly rejected: "
        f"{feas.rejection_reason}"
    )
