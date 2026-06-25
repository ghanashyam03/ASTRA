import math

from astra.optimization.chain_solver import _achieved_turn, _solve_periapsis_bisection
from astra.state.orbital_state import GM


def test_achieved_turn_is_monotonic_decreasing() -> None:
    """Empirically confirm the monotonicity proof holds numerically, not
    just analytically — a real safeguard against a sign error in
    implementation."""
    mu = GM["VENUS"]
    v_in, v_out = 9.0, 7.0
    r_values = [7000.0, 10000.0, 20000.0, 50000.0, 100000.0]
    turns = [_achieved_turn(r, v_in, v_out, mu) for r in r_values]
    for i in range(len(turns) - 1):
        assert turns[i] > turns[i + 1], (
            f"achieved_turn must strictly decrease with r_p: "
            f"turn({r_values[i]})={turns[i]:.6f} <= turn({r_values[i + 1]})={turns[i + 1]:.6f}"
        )


def test_bisection_matches_forward_formula() -> None:
    """The bisection-solved r_p, fed back through the forward formula, must
    reproduce the requested turn angle to high precision — far tighter than
    the old grid search's resolution."""
    mu = GM["EARTH"]
    v_in, v_out = 6.0, 6.0  # equal magnitudes — near the unpowered case
    target_deg = 40.0
    r_p = _solve_periapsis_bisection(
        math.radians(target_deg), v_in, v_out, mu, r_min=6678.0, r_max=100000.0
    )
    assert r_p is not None
    achieved = _achieved_turn(r_p, v_in, v_out, mu)
    assert abs(math.degrees(achieved) - target_deg) < 1e-6, (
        "Bisection-solved periapsis must reproduce the target turn to 1e-6°, "
        "far tighter than the old 200-point grid's resolution"
    )


def test_bisection_returns_none_when_genuinely_out_of_range() -> None:
    mu = GM["MERCURY"]
    r_p = _solve_periapsis_bisection(math.radians(170.0), 5.0, 5.0, mu, r_min=2639.7, r_max=20000.0)
    assert r_p is None, "An unreachable turn angle must return None, not a wrong answer"
