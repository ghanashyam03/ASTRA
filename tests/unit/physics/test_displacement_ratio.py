from pathlib import Path

import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
@pytest.mark.parametrize(
    "body,v_inf,expected_ratio_approx",
    [
        ("MERCURY", 5.0, 19.2),
        ("VENUS", 9.7, 7.2),
        ("EARTH", 9.0, 6.6),
        ("MARS", 5.5, 8.8),
        ("JUPITER", 10.0, 2.6),
        ("SATURN", 8.0, 2.4),
        ("URANUS", 9.0, 1.5),
        ("NEPTUNE", 10.0, 1.1),
    ],
)
def test_displacement_ratio_matches_hand_derivation(
    body: str, v_inf: float, expected_ratio_approx: float
) -> None:
    from astra.physics.kernel import PhysicsKernel
    from astra.physics.soi_passage_estimate import soi_crossing_displacement_ratio

    kernel = PhysicsKernel().load()
    epoch = kernel.epoch_from_date("2030-01-01T00:00:00")
    result = soi_crossing_displacement_ratio(body, v_inf, epoch, kernel)
    print(f"\n{body}: ratio={result.ratio:.2f} (hand-derived approx: {expected_ratio_approx})")
    assert abs(result.ratio - expected_ratio_approx) / expected_ratio_approx < 0.5, (
        f"{body}: computed ratio {result.ratio:.2f} differs from hand-derived "
        f"{expected_ratio_approx} by more than 50% — investigate before "
        f"trusting this result (real ephemeris velocity differs somewhat "
        f"from the circular-orbit approximation used by hand, but not by "
        f"this much)"
    )


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_ratio_trend_decreases_outward() -> None:
    """THE critical sanity check: inner planets must show LARGER ratios than
    outer planets. If this fails, there is a sign/scaling bug — do not trust
    any conclusion drawn from this prompt's findings until it passes."""
    from astra.physics.kernel import PhysicsKernel
    from astra.physics.soi_passage_estimate import soi_crossing_displacement_ratio

    kernel = PhysicsKernel().load()
    epoch = kernel.epoch_from_date("2030-01-01T00:00:00")
    r_mercury = soi_crossing_displacement_ratio("MERCURY", 5.0, epoch, kernel).ratio
    r_neptune = soi_crossing_displacement_ratio("NEPTUNE", 10.0, epoch, kernel).ratio
    assert r_mercury > r_neptune, (
        f"Expected Mercury's ratio ({r_mercury:.2f}) to exceed Neptune's "
        f"({r_neptune:.2f}) — the trend must decrease outward; if it doesn't, "
        f"there is a bug in the SOI radius or velocity computation"
    )
