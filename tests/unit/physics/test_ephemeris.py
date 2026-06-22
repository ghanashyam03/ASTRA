from pathlib import Path

import pytest

from astra.state.orbital_state import CelestialBody

SPICE_DIR = Path("data/spice_kernels")
SPICE_AVAILABLE = (SPICE_DIR / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE kernels not downloaded")
def test_earth_position_at_j2000() -> None:
    from astra.physics.ephemeris import EphemerisEngine

    engine = EphemerisEngine(SPICE_DIR)
    engine.load_kernels()
    state = engine.get_body_state(CelestialBody.EARTH, epoch_j2000=0.0)
    r = state.r
    # Earth at J2000 is ~1.00 AU = ~149.6M km
    assert 1.47e8 < r < 1.52e8, f"Earth r = {r:.3e} km, expected ~1.496e8 km"
    engine.unload()


@pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE kernels not downloaded")
def test_earth_mars_separation() -> None:
    from astra.physics.ephemeris import EphemerisEngine

    engine = EphemerisEngine(SPICE_DIR)
    engine.load_kernels()
    earth = engine.get_body_state(CelestialBody.EARTH, epoch_j2000=0.0)
    mars = engine.get_body_state(CelestialBody.MARS, epoch_j2000=0.0)
    import numpy as np

    separation = float(np.linalg.norm(mars.position - earth.position))
    # Earth-Mars separation ranges 54.6M to 401M km
    assert 5e7 < separation < 5e8, f"Earth-Mars separation {separation:.3e} km unexpected"
    engine.unload()
