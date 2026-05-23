"""SPICE-based ephemeris engine. Loads kernels once and provides
body states throughout the mission design epoch range."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import spiceypy as spice

from astra.state.orbital_state import CelestialBody, OrbitalState, ReferenceFrame

# J2000 epoch offset: 2000-01-01 12:00:00 TDB in seconds from J2000.0
J2000_EPOCH = 0.0

class EphemerisEngine:
    """Manages SPICE kernels and provides planetary state queries."""

    # SPICE body name map
    _SPICE_NAMES: dict[str, str] = {
        "SUN": "SUN", "MERCURY": "MERCURY BARYCENTER", "VENUS": "VENUS BARYCENTER",
        "EARTH": "EARTH", "MOON": "MOON", "MARS": "MARS BARYCENTER",
        "JUPITER": "JUPITER BARYCENTER", "SATURN": "SATURN BARYCENTER",
        "URANUS": "URANUS BARYCENTER", "NEPTUNE": "NEPTUNE BARYCENTER",
        "PLUTO": "PLUTO BARYCENTER",
    }

    def __init__(self, kernel_dir: Path) -> None:
        self.kernel_dir = kernel_dir
        self._loaded = False

    def load_kernels(self) -> None:
        """Load LSK, PCK, and SPK kernels from kernel_dir."""
        spice.kclear()
        lsk = self.kernel_dir / "naif0012.tls"
        pck = self.kernel_dir / "pck00011.tpc"
        spk = self.kernel_dir / "de440.bsp"
        for kf in [lsk, pck, spk]:
            if not kf.exists():
                raise FileNotFoundError(
                    f"SPICE kernel not found: {kf}. "
                    f"Run: uv run python scripts/download_kernels.py"
                )
            spice.furnsh(str(kf))
        self._loaded = True

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call load_kernels() before querying states.")

    def get_body_state(
        self,
        body: CelestialBody,
        epoch_j2000: float,
        observer: str = "SUN",
        frame: str = "ECLIPJ2000",
    ) -> OrbitalState:
        """Return body state [km, km/s] relative to observer at epoch."""
        self._check_loaded()
        spice_name = self._SPICE_NAMES[body.value]
        state, _ = spice.spkezr(spice_name, epoch_j2000, frame, "NONE", observer)
        return OrbitalState(
            epoch=epoch_j2000,
            position=np.array(state[:3]),
            velocity=np.array(state[3:]),
            frame=ReferenceFrame.ECLIPJ2000,
            central_body=CelestialBody.SUN if observer == "SUN" else CelestialBody.EARTH,
        )

    def get_body_position(
        self, body: CelestialBody, epoch_j2000: float
    ) -> np.ndarray:
        """Shorthand: return [x,y,z] km in ECLIPJ2000."""
        return self.get_body_state(body, epoch_j2000).position

    def epoch_from_date(self, iso_date: str) -> float:
        """Convert ISO 8601 date string to J2000 seconds via SPICE."""
        self._check_loaded()
        return float(spice.str2et(iso_date))

    def date_from_epoch(self, epoch_j2000: float) -> str:
        """Convert J2000 seconds to ISO date string."""
        self._check_loaded()
        return str(spice.et2utc(epoch_j2000, "ISOC", 3))

    def unload(self) -> None:
        spice.kclear()
        self._loaded = False
