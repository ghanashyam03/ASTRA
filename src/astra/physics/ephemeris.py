from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import spiceypy as spice

from astra.physics.exceptions import InvalidEphemerisError
from astra.state.orbital_state import CelestialBody, OrbitalState, ReferenceFrame

# J2000 epoch offset: 2000-01-01 12:00:00 TDB in seconds from J2000.0
J2000_EPOCH = 0.0

class TargetType(StrEnum):
    BODY_CENTER = "BODY_CENTER"
    BARYCENTER = "BARYCENTER"

@dataclass
class EphemerisTarget:
    body: CelestialBody
    target_type: TargetType = TargetType.BODY_CENTER

def resolve_central_body(observer: CelestialBody | EphemerisTarget | str) -> CelestialBody:
    """Resolve gravitational central body from an observer to separate
    observational frames from dynamical authorities.
    """
    if isinstance(observer, EphemerisTarget):
        return observer.body
    if isinstance(observer, CelestialBody):
        return observer
    
    obs_upper = observer.upper().strip()
    if "SUN" in obs_upper:
        return CelestialBody.SUN
    if "EARTH" in obs_upper:
        return CelestialBody.EARTH
    if "MOON" in obs_upper:
        return CelestialBody.MOON
    if "MARS" in obs_upper:
        return CelestialBody.MARS
    
    # Safe fallbacks for other standard planets
    for body in CelestialBody:
        if body.value in obs_upper:
            return body
            
    raise InvalidEphemerisError(
        f"Cannot resolve gravitational central body from observer: {observer}"
    )

class EphemerisEngine:
    """Manages SPICE kernels and provides planetary state queries with barycenter safety."""

    # SPICE body name mapping.
    # Solar System Barycenter is ID 0.
    # Celestial body centers (e.g. 399 for Earth, 499 for Mars) vs planetary barycenters
    # (e.g. 3 for Earth Barycenter, 4 for Mars Barycenter).
    # Since DE440 planetary SPK contains planetary barycenters for most planets, queries for
    # planetary body centers (like Mars 499) fail with SPKINSUFFDATA unless a planetary
    # satellite SPK is loaded. Thus, we fall back to planetary barycenters for single-body
    # planets (Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto) in standard
    # DE440, while supporting precise centers for Earth (399) and Moon (301).
    _SPICE_BODY_CENTERS: dict[CelestialBody, str] = {
        CelestialBody.SUN: "SUN",
        CelestialBody.MERCURY: "MERCURY",
        CelestialBody.VENUS: "VENUS",
        CelestialBody.EARTH: "EARTH",
        CelestialBody.MOON: "MOON",
        CelestialBody.MARS: "MARS",
        CelestialBody.JUPITER: "JUPITER",
        CelestialBody.SATURN: "SATURN",
        CelestialBody.URANUS: "URANUS",
        CelestialBody.NEPTUNE: "NEPTUNE",
        CelestialBody.PLUTO: "PLUTO",
    }

    _SPICE_BARYCENTERS: dict[CelestialBody, str] = {
        CelestialBody.SUN: "SUN",
        CelestialBody.MERCURY: "MERCURY BARYCENTER",
        CelestialBody.VENUS: "VENUS BARYCENTER",
        CelestialBody.EARTH: "EARTH BARYCENTER",
        CelestialBody.MOON: "MOON",
        CelestialBody.MARS: "MARS BARYCENTER",
        CelestialBody.JUPITER: "JUPITER BARYCENTER",
        CelestialBody.SATURN: "SATURN BARYCENTER",
        CelestialBody.URANUS: "URANUS BARYCENTER",
        CelestialBody.NEPTUNE: "NEPTUNE BARYCENTER",
        CelestialBody.PLUTO: "PLUTO BARYCENTER",
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

    def _resolve_spice_name(self, target: CelestialBody | EphemerisTarget) -> str:
        if isinstance(target, CelestialBody):
            if target in [CelestialBody.EARTH, CelestialBody.MOON, CelestialBody.SUN]:
                return self._SPICE_BODY_CENTERS[target]
            return self._SPICE_BARYCENTERS[target]

        body = target.body
        if target.target_type == TargetType.BODY_CENTER:
            if body in [CelestialBody.EARTH, CelestialBody.MOON, CelestialBody.SUN]:
                return self._SPICE_BODY_CENTERS[body]
            return self._SPICE_BARYCENTERS[body]
        else:
            return self._SPICE_BARYCENTERS[body]

    def _resolve_observer_name(self, observer: CelestialBody | EphemerisTarget | str) -> str:
        if isinstance(observer, str):
            return observer.upper().strip()
        return self._resolve_spice_name(observer)

    def get_body_state(
        self,
        target: CelestialBody | EphemerisTarget,
        epoch_j2000: float,
        observer: CelestialBody | EphemerisTarget | str = "SUN",
        frame: str = "ECLIPJ2000",
        central_body: CelestialBody | None = None,
    ) -> OrbitalState:
        """Return target state [km, km/s] relative to observer at epoch.
        Enforces float64 precision and separates gravitational authority (central_body)
        from the observer frame.
        """
        self._check_loaded()
        
        target_name = self._resolve_spice_name(target)
        observer_name = self._resolve_observer_name(observer)
        
        try:
            state, _ = spice.spkezr(target_name, epoch_j2000, frame, "NONE", observer_name)
        except Exception as e:
            raise InvalidEphemerisError(
                f"SPICE state query failed for target '{target_name}' "
                f"relative to observer '{observer_name}' in frame '{frame}' "
                f"at epoch {epoch_j2000}: {str(e)}"
            ) from e

        if central_body is None:
            resolved_cb = resolve_central_body(observer)
        else:
            resolved_cb = central_body

        pos = np.asarray(state[:3], dtype=np.float64)
        vel = np.asarray(state[3:], dtype=np.float64)
        
        assert pos.dtype == np.float64, "position must be np.float64"
        assert vel.dtype == np.float64, "velocity must be np.float64"

        try:
            ref_frame = ReferenceFrame(frame.upper().strip())
        except ValueError:
            ref_frame = ReferenceFrame.ECLIPJ2000

        return OrbitalState(
            epoch=epoch_j2000,
            position=pos,
            velocity=vel,
            frame=ref_frame,
            central_body=resolved_cb,
        )

    def get_body_position(
        self, target: CelestialBody | EphemerisTarget, epoch_j2000: float
    ) -> np.ndarray:
        """Shorthand: return [x,y,z] km in ECLIPJ2000."""
        return self.get_body_state(target, epoch_j2000).position

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
