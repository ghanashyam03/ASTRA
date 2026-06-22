"""Deterministic replay manifest for ASTRA optimization runs.
A ReplayManifest captures everything needed to exactly reproduce
an optimization result: mission DSL text, kernel checksums,
random seed, software version, and the optimization parameters.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astra


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ReplayManifest:
    mission_id: str
    mission_yaml: str  # full DSL text
    seed: int
    n_trials: int
    time_limit_seconds: float
    astra_version: str = field(default_factory=lambda: astra.__version__)
    python_version: str = field(default_factory=lambda: platform.python_version())
    kernel_checksums: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> ReplayManifest:
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def verify_kernels(self, kernel_dir: Path) -> bool:
        """Return True if all checksums match current kernel files."""
        for name, expected_sha in self.kernel_checksums.items():
            kf = kernel_dir / name
            if not kf.exists():
                return False
            if _sha256_file(kf) != expected_sha:
                return False
        return True


def build_manifest(
    mission_yaml: str,
    mission_id: str,
    seed: int,
    n_trials: int,
    time_limit_seconds: float,
    kernel_dir: Path = Path("data/spice_kernels"),
) -> ReplayManifest:
    """Build a manifest, computing kernel checksums at build time."""
    checksums: dict[str, str] = {}
    for kname in ["naif0012.tls", "pck00011.tpc", "de440.bsp"]:
        kf = kernel_dir / kname
        if kf.exists():
            checksums[kname] = _sha256_file(kf)
    return ReplayManifest(
        mission_id=mission_id,
        mission_yaml=mission_yaml,
        seed=seed,
        n_trials=n_trials,
        time_limit_seconds=time_limit_seconds,
        kernel_checksums=checksums,
    )
