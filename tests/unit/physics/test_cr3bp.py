"""Unit tests for the CR3BP interface stubs."""

from __future__ import annotations

import numpy as np
import pytest

from astra.physics.cr3bp import (
    CR3BP_SYSTEMS,
    CR3BPSystem,
    cr3bp_eom,
    dimensionalize_state,
    nondimensionalize_state,
    propagate_cr3bp,
)


def test_cr3bp_system_constants() -> None:
    """Verify that CR3BP_SYSTEMS exists and is correctly populated."""
    assert len(CR3BP_SYSTEMS) == 8
    for body in [
        "MERCURY",
        "VENUS",
        "EARTH",
        "MARS",
        "JUPITER",
        "SATURN",
        "URANUS",
        "NEPTUNE",
    ]:
        assert body in CR3BP_SYSTEMS
        system = CR3BP_SYSTEMS[body]
        assert isinstance(system, CR3BPSystem)
        assert system.body == body
        assert system.mu_star > 0
        assert system.L_star_km > 0
        assert system.n_rad_per_s > 0
        assert system.T_star_s > 0

    # Specifically verify Venus mu_star is within 1% of 2.4478e-6
    venus_mu = CR3BP_SYSTEMS["VENUS"].mu_star
    assert abs(venus_mu - 2.4478e-6) / 2.4478e-6 < 0.01


def test_cr3bp_stubs_raise_not_implemented() -> None:
    """Verify that all callable functions raise NotImplementedError("Implemented in P53")."""
    q_dummy = np.zeros(6)
    r_dummy = np.zeros(3)
    v_dummy = np.zeros(3)
    system = CR3BP_SYSTEMS["VENUS"]

    with pytest.raises(NotImplementedError) as exc_info:
        cr3bp_eom(0.0, q_dummy, system.mu_star)
    assert str(exc_info.value) == "Implemented in P53"

    with pytest.raises(NotImplementedError) as exc_info:
        propagate_cr3bp(system, q_dummy, (0.0, 1.0))
    assert str(exc_info.value) == "Implemented in P53"

    with pytest.raises(NotImplementedError) as exc_info:
        nondimensionalize_state(r_dummy, v_dummy, 0.0, system, r_dummy)
    assert str(exc_info.value) == "Implemented in P53"

    with pytest.raises(NotImplementedError) as exc_info:
        dimensionalize_state(q_dummy, 0.0, system, r_dummy)
    assert str(exc_info.value) == "Implemented in P53"
