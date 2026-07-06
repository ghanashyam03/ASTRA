"""Validation tests for the CR3BP propagator implementation."""

from __future__ import annotations

import math

import numpy as np

from astra.physics.cr3bp import (
    CR3BP_SYSTEMS,
    cr3bp_eom,
    jacobi_constant,
    lagrange_l1_x,
    lagrange_l2_x,
    propagate_cr3bp,
)


def test_cr3bp_systems_populated() -> None:
    """All 8 planets must have pre-computed CR3BP systems."""
    for body in ["MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"]:
        assert body in CR3BP_SYSTEMS
        s = CR3BP_SYSTEMS[body]
        assert 0.0 < s.mu_star < 1.0
        assert s.L_star_km > 0.0
        assert s.n_rad_per_s > 0.0


def test_venus_mu_star_value() -> None:
    """Venus μ* must match DE440 masses to within 1%."""
    mu = CR3BP_SYSTEMS["VENUS"].mu_star
    expected = 3.24859e5 / (1.32712440018e11 + 3.24859e5)
    assert abs(mu - expected) / expected < 0.01


def test_jacobi_constant_conserved() -> None:
    """Jacobi constant must be conserved along propagated trajectory to integrator precision."""
    system = CR3BP_SYSTEMS["EARTH"]
    mu = system.mu_star
    # Initial state: displaced from L1 (towards the Sun to avoid Earth collision)
    x_L1 = lagrange_l1_x(mu)
    q0 = np.array([x_L1 - 0.01, 0.0, 0.0, 0.0, 0.01, 0.0])
    C0 = jacobi_constant(q0, mu)
    t, q = propagate_cr3bp(system, q0, (0.0, 10.0), rtol=1e-12, atol=1e-12)
    C_final = jacobi_constant(q[:, -1], mu)
    relative_error = abs(C_final - C0) / abs(C0)
    assert relative_error < 1e-8, (
        f"Jacobi constant not conserved: relative error = {relative_error:.2e}"
    )


def test_lagrange_l4_l5_positions() -> None:
    """L4 and L5 must be at (1/2 − μ*, ±√3/2, 0) to within 1e-10."""
    for body in ["EARTH", "JUPITER"]:
        mu = CR3BP_SYSTEMS[body].mu_star
        x_L4 = 0.5 - mu
        y_L4 = math.sqrt(3.0) / 2.0
        # Verify these are equilibrium points: acceleration at L4/L5 should be zero
        q_L4 = np.array([x_L4, y_L4, 0.0, 0.0, 0.0, 0.0])
        dq = cr3bp_eom(0.0, q_L4, mu)
        # accelerations (indices 3,4,5) should vanish at equilibrium
        acc_norm = float(np.linalg.norm(dq[3:]))
        assert acc_norm < 1e-10, f"L4 not equilibrium for {body}: |acc| = {acc_norm:.2e}"


def test_lagrange_l1_l2_approximate_positions() -> None:
    """L1 and L2 x-coordinates must be close to known values from literature."""
    # For Earth-Sun system, L1 is ~1.5e6 km from Earth (non-dim ≈ 0.990)
    mu = CR3BP_SYSTEMS["EARTH"].mu_star
    x_l1 = lagrange_l1_x(mu)
    x_l2 = lagrange_l2_x(mu)
    assert 0.98 < x_l1 < 0.995, f"L1 x={x_l1:.6f} outside expected range"
    assert 1.005 < x_l2 < 1.015, f"L2 x={x_l2:.6f} outside expected range"


def test_eom_produces_correct_shape() -> None:
    """cr3bp_eom must return a (6,) array."""
    q = np.array([0.9, 0.1, 0.0, 0.0, 0.1, 0.0])
    result = cr3bp_eom(0.0, q, CR3BP_SYSTEMS["VENUS"].mu_star)
    assert result.shape == (6,), f"Expected shape (6,), got {result.shape}"
