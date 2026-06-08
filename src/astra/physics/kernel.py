"""Main physics kernel — unified interface to all physics computations.
Coordinates orbital propagation, ephemeris queries, and Lambert boundary value solvers.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from astra.physics.ephemeris import EphemerisEngine, EphemerisTarget
from astra.physics.lambert import lambert_izzo
from astra.physics.propagator import Integrator, propagate_two_body
from astra.state.orbital_state import CelestialBody, OrbitalState

if TYPE_CHECKING:
    from astra.data.cache import EphemerisCache
    from astra.neural.surrogate import NeuralSurrogate
    from astra.state.trajectory import Trajectory, TrajectoryValidationResult

class PhysicsKernel:
    """Unified physics interface. Owns the ephemeris engine and
    provides all trajectory computation primitives."""

    def __init__(
        self,
        kernel_dir: Path | str = "data/spice_kernels",
        cache: EphemerisCache | None = None,
    ) -> None:
        from astra.data.cache import EphemerisCache as _Cache
        if cache is None:
            cache = _Cache(max_entries=50_000)
        self.ephemeris = EphemerisEngine(
            Path(kernel_dir),
            cache=cache,
        )
        self._kernels_loaded = False

    def load(self) -> PhysicsKernel:
        """Load SPICE kernels. Call once at startup."""
        self.ephemeris.load_kernels()
        self._kernels_loaded = True
        return self

    def get_body_state(
        self,
        target: CelestialBody | EphemerisTarget,
        epoch_j2000: float,
        observer: CelestialBody | EphemerisTarget | str = "SUN",
        frame: str = "ECLIPJ2000",
        central_body: CelestialBody | None = None,
    ) -> OrbitalState:
        """Query state vectors relative to an observer using precise ephemerides."""
        return self.ephemeris.get_body_state(
            target, epoch_j2000, observer=observer, frame=frame, central_body=central_body
        )

    def lambert_solve(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        tof_seconds: float,
        mu: float,
        retrograde: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Solve Lambert BVP using Householder/Halley iterative formulation.
        
        Raises LambertSingularityError or LambertConvergenceError on failures.
        """
        return lambert_izzo(r1, r2, tof_seconds, mu, retrograde)

    def propagate(
        self,
        state: OrbitalState,
        dt_seconds: float,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        integrator: Integrator | None = None,
    ) -> OrbitalState:
        """Numerically propagate Keplerian state using configurable integrators and
        collision safety checks.
        """
        return propagate_two_body(
            state, dt_seconds, rtol=rtol, atol=atol, integrator=integrator
        )

    def compute_delta_v(
        self, state: OrbitalState, target_velocity: np.ndarray
    ) -> float:
        """Magnitude of velocity change needed [km/s]."""
        return float(np.linalg.norm(target_velocity - state.velocity))

    def epoch_from_date(self, iso_date: str) -> float:
        """Convert ISO date string to J2000 seconds via SPICE."""
        return self.ephemeris.epoch_from_date(iso_date)

    def date_from_epoch(self, epoch: float) -> str:
        """Convert J2000 seconds to ISO date string via SPICE."""
        return self.ephemeris.date_from_epoch(epoch)

    def validate_trajectory(
        self,
        trajectory: Trajectory,
        surrogate: NeuralSurrogate | None = None,
        pos_tol_km: float = 5000.0,
        dv_tol_kms: float = 0.1,
    ) -> TrajectoryValidationResult:
        """Perform multi-stage validation of a candidate trajectory.

        Stage 1: Compare with surrogate estimate (if surrogate provided).
        Stage 2: Perform exact Lambert solve between endpoints.
        Stage 3: Numerically propagate exact departure state to check arrival position deviation.
        """
        from astra.state.trajectory import TrajectoryValidationResult

        if len(trajectory.states) < 2:
            return TrajectoryValidationResult(
                is_valid=False,
                dv_diff=99.0,
                pos_diff=999999.0,
                diagnostics={"error": "Trajectory has fewer than 2 states"},
            )

        s_dep = trajectory.states[0]
        s_arr = trajectory.states[-1]
        dep_epoch = s_dep.epoch
        arr_epoch = s_arr.epoch
        tof_seconds = arr_epoch - dep_epoch

        # Find closest planets to endpoints
        def find_closest(position: np.ndarray, epoch: float) -> CelestialBody | None:
            min_dist = float("inf")
            closest = None
            for body in CelestialBody:
                if body == CelestialBody.SUN:
                    continue
                try:
                    st = self.get_body_state(body, epoch)
                    dist = float(np.linalg.norm(position - st.position))
                    if dist < min_dist:
                        min_dist = dist
                        closest = body
                except Exception:
                    pass
            return closest

        body_from = find_closest(s_dep.position, dep_epoch)
        body_to = find_closest(s_arr.position, arr_epoch)

        if body_from is None or body_to is None:
            return TrajectoryValidationResult(
                is_valid=False,
                dv_diff=99.0,
                pos_diff=999999.0,
                diagnostics={"error": "Could not identify departure or arrival celestial body"},
            )

        r1_state = self.get_body_state(body_from, dep_epoch)
        r2_state = self.get_body_state(body_to, arr_epoch)

        # Stage 2: Exact Lambert solve
        from astra.physics.lambert import find_best_transfer
        try:
            sol = find_best_transfer(
                r1=r1_state.position,
                v1_body=r1_state.velocity,
                r2=r2_state.position,
                v2_body=r2_state.velocity,
                tof=tof_seconds,
                mu=self.ephemeris.mu_sun if hasattr(self.ephemeris, "mu_sun") else 1.32712440018e11,
                max_revs=0,
            )
            exact_v_dep = sol.v1
            exact_v_arr = sol.v2
            lambert_success = True
        except Exception:
            lambert_success = False
            exact_v_dep = np.zeros(3)
            exact_v_arr = np.zeros(3)

        if not lambert_success:
            return TrajectoryValidationResult(
                is_valid=False,
                dv_diff=99.0,
                pos_diff=999999.0,
                diagnostics={"error": "Exact Lambert solve failed"},
            )

        # Compute expected patched-conics delta-v
        v_inf_dep = exact_v_dep - r1_state.velocity
        v_inf_arr = r2_state.velocity - exact_v_arr

        from astra.physics.maneuvers import arrival_delta_v, departure_delta_v
        h_park = trajectory.metadata.get("parking_altitude_km", 200.0)
        h_cap = trajectory.metadata.get("capture_altitude_km", 300.0)
        capture_apoapsis_km = trajectory.metadata.get("capture_apoapsis_km", None)

        dv1_mag = departure_delta_v(v_inf_dep, h_park, body_from.name)
        dv2_mag = arrival_delta_v(v_inf_arr, h_cap, body_to.name, apoapsis_km=capture_apoapsis_km)
        expected_patched_dv = dv1_mag + dv2_mag

        # Calculate delta-v difference
        dv_diff = abs(trajectory.delta_v_total - expected_patched_dv)

        # Stage 3: Patched-conic verification (Numerical two-body propagation)
        transfer_state = OrbitalState(
            epoch=dep_epoch,
            position=r1_state.position.copy(),
            velocity=exact_v_dep.copy(),
            central_body=CelestialBody.SUN,
        )
        propagated_state = self.propagate(transfer_state, tof_seconds)
        pos_diff = float(np.linalg.norm(propagated_state.position - r2_state.position))

        # Stage 1: Compare with surrogate if provided
        surr_dv = None
        surr_dv_diff = 0.0
        if surrogate is not None:
            try:
                from astra.explainability.window_rationale import compute_synodic_period
                from astra.neural.features import build_geometric_features

                syn_days = compute_synodic_period(body_from, body_to)
                synodic_period_s = syn_days * 86400.0 if syn_days != float("inf") else 0.0

                dep_min = trajectory.metadata.get(
                    "departure_epoch_start", dep_epoch - 365.0 * 86400.0
                )
                dep_max = trajectory.metadata.get(
                    "departure_epoch_end", dep_epoch + 365.0 * 86400.0
                )
                tof_min = trajectory.metadata.get(
                    "tof_min_seconds", max(1.0, tof_seconds - 100.0 * 86400.0)
                )
                tof_max = trajectory.metadata.get(
                    "tof_max_seconds", tof_seconds + 100.0 * 86400.0
                )

                feat = build_geometric_features(
                    dep_epoch=dep_epoch,
                    tof_seconds=tof_seconds,
                    r1_km=r1_state.position,
                    v1_km_s=r1_state.velocity,
                    r2_km=r2_state.position,
                    dep_epoch_min=dep_min,
                    dep_epoch_max=dep_max,
                    tof_min=tof_min,
                    tof_max=tof_max,
                    synodic_period_s=synodic_period_s,
                )
                pred_obj = surrogate.predict(
                    feat,
                    v_planet_depart=r1_state.velocity,
                    v_planet_arrive=r2_state.velocity,
                )
                surr_dv = pred_obj.prediction
                surr_dv_diff = abs(surr_dv - expected_patched_dv)
            except Exception:
                pass

        is_valid = (pos_diff <= pos_tol_km) and (dv_diff <= dv_tol_kms)

        diagnostics = {
            "origin_body": body_from.name,
            "destination_body": body_to.name,
            "departure_epoch": dep_epoch,
            "arrival_epoch": arr_epoch,
            "tof_days": tof_seconds / 86400.0,
            "lambert_success": lambert_success,
            "expected_patched_dv": expected_patched_dv,
            "trajectory_recorded_dv": trajectory.delta_v_total,
            "surrogate_dv": surr_dv,
            "surrogate_dv_diff": surr_dv_diff,
        }

        return TrajectoryValidationResult(
            is_valid=is_valid,
            dv_diff=dv_diff,
            pos_diff=pos_diff,
            diagnostics=diagnostics,
        )
