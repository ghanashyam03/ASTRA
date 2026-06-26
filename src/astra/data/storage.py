"""DuckDB-based storage for trajectories and optimization results."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np

from astra.optimization.pareto import hypervolume_indicator_2d, pareto_spread
from astra.state.trajectory import Trajectory

if TYPE_CHECKING:
    from astra.physics.kernel import PhysicsKernel


class TrajectoryStore:
    """Stores computed trajectories in DuckDB for querying and replay."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS trajectories (
        id VARCHAR PRIMARY KEY,
        mission_id VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        departure_epoch_j2000 DOUBLE,
        tof_days DOUBLE,
        delta_v_total_km_s DOUBLE,
        duration_days DOUBLE,
        n_maneuvers INTEGER,
        feasible BOOLEAN,
        trajectory_json VARCHAR,
        explanation_json VARCHAR,
        tags VARCHAR DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS optimization_runs (
        id VARCHAR PRIMARY KEY,
        mission_id VARCHAR NOT NULL,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        n_evaluations INTEGER,
        n_feasible INTEGER,
        wall_time_s DOUBLE,
        converged BOOLEAN,
        best_trajectory_id VARCHAR,
        result_json VARCHAR
    );
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)
        self.conn.execute(self.SCHEMA)

    def save_trajectory(
        self,
        trajectory: Trajectory,
        mission_id: str,
        explanation: dict[str, Any] | None = None,
        feasible: bool = True,
        tags: str = "",
        audit_before_save: bool = False,
        kernel: PhysicsKernel | None = None,
    ) -> str:
        """Persist trajectory. Returns assigned UUID.

        If audit_before_save is True and a kernel is provided, an independent
        physics audit will be run on the trajectory prior to persistence.
        """
        if audit_before_save and kernel is not None:
            from astra.optimization.audit import audit_trajectory_physics

            audit_trajectory_physics(trajectory, kernel)

        tid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO trajectories (
                id, mission_id, created_at, departure_epoch_j2000, tof_days,
                delta_v_total_km_s, duration_days, n_maneuvers, feasible,
                trajectory_json, explanation_json, tags
            ) VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                tid,
                mission_id,
                trajectory.departure_epoch,
                trajectory.duration_days,
                trajectory.delta_v_total,
                trajectory.duration_days,
                len(trajectory.maneuvers),
                feasible,
                json.dumps(trajectory.to_dict()),
                json.dumps(explanation) if explanation else None,
                tags,
            ],
        )
        return tid

    def get_trajectory(self, tid: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT trajectory_json, explanation_json FROM trajectories WHERE id = ?",
            [tid],
        ).fetchone()
        if row is None:
            return None
        return {
            "id": tid,
            "trajectory": json.loads(row[0]),
            "explanation": json.loads(row[1]) if row[1] else None,
        }

    def list_trajectories(self, mission_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT id, mission_id, departure_epoch_j2000, tof_days,
               delta_v_total_km_s, duration_days, feasible, created_at
               FROM trajectories WHERE mission_id = ?
               ORDER BY delta_v_total_km_s ASC LIMIT ?""",
            [mission_id, limit],
        ).fetchall()
        keys = [
            "id",
            "mission_id",
            "departure_epoch",
            "tof_days",
            "delta_v_km_s",
            "duration_days",
            "feasible",
            "created_at",
        ]
        return [dict(zip(keys, r)) for r in rows]

    def save_optimization_run(
        self,
        mission_id: str,
        result_dict: dict[str, Any],
        best_trajectory_id: str | None = None,
    ) -> str:
        rid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO optimization_runs (
                id, mission_id, started_at, completed_at, n_evaluations,
                n_feasible, wall_time_s, converged, best_trajectory_id, result_json
            ) VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)""",
            [
                rid,
                mission_id,
                result_dict.get("n_evaluations", 0),
                result_dict.get("n_feasible", 0),
                result_dict.get("wall_time_s", 0.0),
                result_dict.get("converged", False),
                best_trajectory_id,
                json.dumps(result_dict),
            ],
        )
        return rid

    def query_best_per_mission(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Return top-N best Δv trajectory per mission_id."""
        rows = self.conn.execute(
            """
            WITH ranked AS (
                SELECT
                    mission_id,
                    id,
                    delta_v_total_km_s,
                    duration_days,
                    created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY mission_id
                        ORDER BY delta_v_total_km_s ASC
                    ) AS rn
                FROM trajectories
                WHERE feasible = true
            )
            SELECT mission_id, id, delta_v_total_km_s, duration_days, created_at
            FROM ranked
            WHERE rn = 1
            ORDER BY delta_v_total_km_s ASC
            LIMIT ?
            """,
            [top_n],
        ).fetchall()
        keys = ["mission_id", "id", "delta_v_total_km_s", "duration_days", "created_at"]
        return [dict(zip(keys, r)) for r in rows]

    def compute_pareto_metrics(self, mission_id: str) -> dict[str, Any]:
        """Compute Pareto front quality metrics for a mission from stored data."""

        rows = self.conn.execute(
            """SELECT delta_v_total_km_s, duration_days FROM trajectories
               WHERE mission_id = ? AND feasible = true
               ORDER BY delta_v_total_km_s ASC""",
            [mission_id],
        ).fetchall()
        if len(rows) < 2:
            return {"error": "insufficient data"}
        pts = np.array(rows)
        ref = pts.max(axis=0) * 1.1
        return {
            "n_trajectories": len(rows),
            "hypervolume_indicator": round(hypervolume_indicator_2d(pts, ref), 4),
            "pareto_spread": round(pareto_spread(pts), 4),
            "best_dv_km_s": round(float(pts[:, 0].min()), 4),
            "best_tof_days": round(float(pts[:, 1].min()), 2),
        }

    def get_pareto_metrics(self, mission_id: str) -> dict[str, Any]:
        """Compute Pareto quality metrics for all stored trajectories of a mission."""
        rows = self.conn.execute(
            """SELECT delta_v_total_km_s, duration_days FROM trajectories
               WHERE mission_id = ? AND feasible = true
               ORDER BY delta_v_total_km_s ASC""",
            [mission_id],
        ).fetchall()
        if len(rows) < 2:
            return {"error": "insufficient_data", "n_trajectories": len(rows)}
        pts = np.array(rows, dtype=float)
        ref = pts.max(axis=0) * 1.1
        return {
            "mission_id": mission_id,
            "n_trajectories": len(rows),
            "hypervolume_indicator": round(hypervolume_indicator_2d(pts, ref), 6),
            "spread": round(pareto_spread(pts), 6),
            "best_dv_km_s": round(float(pts[:, 0].min()), 4),
            "best_tof_days": round(float(pts[:, 1].min()), 2),
            "worst_dv_km_s": round(float(pts[:, 0].max()), 4),
        }

    def get_best_by_mission(self) -> list[dict[str, Any]]:
        """Return the single best (lowest Δv) trajectory per mission_id."""
        rows = self.conn.execute(
            """SELECT DISTINCT ON (mission_id) mission_id, id,
                   delta_v_total_km_s, duration_days, created_at
               FROM trajectories
               WHERE feasible = true
               ORDER BY mission_id, delta_v_total_km_s ASC"""
        ).fetchall()
        keys = ["mission_id", "trajectory_id", "delta_v_km_s", "duration_days", "created_at"]
        return [dict(zip(keys, r)) for r in rows]

    def close(self) -> None:
        self.conn.close()
