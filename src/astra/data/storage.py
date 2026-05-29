"""DuckDB-based storage for trajectories and optimization results."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import duckdb

from astra.state.trajectory import Trajectory


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
    ) -> str:
        """Persist trajectory. Returns assigned UUID."""
        tid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO trajectories (
                id, mission_id, created_at, departure_epoch_j2000, tof_days,
                delta_v_total_km_s, duration_days, n_maneuvers, feasible,
                trajectory_json, explanation_json, tags
            ) VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                tid, mission_id,
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
        keys = ["id", "mission_id", "departure_epoch", "tof_days",
                "delta_v_km_s", "duration_days", "feasible", "created_at"]
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
                rid, mission_id,
                result_dict.get("n_evaluations", 0),
                result_dict.get("n_feasible", 0),
                result_dict.get("wall_time_s", 0.0),
                result_dict.get("converged", False),
                best_trajectory_id,
                json.dumps(result_dict),
            ],
        )
        return rid

    def close(self) -> None:
        self.conn.close()
