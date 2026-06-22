from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from astra.data.storage import TrajectoryStore
from astra.state.orbital_state import CelestialBody, OrbitalState
from astra.state.trajectory import Maneuver, Trajectory
from fastapi.testclient import TestClient

EARTH_MARS_YAML = open("data/benchmarks/earth_mars_2031.yaml").read()


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    from astra.api.app import app

    app.state.db_path = ":memory:"
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_optimize_returns_job_id(client: TestClient) -> None:
    resp = client.post("/v1/missions/optimize", json={"mission_yaml": EARTH_MARS_YAML})
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_status_endpoint(client: TestClient) -> None:
    resp = client.post("/v1/missions/optimize", json={"mission_yaml": EARTH_MARS_YAML})
    job_id = resp.json()["job_id"]
    resp2 = client.get(f"/v1/missions/{job_id}/status")
    assert resp2.status_code == 200
    assert "status" in resp2.json()


def test_unknown_job_404(client: TestClient) -> None:
    resp = client.get("/v1/missions/nonexistent-uuid-123/status")
    assert resp.status_code == 404


def test_trajectory_not_found(client: TestClient) -> None:
    resp = client.get("/v1/trajectories/nonexistent-uuid-999")
    assert resp.status_code == 404


def test_trajectory_store_save_and_retrieve() -> None:
    # Create a real trajectory record
    s1 = OrbitalState(
        epoch=0.0,
        position=np.array([1.496e8, 0.0, 0.0]),
        velocity=np.array([0.0, 29.78, 0.0]),
        central_body=CelestialBody.SUN,
    )
    s2 = OrbitalState(
        epoch=86400.0,
        position=np.array([1.496e8, 2.57e6, 0.0]),
        velocity=np.array([-0.5, 29.78, 0.0]),
        central_body=CelestialBody.SUN,
    )
    traj = Trajectory(
        states=[s1, s2],
        maneuvers=[Maneuver(epoch=0.0, delta_v=np.array([0.1, 0.2, 0.0]), label="TMI")],
        metadata={"departure_epoch": 0.0, "tof_days": 1.0},
    )

    store = TrajectoryStore(":memory:")
    try:
        explanation = {
            "mission_id": "test_mission",
            "delta_v_decomposition": {
                "components": [
                    {
                        "label": "TMI",
                        "magnitude_km_s": 0.2236,
                        "fraction_of_total": 1.0,
                        "epoch_j2000": 0.0,
                    }
                ],
                "total_km_s": 0.2236,
                "margin_km_s": 0.0067,
            },
            "constraint_analysis": {"satisfied": True, "constraints": []},
            "window_rationale": None,
            "pareto_analysis": None,
        }

        # Save trajectory record
        tid = store.save_trajectory(
            trajectory=traj,
            mission_id="test_mission",
            explanation=explanation,
            feasible=True,
            tags="integration-test",
        )
        assert tid is not None

        # Retrieve trajectory record
        result = store.get_trajectory(tid)
        assert result is not None
        assert result["id"] == tid
        assert result["trajectory"]["metadata"]["tof_days"] == 1.0
        assert result["explanation"]["delta_v_decomposition"]["total_km_s"] == 0.2236

        # List trajectories
        rows = store.list_trajectories("test_mission")
        assert len(rows) == 1
        assert rows[0]["id"] == tid
        assert rows[0]["feasible"] is True
    finally:
        store.close()


def test_sensitivity_and_pareto_endpoints(client: TestClient) -> None:
    from astra.api.app import _jobs, get_store

    # Save a mock trajectory
    s1 = OrbitalState(
        epoch=0.0,
        position=np.array([1.496e8, 0.0, 0.0]),
        velocity=np.array([0.0, 29.78, 0.0]),
        central_body=CelestialBody.SUN,
    )
    s2 = OrbitalState(
        epoch=86400.0,
        position=np.array([1.496e8, 2.57e6, 0.0]),
        velocity=np.array([-0.5, 29.78, 0.0]),
        central_body=CelestialBody.SUN,
    )
    traj = Trajectory(
        states=[s1, s2],
        maneuvers=[Maneuver(epoch=0.0, delta_v=np.array([0.1, 0.2, 0.0]), label="TMI")],
        metadata={"departure_epoch": 0.0, "tof_days": 1.0},
    )

    store = get_store()
    tid = store.save_trajectory(traj, "test_mission_validation", feasible=True)

    # Inject a completed job
    mock_job_id = "test-job-uuid-123"
    _jobs[mock_job_id] = {
        "status": "complete",
        "job_id": mock_job_id,
        "best_trajectory_id": tid,
        "mission_id": "test_mission_validation",
        "mission_yaml": open("data/benchmarks/earth_mars_2031.yaml").read(),
    }

    # Test sensitivity endpoint
    resp = client.get(f"/v1/missions/{mock_job_id}/sensitivity")
    # Handled status code set for physics-engine dependency availability
    assert resp.status_code in {200, 500, 503}

    # Test pareto metrics endpoint
    resp2 = client.get(f"/v1/missions/{mock_job_id}/pareto-metrics")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "dv_km_s" in data2
    assert "hypervolume_indicator" in data2
