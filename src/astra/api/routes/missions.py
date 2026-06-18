"""Missions routes for the ASTRA API.

Contains:
- POST /v1/missions/optimize: Submit a mission optimization job.
- GET /v1/missions/{job_id}/status: Check optimization job status.
- GET /v1/missions/{job_id}/result: Get completed optimization job results.
- GET /v1/missions/{id}/sensitivity: Run sensitivity analysis on optimized trajectory.
- GET /v1/missions/{id}/pareto-metrics: Get Pareto front metrics for optimization run.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException

from astra.api.dependencies import _jobs, get_kernel, get_store
from astra.api.schemas.requests import OptimizeRequest
from astra.api.schemas.responses import (
    JobStatusResponse,
    JobSubmittedResponse,
    MissionSummaryResponse,
)
from astra.dsl.compiler import compile_mission
from astra.dsl.parser import parse_mission_string
from astra.explainability.engine import explain
from astra.optimization.engine import optimize_mission_bayesian
from astra.state.orbital_state import CelestialBody, OrbitalState
from astra.state.trajectory import Maneuver, Trajectory
from astra.visualization.sensitivity import analyze_sensitivity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["missions"])


def _run_optimization(job_id: str, mission_yaml: str) -> None:
    """Background task: parse, compile, optimize, save."""
    store = get_store()
    kernel = get_kernel()
    try:
        _jobs[job_id]["status"] = "running"
        dsl = parse_mission_string(mission_yaml)
        mission = compile_mission(
            dsl,
            kernel.ephemeris if kernel._kernels_loaded else None
        )
        result = optimize_mission_bayesian(
            mission, kernel,
            n_trials=mission.max_evaluations,
            time_limit=300.0,
            seed=mission.seed,
        )
        result_dict = result.to_dict()

        best_tid = None
        if result.best_trajectory:
            trace = explain(
                result.best_trajectory, mission,
                pareto_front=result.pareto_front,
                ephemeris=kernel.ephemeris if kernel._kernels_loaded else None,
            )
            best_tid = store.save_trajectory(
                result.best_trajectory, mission.mission_id,
                explanation=trace.to_dict(),
            )
            for traj in result.pareto_front[:20]:  # store top 20 Pareto
                store.save_trajectory(traj, mission.mission_id, feasible=True)

        run_id = store.save_optimization_run(mission.mission_id, result_dict, best_tid)
        _jobs[job_id].update({
            "status": "complete",
            "result": result_dict,
            "best_trajectory_id": best_tid,
            "run_id": run_id,
            "mission_id": mission.mission_id,
            "mission_yaml": mission_yaml,
        })
    except Exception as e:
        logger.exception(f"Optimization job {job_id} failed")
        _jobs[job_id].update({"status": "failed", "error": str(e)})


@router.post("/v1/missions/optimize", status_code=202, response_model=JobSubmittedResponse)
async def optimize_mission(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
) -> JobSubmittedResponse:
    """Submit a mission optimization job to run in the background."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "job_id": job_id, "mission_yaml": request.mission_yaml}
    background_tasks.add_task(_run_optimization, job_id, request.mission_yaml)
    return JobSubmittedResponse(
        job_id=job_id,
        status="queued",
        poll_url=f"/v1/missions/{job_id}/status"
    )


@router.get("/v1/missions/{job_id}/status", response_model=JobStatusResponse)
async def mission_status(job_id: str) -> JobStatusResponse:
    """Check the status of a background mission optimization job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(job_id=job_id, status=job["status"])


@router.get("/v1/missions/{job_id}/result")
async def mission_result(job_id: str) -> dict[str, Any]:
    """Retrieve the output of a completed optimization job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Job not complete. Status: {job['status']}"
        )
    return job


@router.get("/v1/missions/{job_id}/sensitivity")
async def mission_sensitivity(job_id: str) -> dict[str, Any]:
    """Compute departure epoch and TOF sensitivity for the best trajectory of a job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")
    best_tid = job.get("best_trajectory_id")
    if not best_tid:
        raise HTTPException(status_code=404, detail="No trajectory stored for this job")
    
    kernel = get_kernel()
    store = get_store()
    row = store.get_trajectory(best_tid)
    if row is None:
        raise HTTPException(status_code=404, detail="Trajectory not found in store")
    
    td = row["trajectory"]
    dep_epoch = td["departure_epoch_j2000"]
    arr_epoch = td["arrival_epoch_j2000"]
    maneuvers = [
        Maneuver(
            epoch=m["epoch"],
            delta_v=np.array(m["dv_km_s"]),
            label=m["label"],
        )
        for m in td["maneuvers"]
    ]
    states = [
        OrbitalState(epoch=dep_epoch, position=np.zeros(3), velocity=np.zeros(3),
                     central_body=CelestialBody.SUN),
        OrbitalState(epoch=arr_epoch, position=np.zeros(3), velocity=np.zeros(3),
                     central_body=CelestialBody.SUN),
    ]
    traj = Trajectory(states=states, maneuvers=maneuvers, metadata=td.get("metadata", {}))
    
    mission_yaml = job.get("mission_yaml", "")
    if not mission_yaml:
        raise HTTPException(status_code=422, detail="Mission YAML not stored with job")
    dsl = parse_mission_string(mission_yaml)
    mission = compile_mission(dsl, kernel.ephemeris if kernel._kernels_loaded else None)
    
    result = analyze_sensitivity(traj, mission, kernel)
    return result.to_dict()

@router.get("/v1/missions/{job_id}/pareto-metrics")
async def pareto_metrics(job_id: str) -> dict[str, Any]:
    """Return Pareto quality metrics for all stored trajectories from a job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    mission_id = job.get("mission_id")
    if not mission_id:
        raise HTTPException(status_code=422, detail="No mission_id in job")
    store = get_store()
    metrics = store.get_pareto_metrics(mission_id)
    
    rows = store.conn.execute(
        """SELECT delta_v_total_km_s, duration_days FROM trajectories
           WHERE mission_id = ? AND feasible = true
           ORDER BY delta_v_total_km_s ASC""",
        [mission_id],
    ).fetchall()
    dvs = [round(float(r[0]), 4) for r in rows]
    days = [round(float(r[1]), 2) for r in rows]
    
    if "error" in metrics:
        return {
            **metrics,
            "dv_km_s": dvs,
            "tof_days": days,
            "hypervolume_indicator": 0.0,
        }
    return {
        **metrics,
        "dv_km_s": dvs,
        "tof_days": days,
    }


@router.get("/v1/missions/{job_id}/summary", response_model=MissionSummaryResponse)
async def mission_summary(job_id: str) -> dict[str, Any]:
    """Return a high-level MissionSummary for a completed job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")

    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_string
    from astra.state.mission import mission_summary_from_result

    result_dict = job.get("result", {})
    mission_yaml = job.get("mission_yaml", "")
    if not mission_yaml:
        raise HTTPException(status_code=422, detail="Mission YAML not available")

    kernel = get_kernel()
    dsl = parse_mission_string(mission_yaml)
    mission = compile_mission(dsl, kernel.ephemeris if kernel._kernels_loaded else None)

    from astra.optimization.engine import OptimizationResult

    # Reconstruct a minimal OptimizationResult for summary
    res_obj = OptimizationResult(
        best_trajectory=None,
        n_evaluations=result_dict.get("n_evaluations", 0),
        n_feasible=result_dict.get("n_feasible", 0),
        wall_time_s=result_dict.get("wall_time_s", 0.0),
    )

    best_tid = job.get("best_trajectory_id")
    if best_tid:
        store = get_store()
        row = store.get_trajectory(best_tid)
        if row:
            # Reconstruct Trajectory from stored JSON for summary
            import numpy as np

            from astra.state.orbital_state import CelestialBody, OrbitalState
            from astra.state.trajectory import Maneuver, Trajectory
            td = row["trajectory"]
            maneuvers = [
                Maneuver(epoch=m["epoch"], delta_v=np.array(m["dv_km_s"]), label=m["label"])
                for m in td["maneuvers"]
            ]
            states = [
                OrbitalState(epoch=td["departure_epoch_j2000"],
                             position=np.zeros(3), velocity=np.zeros(3),
                             central_body=CelestialBody.SUN),
                OrbitalState(epoch=td["arrival_epoch_j2000"],
                             position=np.zeros(3), velocity=np.zeros(3),
                             central_body=CelestialBody.SUN),
            ]
            res_obj.best_trajectory = Trajectory(states=states, maneuvers=maneuvers,
                                                 metadata=td.get("metadata", {}))

    summary = mission_summary_from_result(res_obj, mission)
    return summary.to_dict()

