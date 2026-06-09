"""Missions routes for the ASTRA API.

Contains:
- POST /v1/missions/optimize: Submit a mission optimization job.
- GET /v1/missions/{job_id}/status: Check optimization job status.
- GET /v1/missions/{job_id}/result: Get completed optimization job results.
- GET /v1/missions/{id}/sensitivity: Run sensitivity analysis on optimized trajectory.
- GET /v1/missions/{id}/pareto-metrics: Get Pareto front metrics for optimization run.
"""

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from astra.api.app import get_kernel, get_store, _jobs
from astra.api.schemas.requests import OptimizeRequest
from astra.api.schemas.responses import JobSubmittedResponse, JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["missions"])


def _run_optimization(job_id: str, mission_yaml: str) -> None:
    """Background task: parse, compile, optimize, save."""
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_string
    from astra.explainability.engine import explain
    from astra.optimization.engine import optimize_mission_bayesian
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


@router.get("/v1/missions/{id}/sensitivity")
async def get_sensitivity(id: str) -> dict[str, Any]:
    """Run sensitivity analysis on the optimized best trajectory."""
    job = _jobs.get(id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {id} not found")
    if job["status"] != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Job not complete. Status: {job['status']}"
        )

    best_tid = job.get("best_trajectory_id")
    if not best_tid:
        raise HTTPException(status_code=404, detail="No trajectory optimized for this job")

    store = get_store()
    traj_data = store.get_trajectory(best_tid)
    if traj_data is None:
        raise HTTPException(status_code=404, detail="Trajectory not found")

    import numpy as np
    from astra.state.orbital_state import OrbitalState
    from astra.state.trajectory import Maneuver, Trajectory

    t_dict = traj_data["trajectory"]
    maneuvers = [
        Maneuver(
            epoch=m["epoch"],
            delta_v=np.array(m["dv_km_s"]),
            label=m.get("label", ""),
        )
        for m in t_dict["maneuvers"]
    ]
    s0 = OrbitalState(
        epoch=t_dict["departure_epoch_j2000"],
        position=np.zeros(3),
        velocity=np.zeros(3),
    )
    s1 = OrbitalState(
        epoch=t_dict["arrival_epoch_j2000"],
        position=np.zeros(3),
        velocity=np.zeros(3),
    )
    trajectory = Trajectory(
        states=[s0, s1],
        maneuvers=maneuvers,
        metadata=t_dict.get("metadata", {}),
    )

    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_string
    kernel = get_kernel()
    mission_yaml = job.get("mission_yaml")
    if not mission_yaml:
        raise HTTPException(status_code=404, detail="Mission YAML not found in job")
    dsl = parse_mission_string(mission_yaml)
    mission = compile_mission(
        dsl,
        kernel.ephemeris if kernel._kernels_loaded else None
    )

    from astra.visualization.sensitivity import analyze_trajectory_sensitivity
    try:
        sens_result = analyze_trajectory_sensitivity(trajectory, mission, kernel)
        return sens_result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis failed: {e}")


@router.get("/v1/missions/{id}/pareto-metrics")
async def get_pareto_metrics(id: str) -> dict[str, Any]:
    """Retrieve Pareto frontier metrics for the optimization run."""
    job = _jobs.get(id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {id} not found")
    if job["status"] != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Job not complete. Status: {job['status']}"
        )

    mission_id = job.get("mission_id")
    if not mission_id:
        raise HTTPException(status_code=404, detail="Mission ID not found in job")

    store = get_store()
    rows = store.conn.execute(
        "SELECT trajectory_json FROM trajectories WHERE mission_id = ? AND feasible = true",
        [mission_id],
    ).fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No feasible trajectories found for this mission"
        )

    import numpy as np
    from astra.state.orbital_state import OrbitalState
    from astra.state.trajectory import Maneuver, Trajectory

    trajectories = []
    for r in rows:
        t_dict = json.loads(r[0])
        maneuvers = [
            Maneuver(
                epoch=m["epoch"],
                delta_v=np.array(m["dv_km_s"]),
                label=m.get("label", ""),
            )
            for m in t_dict["maneuvers"]
        ]
        s0 = OrbitalState(
            epoch=t_dict["departure_epoch_j2000"],
            position=np.zeros(3),
            velocity=np.zeros(3),
        )
        s1 = OrbitalState(
            epoch=t_dict["arrival_epoch_j2000"],
            position=np.zeros(3),
            velocity=np.zeros(3),
        )
        trajectories.append(Trajectory(
            states=[s0, s1],
            maneuvers=maneuvers,
            metadata=t_dict.get("metadata", {}),
        ))

    from astra.visualization.pareto_plot import build_pareto_plot
    plot_data = build_pareto_plot(trajectories)
    return plot_data.to_dict()
