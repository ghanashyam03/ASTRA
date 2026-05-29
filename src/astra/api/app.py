"""ASTRA FastAPI application. Zero physics logic in this file."""
from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from astra.data.storage import TrajectoryStore
from astra.physics.kernel import PhysicsKernel

logger = logging.getLogger(__name__)

# Global state (module-level singletons, not thread-safe — single worker)
_kernel: PhysicsKernel | None = None
_store: TrajectoryStore | None = None
_jobs: dict[str, dict[str, Any]] = {}   # in-memory job tracking

def get_kernel() -> PhysicsKernel:
    global _kernel
    if _kernel is None:
        raise RuntimeError("Physics kernel not initialized")
    return _kernel

def get_store() -> TrajectoryStore:
    global _store
    if _store is None:
        raise RuntimeError("Storage not initialized")
    return _store

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _kernel, _store
    logger.info("Starting ASTRA — loading SPICE kernels...")
    try:
        _kernel = PhysicsKernel().load()
        logger.info("SPICE kernels loaded.")
    except Exception as e:
        logger.warning(f"SPICE kernels not loaded: {e}. Physics endpoints will be unavailable.")
        _kernel = PhysicsKernel()  # no kernels — degraded mode
    
    db_path = getattr(app.state, "db_path", "data/astra.duckdb")
    if db_path != ":memory:":
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _store = TrajectoryStore(db_path)
    
    logger.info(f"ASTRA ready with database {db_path}.")
    yield
    if _store:
        _store.close()
    logger.info("ASTRA shutdown complete.")

app = FastAPI(
    title="ASTRA",
    description="Autonomous Space Trajectory Reasoning Architecture",
    version="0.1.0",
    lifespan=lifespan,
)

# ─── Health ────────────────────────────────────────────────────────────────
@app.get("/v1/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "ASTRA",
        "version": "0.1.0",
        "spice_loaded": _kernel is not None and _kernel._kernels_loaded,
    }

# ─── Mission Optimization ─────────────────────────────────────────────────
class OptimizeRequest(BaseModel):
    mission_yaml: str   # full YAML string

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

@app.post("/v1/missions/optimize", status_code=202)
async def optimize_mission(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "job_id": job_id}
    background_tasks.add_task(_run_optimization, job_id, request.mission_yaml)
    return {"job_id": job_id, "status": "queued",
            "poll_url": f"/v1/missions/{job_id}/status"}

@app.get("/v1/missions/{job_id}/status")
async def mission_status(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, "status": job["status"]}

@app.get("/v1/missions/{job_id}/result")
async def mission_result(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "complete":
        raise HTTPException(status_code=409,
                            detail=f"Job not complete. Status: {job['status']}")
    return job

# ─── Trajectories ─────────────────────────────────────────────────────────
@app.get("/v1/trajectories/{trajectory_id}")
async def get_trajectory(trajectory_id: str) -> dict[str, Any]:
    store = get_store()
    result = store.get_trajectory(trajectory_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    return result

@app.get("/v1/trajectories/{trajectory_id}/explanation")
async def get_explanation(trajectory_id: str) -> dict[str, Any]:
    store = get_store()
    result = store.get_trajectory(trajectory_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    explanation = result.get("explanation")
    if not isinstance(explanation, dict):
        raise HTTPException(status_code=404, detail="No explanation for this trajectory")
    return explanation

# ─── Physics endpoints ─────────────────────────────────────────────────────
@app.get("/v1/bodies/{body_name}/state")
async def get_body_state(body_name: str, epoch_j2000: float = 0.0) -> dict[str, Any]:
    kernel = get_kernel()
    if not kernel._kernels_loaded:
        raise HTTPException(status_code=503, detail="SPICE kernels not loaded")
    from astra.state.orbital_state import CelestialBody
    try:
        body = CelestialBody[body_name.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown body: {body_name}")
    try:
        state = kernel.get_body_state(body, epoch_j2000)
        return state.to_dict()
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

# ─── Porkchop ─────────────────────────────────────────────────────────────
class PorkchopRequest(BaseModel):
    mission_yaml: str
    n_dep: int = 50
    n_tof: int = 50

@app.post("/v1/windows/porkchop")
async def porkchop(request: PorkchopRequest) -> dict[str, Any]:
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_string
    from astra.optimization.engine import compute_porkchop
    from astra.visualization.heatmap import build_porkchop_plot
    kernel = get_kernel()
    dsl = parse_mission_string(request.mission_yaml)
    mission = compile_mission(
        dsl,
        kernel.ephemeris if kernel._kernels_loaded else None
    )
    dep_epochs, tof_days, dv_grid = compute_porkchop(
        mission, kernel,
        n_dep=min(request.n_dep, 100),
        n_tof=min(request.n_tof, 100),
    )
    plot_data = build_porkchop_plot(
        dep_epochs, tof_days, dv_grid,
        ephemeris=kernel.ephemeris if kernel._kernels_loaded else None,
    )
    return plot_data.to_dict()
