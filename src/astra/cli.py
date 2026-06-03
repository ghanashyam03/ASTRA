"""ASTRA command-line interface for mission optimization."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("astra.cli")

def cmd_optimize(args: argparse.Namespace) -> int:
    from astra.data.replay import ReplayManifest
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file, parse_mission_string
    from astra.explainability.engine import explain
    from astra.optimization.engine import (
        optimize_mission_bayesian,
        optimize_mission_neural_accelerated,
    )
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel(args.kernels).load()

    manifest = None
    if args.replay:
        logger.info(f"Replaying optimization from manifest: {args.replay}")
        manifest = ReplayManifest.load(Path(args.replay))
        if not manifest.verify_kernels(Path(args.kernels)):
            logger.warning(
                "SPICE kernel verification failed (checksum mismatch or missing). "
                "Continuing anyway."
            )
        dsl = parse_mission_string(manifest.mission_yaml, "yaml")
        mission = compile_mission(dsl, kernel.ephemeris)
        # Override parameters from the manifest for deterministic replay
        mission.seed = manifest.seed
        trials = manifest.n_trials
        time_limit = manifest.time_limit_seconds
    else:
        if not args.mission:
            logger.error(
                "Error: Path to mission YAML file is required unless --replay is specified."
            )
            return 1
        logger.info(f"Loading mission: {args.mission}")
        dsl = parse_mission_file(args.mission)
        mission = compile_mission(dsl, kernel.ephemeris)
        trials = args.trials
        time_limit = float(args.time_limit)

    logger.info(f"Optimizing {mission.mission_id} with {trials} trials...")
    strategy = getattr(args, "strategy", "bayesian")
    if getattr(args, "neural", False) and strategy == "bayesian":
        strategy = "neural"

    if strategy == "neural":
        result = optimize_mission_neural_accelerated(
            mission, kernel,
            n_trials=trials,
            time_limit=time_limit,
            seed=mission.seed,
            pretrain_samples=args.pretrain,
        )
    elif strategy == "hybrid":
        from astra.optimization.engine import optimize_mission_hybrid
        result = optimize_mission_hybrid(
            mission, kernel,
            n_trials_bayesian=trials,
            time_limit=time_limit,
            seed=mission.seed,
        )
    else:
        result = optimize_mission_bayesian(
            mission, kernel,
            n_trials=trials,
            time_limit=time_limit,
            seed=mission.seed,
        )


    if not result.converged or result.best_trajectory is None:
        logger.error("No feasible trajectory found.")
        return 1

    logger.info(f"DONE. Evaluations: {result.n_evaluations}, "
                f"Feasible: {result.n_feasible}, "
                f"Time: {result.wall_time_s:.1f}s")

    best = result.best_trajectory
    logger.info(f"Best Δv: {best.delta_v_total:.4f} km/s")
    logger.info(f"Duration: {best.duration_days:.1f} days")

    trace = explain(
        best, mission,
        pareto_front=result.pareto_front,
        ephemeris=kernel.ephemeris,
    )

    output = {
        "mission_id": mission.mission_id,
        "result": result.to_dict(),
        "explanation": trace.to_dict(),
    }
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(output, indent=2, default=str))
        logger.info(f"Results written to {out_path}")
    else:
        print(json.dumps(output, indent=2, default=str))

    if args.save_manifest:
        from astra.data.replay import build_manifest
        if args.mission:
            yaml_text = Path(args.mission).read_text(encoding="utf-8")
        else:
            assert manifest is not None
            yaml_text = manifest.mission_yaml
        manifest_to_save = build_manifest(
            mission_yaml=yaml_text,
            mission_id=mission.mission_id,
            seed=mission.seed,
            n_trials=trials,
            time_limit_seconds=time_limit,
            kernel_dir=Path(args.kernels),
        )
        manifest_to_save.save(Path(args.save_manifest))
        logger.info(f"Replay manifest saved to {args.save_manifest}")

    return 0

def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA trajectory optimizer")
    sub = parser.add_subparsers(dest="command")

    opt = sub.add_parser("optimize", help="Optimize a mission trajectory")
    opt.add_argument("mission", nargs="?", help="Path to mission YAML file")
    opt.add_argument("--trials", type=int, default=2000)
    opt.add_argument("--time-limit", type=int, default=120)
    opt.add_argument("--strategy", choices=["bayesian", "neural", "hybrid"],
                     default="bayesian", help="Optimization strategy to use")
    opt.add_argument("--neural", action="store_true",
                     help="Use neural-accelerated optimization (legacy)")
    opt.add_argument("--pretrain", type=int, default=500,
                     help="Samples for neural pretraining")
    opt.add_argument("--kernels", default="data/spice_kernels",
                     help="Path to SPICE kernel directory")
    opt.add_argument("-o", "--output", help="Output JSON file path")
    opt.add_argument("--save-manifest", metavar="PATH",
                     help="Save replay manifest to this path after optimization")
    opt.add_argument("--replay", metavar="PATH",
                     help="Replay optimization from a saved manifest file")

    args = parser.parse_args()
    if args.command == "optimize":
        sys.exit(cmd_optimize(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
