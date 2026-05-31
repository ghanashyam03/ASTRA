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
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.explainability.engine import explain
    from astra.optimization.engine import (
        optimize_mission_bayesian,
        optimize_mission_neural_accelerated,
    )
    from astra.physics.kernel import PhysicsKernel

    logger.info(f"Loading mission: {args.mission}")
    kernel = PhysicsKernel(args.kernels).load()
    dsl = parse_mission_file(args.mission)
    mission = compile_mission(dsl, kernel.ephemeris)

    logger.info(f"Optimizing {mission.mission_id} with {args.trials} trials...")
    if args.neural:
        result = optimize_mission_neural_accelerated(
            mission, kernel,
            n_trials=args.trials,
            time_limit=float(args.time_limit),
            seed=mission.seed,
            pretrain_samples=args.pretrain,
        )
    else:
        result = optimize_mission_bayesian(
            mission, kernel,
            n_trials=args.trials,
            time_limit=float(args.time_limit),
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
    return 0

def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA trajectory optimizer")
    sub = parser.add_subparsers(dest="command")

    opt = sub.add_parser("optimize", help="Optimize a mission trajectory")
    opt.add_argument("mission", help="Path to mission YAML file")
    opt.add_argument("--trials", type=int, default=2000)
    opt.add_argument("--time-limit", type=int, default=120)
    opt.add_argument("--neural", action="store_true",
                     help="Use neural-accelerated optimization")
    opt.add_argument("--pretrain", type=int, default=500,
                     help="Samples for neural pretraining")
    opt.add_argument("--kernels", default="data/spice_kernels",
                     help="Path to SPICE kernel directory")
    opt.add_argument("-o", "--output", help="Output JSON file path")

    args = parser.parse_args()
    if args.command == "optimize":
        sys.exit(cmd_optimize(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
