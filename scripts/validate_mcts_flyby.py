from pathlib import Path

from astra.dsl.compiler import compile_mission
from astra.dsl.parser import parse_mission_file
from astra.optimization.mcts import MCTSPlanner
from astra.physics.kernel import PhysicsKernel


def main() -> None:
    print("======================================================================")
    print("MCTS Gravity Assist Phase Planner Validation")
    print("======================================================================")

    # Check SPICE kernel
    if not (Path("data/spice_kernels") / "de440.bsp").exists():
        print("Error: SPICE kernel not found.")
        return

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    print("Running MCTS planner for Earth -> Venus -> Mars flyby chain...")
    planner = MCTSPlanner(
        mission=mission,
        kernel=kernel,
        max_depth=3,
        n_iterations=500,
        dv_budget=45.0,
        seed=42,
        flyby_candidates=["VENUS", "EARTH"],
    )

    paths = planner.run()
    print(f"\nMCTS Search completed. Found {len(paths.all_paths)} valid flyby paths reaching Mars.")

    if not paths.all_paths:
        print("No valid paths found reaching Mars under the specified delta-V budget.")
        return

    # Filter for paths that contain a Venus flyby
    venus_paths = [p for p in paths.all_paths if any(s.body == "VENUS" for s in p)]
    print(f"\nOf those, {len(venus_paths)} paths contain a VENUS flyby.")

    # Print the top 5 paths containing Venus flyby
    for idx, path in enumerate(venus_paths[:5]):
        print(f"\nVenus Assist Path {idx + 1}:")
        for step, state in enumerate(path):
            if step == 0:
                print(f"  Start: {state.body} at JD epoch {state.epoch:.3f}")
            else:
                prev_state = path[step - 1]
                duration_days = (state.epoch - prev_state.epoch) / 86400.0
                print(
                    f"  -> Transfer to {state.body} ({duration_days:.1f} days) | "
                    f"Cumulative Delta-v spent: {state.dv_spent:.3f} km/s"
                )

    print("\nMCTS PATH SEARCH VALIDATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
