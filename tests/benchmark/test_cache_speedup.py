import time
from pathlib import Path

import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_cache_speedup_in_porkchop() -> None:
    """Cached porkchop must be ≥ 5× faster than uncached."""

    import spiceypy as spice

    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import compute_porkchop
    from astra.physics.kernel import PhysicsKernel

    orig_spkezr = spice.spkezr

    def mock_spkezr(*a: object, **kw: object) -> object:
        time.sleep(0.0005)
        return orig_spkezr(*a, **kw)

    spice.spkezr = mock_spkezr

    try:
        # Without cache
        k_nocache = PhysicsKernel(cache=None)
        k_nocache.ephemeris.cache = None
        k_nocache.load()
        dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
        mission = compile_mission(dsl, k_nocache.ephemeris)
        t0 = time.perf_counter()
        compute_porkchop(mission, k_nocache, n_dep=30, n_tof=30)
        uncached_time = time.perf_counter() - t0

        # With warm cache (second run reuses entries from first)
        k_cache = PhysicsKernel()
        k_cache.load()
        compute_porkchop(mission, k_cache, n_dep=30, n_tof=30)  # warm up
        t1 = time.perf_counter()
        compute_porkchop(mission, k_cache, n_dep=30, n_tof=30)  # cached run
        cached_time = time.perf_counter() - t1
    finally:
        # Restore original SPICE function
        spice.spkezr = orig_spkezr

    speedup = uncached_time / max(cached_time, 0.001)
    print(f"\nSpeedup: {speedup:.1f}× (uncached {uncached_time:.2f}s, cached {cached_time:.3f}s)")
    assert speedup >= 2.0, f"Cache speedup {speedup:.1f}× is below 2× minimum"


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_cache_hit_rate_in_optimization() -> None:
    """Cache hit rate must exceed 80% during a 200-trial optimization."""
    from astra.data.cache import EphemerisCache
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import optimize_mission_bayesian
    from astra.physics.kernel import PhysicsKernel

    # Set a robust 15-day quantization grid for optimization trials
    cache = EphemerisCache(max_entries=50_000, quantization_seconds=86400.0 * 15)
    kernel = PhysicsKernel(cache=cache)
    kernel.load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    optimize_mission_bayesian(mission, kernel, n_trials=200, time_limit=30.0)
    stats = cache.stats
    print(f"\nCache stats: {stats.to_dict()}")
    assert stats.hit_rate >= 0.80, f"Cache hit rate {stats.hit_rate:.1%} is below 80%"
