from astra.data.cache import EphemerisCache, _quantize_epoch


def test_epoch_quantization():
    q = 60.0
    assert _quantize_epoch(0.0, q) == 0.0
    assert _quantize_epoch(30.0, q) == 0.0
    assert _quantize_epoch(60.0, q) == 60.0
    assert _quantize_epoch(89.9, q) == 60.0
    assert _quantize_epoch(90.1, q) == 120.0


def test_cache_miss_then_hit():
    import numpy as np
    cache = EphemerisCache(max_entries=100)
    assert cache.get("EARTH", 0.0, "ECLIPJ2000", "SUN") is None
    assert cache.stats.misses == 1
    pos = np.array([1.496e8, 0.0, 0.0])
    vel = np.array([0.0, 29.78, 0.0])
    cache.put("EARTH", 0.0, "ECLIPJ2000", "SUN", pos, vel)
    result = cache.get("EARTH", 0.0, "ECLIPJ2000", "SUN")
    assert result is not None
    np.testing.assert_array_equal(result[0], pos)
    assert cache.stats.hits == 1
    assert cache.stats.hit_rate == 0.5


def test_cache_lru_eviction():
    import numpy as np
    cache = EphemerisCache(max_entries=3)
    dummy = np.zeros(3)
    for i in range(4):
        cache.put("BODY", float(i * 60), "ECLIPJ2000", "SUN", dummy, dummy)
    assert len(cache) == 3
    assert cache.stats.evictions == 1
    assert cache.get("BODY", 0.0, "ECLIPJ2000", "SUN") is None  # evicted


def test_cache_epoch_quantization_collapses_nearby_epochs():
    import numpy as np
    cache = EphemerisCache(max_entries=100)
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.1, 0.2, 0.3])
    cache.put("MARS", 0.0, "ECLIPJ2000", "SUN", pos, vel)
    # epoch 30.0 quantizes to 0.0 — should be a cache hit
    result = cache.get("MARS", 30.0, "ECLIPJ2000", "SUN")
    assert result is not None, "Nearby epoch should hit quantized cache entry"


def test_cache_persist_roundtrip(tmp_path):
    import numpy as np
    path = tmp_path / "cache.json"
    cache1 = EphemerisCache(max_entries=100, persist_path=path)
    pos = np.array([1.496e8, 0.1, -0.3])
    vel = np.array([0.05, 29.78, 0.002])
    cache1.put("EARTH", 0.0, "ECLIPJ2000", "SUN", pos, vel)
    cache1.save_if_configured()
    cache2 = EphemerisCache(max_entries=100, persist_path=path)
    result = cache2.get("EARTH", 0.0, "ECLIPJ2000", "SUN")
    assert result is not None
    np.testing.assert_allclose(result[0], pos, rtol=1e-12)


def test_replay_manifest_roundtrip(tmp_path):
    from astra.data.replay import ReplayManifest
    m = ReplayManifest(
        mission_id="test_mission",
        mission_yaml="version: '1.0'\nmission_id: test",
        seed=42,
        n_trials=1000,
        time_limit_seconds=120.0,
    )
    path = tmp_path / "manifest.json"
    m.save(path)
    m2 = ReplayManifest.load(path)
    assert m2.mission_id == "test_mission"
    assert m2.seed == 42
    assert m2.n_trials == 1000
    assert m2.astra_version == m.astra_version
