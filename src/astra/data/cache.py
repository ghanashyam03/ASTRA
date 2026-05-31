"""Epoch-keyed LRU cache for SPICE ephemeris state queries.

Thread-unsafe — single-process use only (matches ASTRA's single-worker API).
Cache key: (body_name: str, epoch_j2000: float quantized to 1-second grid,
            frame: str, observer: str)
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_QUANTIZATION_SECONDS = 60.0

def _quantize_epoch(
    epoch: float,
    quantization_seconds: float,
) -> float:
    """Round epoch to nearest quantization grid point."""
    return (
        round(epoch / quantization_seconds)
        * quantization_seconds
    )

def _cache_key(
    body: str,
    epoch: float,
    frame: str,
    observer: str,
    quantization_seconds: float,
) -> tuple[str, float, str, str]:
    return (
        body,
        _quantize_epoch(epoch, quantization_seconds),
        frame,
        observer,
    )

@dataclass
class CacheEntry:
    position: np.ndarray   # shape (3,) float64 km
    velocity: np.ndarray   # shape (3,) float64 km/s
    created_at: float = field(default_factory=time.time)

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate_pct": round(self.hit_rate * 100, 1),
        }

class EphemerisCache:
    """In-memory LRU cache for ephemeris state vectors.

    Parameters
    ----------
    max_entries : int
        Maximum number of (body, epoch, frame, observer) entries.
        Default 50_000 covers a full 2-year porkchop at 1-minute resolution.
    persist_path : Path | None
        If provided, cache is saved/loaded from this JSON file.
    """

    def __init__(
        self,
        max_entries: int = 50_000,
        persist_path: Path | None = None,
        quantization_seconds: float = DEFAULT_QUANTIZATION_SECONDS,
    ) -> None:
        self._cache: OrderedDict[tuple[str, float, str, str], CacheEntry] = OrderedDict()
        self.max_entries = max_entries
        self.persist_path = persist_path
        self.quantization_seconds = quantization_seconds
        self.stats = CacheStats()
        if persist_path and persist_path.exists():
            self._load(persist_path)

    def get(
        self, body: str, epoch: float, frame: str, observer: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (position, velocity) arrays if cached, else None."""
        key = _cache_key(
            body,
            epoch,
            frame,
            observer,
            self.quantization_seconds,
        )
        entry = self._cache.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        # LRU: move to end
        self._cache.move_to_end(key)
        self.stats.hits += 1
        return entry.position.copy(), entry.velocity.copy()

    def put(
        self,
        body: str,
        epoch: float,
        frame: str,
        observer: str,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        """Store (position, velocity) under cache key."""
        key = _cache_key(
            body,
            epoch,
            frame,
            observer,
            self.quantization_seconds,
        )
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)  # evict LRU
                self.stats.evictions += 1
        self._cache[key] = CacheEntry(
            position=np.asarray(position, dtype=np.float64).copy(),
            velocity=np.asarray(velocity, dtype=np.float64).copy(),
        )

    def clear(self) -> None:
        self._cache.clear()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._cache)

    def _save(self, path: Path) -> None:
        """Persist cache to JSON for cross-run reuse."""
        data = {}
        for key, entry in self._cache.items():
            k = json.dumps(list(key))
            data[k] = {
                "pos": entry.position.tolist(),
                "vel": entry.velocity.tolist(),
            }
        path.write_text(json.dumps(data))

    def _load(self, path: Path) -> None:
        """Load cache from JSON file."""
        try:
            data = json.loads(path.read_text())
            for k_str, v in data.items():
                k_list = json.loads(k_str)
                key = (k_list[0], float(k_list[1]), k_list[2], k_list[3])
                self._cache[key] = CacheEntry(
                    position=np.array(v["pos"], dtype=np.float64),
                    velocity=np.array(v["vel"], dtype=np.float64),
                )
        except Exception:
            self._cache.clear()

    def save_if_configured(self) -> None:
        if self.persist_path:
            self._save(self.persist_path)
