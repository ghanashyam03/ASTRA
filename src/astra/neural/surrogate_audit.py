"""Mandatory audit template for any neural surrogate before it may be used in
an optimizer. Reports the SAME categories of evidence that distinguished the
one successful neural change (PINN warm-start) from the three that failed
(GNN policy, FNO surrogate, JAX gradient refinement): absolute accuracy,
ranking quality, and a multi-seed comparison against the no-surrogate
baseline on a real benchmark mission.
"""
# TODO: Implement a CI enforcement test that dynamically scans all NeuralSurrogate
# subclasses and verifies that run_surrogate_audit has been run and reported.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr


@dataclass
class SurrogateAuditReport:
    surrogate_name: str
    n_test_samples: int
    mae: float
    rmse: float
    accuracy_within_1km_s: float
    spearman_correlation: float
    kendall_correlation: float
    multi_seed_speedup: float | None
    multi_seed_quality_delta_pct: float | None
    verdict: str  # human-readable summary, NOT a pass/fail boolean —
    # the audit's job is to inform a merge DECISION, not make it

    def to_dict(self) -> dict[str, Any]:
        return {
            "surrogate_name": self.surrogate_name,
            "n_test_samples": self.n_test_samples,
            "absolute_accuracy": {
                "mae_km_s": round(self.mae, 4),
                "rmse_km_s": round(self.rmse, 4),
                "within_1km_s_fraction": round(self.accuracy_within_1km_s, 4),
            },
            "ranking_quality": {
                "spearman": round(self.spearman_correlation, 4),
                "kendall": round(self.kendall_correlation, 4),
            },
            "multi_seed_comparison": {
                "speedup": (round(self.multi_seed_speedup, 2) if self.multi_seed_speedup else None),
                "quality_delta_pct": (
                    round(self.multi_seed_quality_delta_pct, 4)
                    if self.multi_seed_quality_delta_pct
                    else None
                ),
            },
            "verdict": self.verdict,
        }


def run_surrogate_audit(
    surrogate_name: str,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    baseline_fn: Callable[[int], tuple[float, float]] | None = None,
    surrogate_fn: Callable[[int], tuple[float, float]] | None = None,
    seeds: list[int] | None = None,
) -> SurrogateAuditReport:
    """Run the mandatory audit.

    predictions, ground_truth: matched arrays of surrogate output vs. true
      values (e.g. Δv in km/s) on a held-out test set.
    baseline_fn, surrogate_fn: optional callables taking a seed and returning
      (quality_metric, wall_time_s) for the no-surrogate baseline and the
      surrogate-assisted run respectively — if provided, a multi-seed
      comparison is run. If not provided, only the static accuracy/ranking
      metrics are computed (the report will note the comparison was skipped).
    """
    errors = predictions - ground_truth
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    within_1 = float(np.mean(np.abs(errors) < 1.0))

    spearman_corr, _ = spearmanr(predictions, ground_truth)
    kendall_corr, _ = kendalltau(predictions, ground_truth)

    speedup = None
    quality_delta = None
    if baseline_fn is not None and surrogate_fn is not None and seeds:
        baseline_qualities, baseline_times = [], []
        surrogate_qualities, surrogate_times = [], []
        for seed in seeds:
            bq, bt = baseline_fn(seed)
            sq, st = surrogate_fn(seed)
            baseline_qualities.append(bq)
            baseline_times.append(bt)
            surrogate_qualities.append(sq)
            surrogate_times.append(st)
        mean_baseline_time = float(np.mean(baseline_times))
        mean_surrogate_time = float(np.mean(surrogate_times))
        speedup = mean_baseline_time / max(mean_surrogate_time, 1e-6)
        mean_baseline_q = float(np.mean(baseline_qualities))
        mean_surrogate_q = float(np.mean(surrogate_qualities))
        quality_delta = (
            (mean_surrogate_q - mean_baseline_q) / max(abs(mean_baseline_q), 1e-6) * 100.0
        )

    # Verdict is descriptive, not a gate decision — matching how the actual
    # PINN warm-start audit was written and used.
    verdict_parts = []
    if within_1 < 0.3:
        verdict_parts.append(f"Poor absolute regressor ({within_1:.0%} within 1 km/s).")
    else:
        verdict_parts.append(f"Reasonable absolute regressor ({within_1:.0%} within 1 km/s).")
    if abs(spearman_corr) > 0.2:
        verdict_parts.append(
            f"Useful ranker (Spearman={spearman_corr:.2f}) — "
            f"may be valuable for warm-starting or ordering "
            f"candidates even if absolute values are unreliable."
        )
    else:
        verdict_parts.append(
            f"Weak ranker (Spearman={spearman_corr:.2f}) — "
            f"unlikely to provide useful ordering information."
        )
    if speedup is not None and quality_delta is not None:
        verdict_parts.append(
            f"Multi-seed comparison: {speedup:.2f}x speedup with "
            f"{quality_delta:+.2f}% quality change vs. baseline."
        )
    else:
        verdict_parts.append(
            "Multi-seed comparison was not run — provide "
            "baseline_fn/surrogate_fn/seeds for a complete audit."
        )

    return SurrogateAuditReport(
        surrogate_name=surrogate_name,
        n_test_samples=len(predictions),
        mae=mae,
        rmse=rmse,
        accuracy_within_1km_s=within_1,
        spearman_correlation=float(spearman_corr),
        kendall_correlation=float(kendall_corr),
        multi_seed_speedup=speedup,
        multi_seed_quality_delta_pct=quality_delta,
        verdict=" ".join(verdict_parts),
    )
