# Post-Mortem Analysis: Fourier Feature Network (PorkchopFNO) for Grid Acceleration

**Status:** Rejected / Abandoned (Kept in feature branch, not merged to main)  
**Date:** June 13, 2026  
**Author:** Antigravity Coding Assistant & ASTRA Team  

---

## Executive Summary
This document outlines the findings, benchmark data, and scientific analysis regarding the integration of `PorkchopFNO` (a Fourier Feature Network surrogate) for accelerating the calculation of 2-D porkchop $\Delta v$ grids. 

Our benchmark results indicate that while the PorkchopFNO grid computation achieves a **~8x speedup factor**, the exact optimum recovery rate is **0.0%** (well below the required **95.0%** threshold). Therefore, this approach is rejected for production and will remain in the feature branch for reference only.

---

## 1. Benchmark Results

A comparison of exact Lambert grid generation against the PorkchopFNO accelerated grid (trained on a coarse 30x30 Lambert grid):

| Resolution | Lambert Time | FNO Hybrid Time | Speedup | Exact Min $\Delta v$ | FNO Min $\Delta v$ | Error from Min | Optimum Recovered? |
|---|---|---|---|---|---|---|---|
| **30x30** | 0.712s | 0.094s | 7.56x | 4.963 km/s | 6.990 km/s | 2.027 km/s | **False** (0.0%) |
| **50x50** | 1.947s | 0.233s | 8.34x | 4.963 km/s | 7.117 km/s | 2.154 km/s | **False** (0.0%) |
| **100x100** | 7.784s | 0.982s | 7.93x | 4.963 km/s | 7.185 km/s | 2.222 km/s | **False** (0.0%) |
| **150x150** | 17.935s | 2.218s | 8.09x | 4.963 km/s | 7.222 km/s | 2.260 km/s | **False** (0.0%) |

### Precision Distribution (predicted cells relative to exact):
- **Within 0.25 km/s**: ~2.0% of cells
- **Within 0.50 km/s**: ~4.0% of cells
- **Within 1.00 km/s**: ~8.0% of cells

---

## 2. Why the FNO Approach Failed

### I. Extreme Discontinuity & Valley Smoothing
Porkchop plots are characterized by narrow, highly non-convex valleys of low $\Delta v$ (optima) surrounded by massive ridges of high or infinite $\Delta v$ (infeasible boundaries). Neural networks, including those with random Fourier feature encodings (FFNs), act as smooth function approximators. 
- The network completely smooths out these local, sharp valleys, shifting the predicted optimum into sub-optimal regions.

### II. Log-Space Target Compression
To prevent massive gradient domination by infeasible values, targets are trained in log-space: $\log(\Delta v + 1)$.
- While log-space stabilizes gradient descent, it heavily compresses the difference between optimal values (e.g., 4.9 km/s vs. 7.0 km/s). This flattening reduces the gradient signals in critical valley zones, making the model insensitive to fine-grained optimal structures.

### III. Poor Generalization to Higher Resolutions
Because the coarse grid (30x30) has a resolution step of ~7 days in departure date, it misses the extremely sharp, high-frequency variations in the porkchop grid. When predicting on a finer resolution (e.g., 150x150), the FNO simply interpolates a blurry, low-frequency approximation of the true structure, which is mathematically incapable of identifying the exact local optima.

---

## 3. Alternative Directions
To achieve fast porkchop grid generation without losing optimum recovery:
1. **Adaptive Grid Refining**: Compute a very coarse grid (e.g. 20x20) using exact physics, then adaptively refine only the sub-regions containing the local minima (quadtree style). This achieves >10x speedups without any neural network overhead or loss of accuracy.
2. **Deep Physics-Informed Surrogates**: Rather than training an offline MLP on a single grid, use pre-trained Physics-Informed Neural Networks (PINNs) that exploit the governing orbital differential equations to predict state transfers.
