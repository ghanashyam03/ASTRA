# Flyby SOI Passage Displacement Ratio Findings

This document presents the screening-level results estimating how much a planet moves along its orbit during the time a spacecraft spends inside its Sphere of Influence (SOI), relative to the SOI radius itself.

## 1. Findings Table (Epoch: 2030-01-01T00:00:00)

Using precise SPICE ephemerides, the following values were computed at epoch J2000 `2030-01-01T00:00:00` (J2000 seconds = `946728000.0`):

| Body | Chosen \(v_{\infty}\) (km/s) | SOI Radius \(r_{\text{SOI}}\) (km) | Crossing Duration \(t_{\text{cross}}\) (days) | Planet Orbital Speed \(v_{\text{planet}}\) (km/s) | Planet Displacement \(d_{\text{planet}}\) (km) | Displacement Ratio (\(d_{\text{planet}} / r_{\text{SOI}}\)) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mercury** | 5.0 | 112,408 | 0.52 | 58.166 | 2,615,343 | 23.267 |
| **Mars** | 5.5 | 577,117 | 2.43 | 26.497 | 5,560,737 | 9.635 |
| **Venus** | 9.7 | 616,243 | 1.47 | 35.216 | 4,474,614 | 7.261 |
| **Earth** | 9.0 | 924,600 | 2.38 | 30.295 | 6,224,620 | 6.732 |
| **Saturn** | 8.0 | 54,747,956 | 158.41 | 10.064 | 137,741,396 | 2.516 |
| **Jupiter** | 10.0 | 48,207,344 | 111.59 | 12.526 | 120,766,834 | 2.505 |
| **Uranus** | 9.0 | 51,696,505 | 132.96 | 6.785 | 77,951,220 | 1.508 |
| **Neptune**| 10.0 | 86,978,993 | 201.34 | 5.466 | 95,092,883 | 1.093 |

### The Inner-Planet-Largest Trend and Its Implications

The data demonstrates a clear, counter-intuitive trend: **inner planets exhibit significantly larger displacement ratios than outer gas giants**. While gas giants have massive SOIs (e.g., Neptune's \(r_{\text{SOI}} \approx 87 \times 10^6\) km compared to Mercury's \(r_{\text{SOI}} \approx 112 \times 10^3\) km) and spacecraft spend hundreds of days crossing them, the planets themselves move very slowly in their orbits. Conversely, inner planets travel at high orbital velocities (up to 58 km/s for Mercury) and have compact SOIs. Because the displacement ratio simplifies mathematically to:
\[
\text{ratio} = \frac{2 \cdot v_{\text{planet}}}{v_{\infty}}
\]
the ratio is directly proportional to the planet's orbital speed and inversely proportional to the encounter \(v_{\infty}\), independent of the SOI radius itself. 

Consequently, the "planet frozen during flyby" patched-conics approximation is **least accurate for the inner planets** (Mercury, Mars, Venus, Earth) and **most accurate for the outer planets** (Uranus, Neptune). This indicates that historical missions relying on inner-planet gravity assists (such as MESSENGER or Galileo) require closer physical auditing (re-audited in later stages) because the body displacement during crossing is several times larger than the SOI radius itself, whereas outer-planet missions (such as Voyager 2) are significantly less sensitive to this approximation error.

---

## 2. Verification of the Screening Metric Interpretation

The displacement ratio is **strictly a screening metric** and cannot be interpreted as a direct measure of patched-conics trajectory error:
- **A large ratio does not automatically imply a large trajectory error.** If a spacecraft is on a high-energy trajectory where gravitational deflection is minimal, or if the flyby geometry is such that the entry and exit errors cancel out, the final trajectory error may remain very small despite a large body displacement.
- **A small ratio does not prove that patched-conics is exact.** For extremely close flybys, the spacecraft passes deep into the planet's gravity well, where the gravitational acceleration is highly sensitive to the spacecraft-planet distance. In this regime, even a tiny displacement of the planet can cause a significant change in the deflection angle.
- **Purpose:** The metric is designed purely as a low-cost screening tool to flag which celestial bodies require higher-fidelity propagation and analysis.

---

## 3. Verification of the Conservative Crossing-Time Assumption

We verify that the crossing duration \(t_{\text{cross}} = \frac{2 r_{\text{SOI}}}{v_{\infty}}\) is indeed a conservative upper bound.
The instantaneous planet-relative speed \(v(r)\) of the spacecraft on a hyperbolic flyby trajectory at distance \(r\) satisfies the vis-viva equation:
\[
v(r)^2 = v_{\infty}^2 + \frac{2\mu}{r}
\]
Since \(r \le r_{\text{SOI}}\) throughout the SOI passage, we have:
\[
v(r) \ge v_{\text{min}} = \sqrt{v_{\infty}^2 + \frac{2\mu}{r_{\text{SOI}}}}
\]
Because \(\mu > 0\), the term \(\frac{2\mu}{r_{\text{SOI}}}\) is strictly positive, meaning \(v(r) > v_{\infty}\) at all points inside the SOI. Since the spacecraft's actual path length inside the SOI sphere is at most its diameter \(2 r_{\text{SOI}}\), and its speed is strictly greater than \(v_{\infty}\) everywhere, the actual duration of the SOI passage is guaranteed to be shorter than \(t_{\text{cross}}\).

The minimum spacecraft speeds \(v_{\text{min}}\) at the SOI boundary were calculated as:

| Body | Chosen \(v_{\infty}\) (km/s) | SOI boundary minimum speed \(v_{\text{min}}\) (km/s) | Inequality \(v(r) \ge v_{\infty}\) Holds? |
| :--- | :---: | :---: | :---: |
| **Mercury** | 5.0 | 5.039 | Yes (strictly greater) |
| **Venus** | 9.7 | 9.754 | Yes (strictly greater) |
| **Earth** | 9.0 | 9.048 | Yes (strictly greater) |
| **Mars** | 5.5 | 5.513 | Yes (strictly greater) |
| **Jupiter** | 10.0 | 10.259 | Yes (strictly greater) |
| **Saturn** | 8.0 | 8.086 | Yes (strictly greater) |
| **Uranus** | 9.0 | 9.012 | Yes (strictly greater) |
| **Neptune** | 10.0 | 10.008 | Yes (strictly greater) |

Because \(v(r) \ge v_{\infty}\) holds strictly for all cases, the crossing-time estimate \(t_{\text{cross}}\) is a mathematically guaranteed upper bound.

---

## 4. Verification of Trend Robustness Across Epochs

To ensure the outward-decreasing trend is robust against changes in planetary positions along their eccentric orbits, we computed the displacement ratios at three different epochs:

| Body | Ratio (2030-01-01) | Ratio (2035-01-01) | Ratio (2040-01-01) |
| :--- | :---: | :---: | :---: |
| **Mercury** | 23.267 | 19.668 | 15.709 |
| **Mars** | 9.635 | 8.304 | 8.376 |
| **Venus** | 7.261 | 7.269 | 7.247 |
| **Earth** | 6.732 | 6.730 | 6.727 |
| **Jupiter** | 2.505 | 2.741 | 2.502 |
| **Saturn** | 2.516 | 2.533 | 2.415 |
| **Uranus** | 1.508 | 1.537 | 1.562 |
| **Neptune** | 1.093 | 1.098 | 1.094 |

### Trend Observations and Ordering Anomalies

1. **Robustness of the Inward-to-Outward Trend:** The primary trend holds across all epochs: the inner planets (ratios ranging from 6.7 to 23.3) have significantly larger ratios than the outer planets (ratios ranging from 1.1 to 2.7).
2. **Mars Deviation:** Mars consistently has a larger ratio than Venus and Earth across all epochs. This is not a bug; rather, it is a consequence of the lower screening \(v_{\infty} = 5.5\) km/s chosen for Mars relative to Venus (\(v_{\infty} = 9.7\) km/s) and Earth (\(v_{\infty} = 9.0\) km/s), combined with Mars's relatively high orbital speed.
3. **Saturn/Jupiter Ordering Swaps:** In 2030, Saturn's ratio (2.516) is slightly higher than Jupiter's (2.505). In 2035 and 2040, Jupiter's ratio (2.741 and 2.502) is higher than Saturn's (2.533 and 2.415). This minor swapping is due to orbital eccentricity. Because Jupiter and Saturn travel on eccentric orbits, their instantaneous orbital velocities vary:
   - Jupiter's orbital speed varies between approximately 12.4 km/s (aphelion) and 13.7 km/s (perihelion).
   - Saturn's orbital speed varies between approximately 9.1 km/s (aphelion) and 10.2 km/s (perihelion).
   Since their ratios are very close due to their respective \(v_{\infty}\) parameters, these natural velocity variations cause them to swap order depending on where they are in their orbits at the chosen epoch.
