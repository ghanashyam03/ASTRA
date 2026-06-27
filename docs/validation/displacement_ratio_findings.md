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

Consequently, the physical planetary displacement relative to the SOI radius is **largest for the inner planets** (Mercury, Mars, Venus, Earth) and **smallest for the outer planets** (Uranus, Neptune). This indicates that missions relying on inner-planet gravity assists (such as MESSENGER or Galileo) experience a much larger planetary motion relative to their SOI during passage, pointing to them as high-priority candidates for higher-fidelity trajectory auditing, whereas outer-planet missions (such as Voyager 2) have relatively small planetary displacements relative to their SOI.


---

## 2. Verification of the Screening Metric Interpretation

The displacement ratio is **strictly a screening metric** and cannot be interpreted as a direct measure of patched-conics mission-level trajectory errors (such as C3, flight duration, or Delta-V error):
- **A large ratio does not automatically imply a large trajectory error.** If a spacecraft is on a high-energy trajectory where gravitational deflection is minimal, or if the flyby geometry is such that the entry and exit errors cancel out, the final trajectory cost error may remain very small despite a large body displacement.
- **A small ratio does not prove that patched-conics is exact.** For extremely close flybys, the spacecraft passes deep into the planet's gravity well, where the gravitational acceleration is highly sensitive to the spacecraft-planet distance. In this regime, even a tiny displacement of the planet can cause a significant change in the deflection angle and resulting flight path.
- **Purpose:** The metric is designed purely as a low-cost screening tool to flag which celestial bodies warrant higher-fidelity propagation and analysis.


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

---

## 5. Three-Body Heliocentric Propagation Comparison

To validate the physical effect of the planet's motion during the SOI crossing, we performed a genuine three-body numerical propagation (Sun + planet gravity using time-varying SPICE ephemerides) and compared it directly against a patched-conics-inspired propagation where the planet's position is frozen at the periapsis epoch \(t_{\text{peri}}\). 

The initial state in both cases is the exact same heliocentric state vector constructed at periapsis. The propagation was run for a half-crossing duration \(dt_{\text{half}} = r_{\text{SOI}} / v_{\infty}\) to the SOI exit.

### Results of Three-Body Comparison

At the shared exit epoch \(t_{\text{exit}} = t_{\text{peri}} + dt_{\text{half}}\), the raw state deviations (position and velocity) are:

| Body | Half-Crossing Duration (days) | Absolute Position Deviation (km) | Absolute Velocity Deviation (km/s) | Relative Position Deviation (\(d_{\text{dev}} / r_{\text{SOI}}\)) | Relative Velocity Deviation (\(v_{\text{dev}} / v_{\infty}\)) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mercury** | 0.3 | 24,092.8 | 1.159 | 21.4% | 23.2% |
| **Mars** | 1.2 | 170,212.5 | 1.663 | 29.5% | 30.2% |
| **Neptune** | 100.7 | 24,709,339.8 | 2.861 | 28.4% | 28.6% |

### Analysis of the Deviations

1. **Absolute vs. Relative Deviations:** 
   Neptune exhibits a massive absolute position deviation of \(24.7 \times 10^6\) km, which is much larger than Mercury's (24,092.8 km) and Mars's (170,212.5 km). This is due to the vast difference in the propagation time-scale: Neptune's half-crossing takes **100.7 days**, allowing the difference in time-varying gravitational pull to integrate over a long period.
2. **Normalized Deviations:** 
   When normalized by the characteristic scale of each body's gravity well, Mercury shows a position deviation equal to **21.4%** of its SOI radius over just **0.3 days**, and Mars shows **29.5%** over **1.2 days**, whereas Neptune shows **28.4%** over a vastly longer **100.7 days**. This highlights that the rate of error accumulation per unit time is far higher for the inner planets (Mercury: \(80,300\) km/day; Mars: \(141,800\) km/day) than the outer planets (Neptune: \(245,300\) km/day relative to its giant SOI of 87 million km).
3. **Scientific Verification Compliance:**
   - **Common Initial State:** Verified to start from the exact same heliocentric state.
   - **Only One Variable Changes:** Forces and propagation settings are identical; only the time-varying planet position is toggled.
   - **Integration Tolerance Convergence:** Confirmed that running with tighter tolerances (\(10^{-13}\) vs \(10^{-10}\)) changes the measured position deviation by less than 3.7 km (a relative variance of only \(0.002\%\)), showing the result is physically driven rather than an integration artifact.
   - **Zero-Motion Control Case:** A control experiment using a mock static kernel resulted in perfect numerical agreement (position difference \(< 10^{-9}\) km, velocity difference \(< 10^{-11}\) km/s), demonstrating implementation consistency.

---

## 6. Verification of Encounter Orientation Independence

To ensure the measured frozen-vs-moving trajectory deviations are physically robust and not artifacts of the chosen periapsis coordinate frame, we repeated the three-body propagation for **10 randomly rotated encounter geometries** (varying both the flyby-plane orientation and incoming \(v_{\infty}\) direction) while holding the body, \(v_{\infty}\), periapsis altitude, and encounter epoch constant.

### Rotation Distribution Results

| Body | Absolute Position Dev (km) [Mean ± Std] | Absolute Velocity Dev (km/s) [Mean ± Std] | Relative Position Dev (\(d_{\text{dev}} / r_{\text{SOI}}\)) [Mean Range] |
| :--- | :---: | :---: | :---: |
| **Mercury** | 51,033.1 ± 20,797.0 | 2.3795 ± 0.9383 | **45.40%** (23.29% – 86.99%) |
| **Mars** | 210,289.3 ± 27,551.5 | 2.0514 ± 0.2666 | **36.44%** (30.30% – 46.25%) |
| **Neptune** | 45,509,961.4 ± 10,010,662.6 | 5.3825 ± 1.9364 | **52.32%** (26.12% – 70.81%) |

### Orientation Independence Analysis

1. **Orientation-Independent Conclusions:** Across all randomly rotated geometries, the relative position deviation at the comparison epoch is consistently large—ranging between **23% and 87%** of the SOI radius. This confirms that the frozen-planet patched-conics approximation introduces substantial boundary state-vector deviations regardless of the encounter plane or incoming trajectory direction. The scientific conclusions are entirely independent of the arbitrary choice of local coordinate frame.
2. **Absolute vs. Normalized Trends:** 
   - While Neptune's displacement ratio screening metric is the smallest (~1.1), its actual normalized position deviation at the comparison epoch (Mean: 52.32% of SOI) is comparable to or slightly larger than Mars (Mean: 36.44%) and Mercury (Mean: 45.40%). 
   - This occurs because Neptune's passage takes **100.7 days**, allowing the difference between a moving gravitational source and a frozen point-mass to integrate over a vast timescale. 
   - Thus, while the displacement ratio correctly screens for the magnitude of planetary motion during a passage, the final heliocentric state-vector deviation is heavily influenced by the propagation timescale. Both inner and outer planets suffer significant boundary state-vector deviations under a frozen-planet model.

