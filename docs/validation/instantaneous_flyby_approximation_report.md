# Instantaneous-Flyby Approximation — Measured Error

## Method
The error in the instantaneous-flyby approximation (used by `compute_flyby`'s closed-form Rodrigues rotation) is measured by:
1. Constructing the true periapsis state of the hyperbolic encounter based on the vis-viva equation:
   $$v_{\text{peri}} = \sqrt{v_{\infty,\text{in}}^2 + \frac{2\mu}{r_p}}$$
   where the position vector at periapsis points along the local $x$-axis ($\mathbf{r}_0 = [r_p, 0, 0]$) and the velocity vector is purely tangential ($\mathbf{v}_0 = [0, v_{\text{peri}}, 0]$) in the planet-centric frame.
2. Deriving a principled stopping distance $r_{\text{stop}}$ corresponding to a strict speed convergence tolerance (fractional speed deviation squared $< 10^{-8}$):
   $$r_{\text{stop}} = \frac{2\mu}{v_{\infty,\text{in}}^2 \cdot 10^{-8}}$$
3. Propagating the initial periapsis state forward and backward using a Runge-Kutta 4(5) adaptive integrator (`propagate_two_body`) until the position magnitude reaches or exceeds $r_{\text{stop}}$. The resulting vectors at these boundaries represent the true numerical incoming and outgoing asymptotes.
4. Feeding the numerical incoming asymptote to `compute_flyby` using the same flyby plane normal, and performing an apples-to-apples comparison of the resulting outgoing excess velocity directions (measuring the angular discrepancy in degrees).

## Results

| Body | $v_{\infty}$ (km/s) | Periapsis alt (km) | Angular discrepancy (deg) | $r_{\text{stop}}$ vs SOI |
| :--- | :--- | :--- | :--- | :--- |
| **JUPITER** | 10.0 | 350,000.0 | $1.71 \times 10^{-6}$ | $2.53 \times 10^{14}$ km vs $4.82 \times 10^7$ km (exceeds SOI) |
| **SATURN** | 8.0 | 200,000.0 | $0.00$ | $1.19 \times 10^{14}$ km vs $5.47 \times 10^7$ km (exceeds SOI) |
| **URANUS** | 9.0 | 100,000.0 | $0.00$ | $1.43 \times 10^{13}$ km vs $5.17 \times 10^7$ km (exceeds SOI) |

## Stopping Criteria Comparison: Speed vs. Direction

An optional direction-based stopping criterion was implemented and evaluated against the speed-based criterion on a Jupiter flyby ($v_{\infty} = 10.0$ km/s, $r_p = 421,492$ km):

* **Direction-based criterion**: Stops when the gravity deflection angular rate $\omega = \frac{\mu h}{r^3 v}$ drops below a strict tolerance (e.g. $10^{-12}$ rad/s).
* **Speed-based criterion**: Stops when the fractional speed deviation squared is below $10^{-8}$.

### Comparison Results

* **Angular Agreement**: Both criteria produce the identical outgoing asymptotic direction within $1.21 \times 10^{-6}$ degrees (numerical float precision limit).
* **Distance Reduction**: The direction-based criterion terminates at $r_{\text{stop}} \approx 5.2 \times 10^8$ km compared to $2.5 \times 10^{14}$ km for the speed-based criterion. This represents a **480,000-fold reduction** in propagation distance.
* **Runtime Reduction**: The runtime drops from $\approx 2.5$ seconds to $\approx 0.6$ seconds (a **4-fold reduction** in execution time).

### Rationale for Default Selection
Although the direction-based stopping criterion is significantly faster and uses a much smaller propagation volume, the **speed-based criterion is kept as the default**. The speed-based criterion is:
1. Simpler to formulate and trace (based directly on the vis-viva equation).
2. Physically better justified for defining absolute escape from the planetary gravity well (ensuring kinetic and potential energy ratios have fully stabilized to within $1e-8$).

The direction-based criterion remains supported as an optional mode for performance-sensitive validations.

## Conclusion
The measured angular discrepancies are extremely tiny (virtually zero, on the order of $10^{-6}$ degrees or less), confirming that the instantaneous Rodrigues rotation is mathematically exact relative to the two-body hyperbolic trajectory itself. 

However, we find a major physics-fidelity limitation: the derived stopping distance $r_{\text{stop}}$ (required for the speed to converge to $v_{\infty}$ within $1e-8$) is $10^{13}$ to $10^{14}$ km, which exceeds the Sphere of Influence (SOI) of the giant planets by many orders of magnitude. 

At the actual SOI boundary, the planet's gravitational pull is still substantial, meaning that the spacecraft's speed has **not** converged to the asymptotic $v_{\infty}$ value (for example, at Jupiter's SOI boundary, the spacecraft's excess speed is still $\sim 10.26$ km/s, which is a $2.6\%$ deviation from the $10.0$ km/s asymptote). Under a patched-conics model, we assume instant conversion to and from $v_{\infty}$ at the SOI boundaries. The fact that the physical velocity does not converge to its asymptotic value within the planetary SOI introduces a fundamental model fidelity limit when joining heliocentric legs to planetary flybys. 

Nevertheless, for the flyby itself, the closed-form Rodrigues rotation introduces zero internal angular error, meaning that the approximation is highly robust and does not introduce artificial errors. We recommend merging the implementation since the analytical model is structurally correct.

