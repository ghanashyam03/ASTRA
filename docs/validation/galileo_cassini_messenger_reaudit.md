# Re-audit: Galileo, Cassini, MESSENGER Against Corrected Fidelity Evidence

## Context
The original validation of Galileo, Cassini, and MESSENGER was performed using a simplified, closed-form, instantaneous patched-conics flyby model. This model assumes that celestial bodies are frozen in space during the spacecraft's passage through their sphere of influence (SOI).

However, recent investigations in Prompts 44 and 45 revealed a highly counter-intuitive physical trend: the inner solar system bodies (Venus, Earth, Mercury) exhibit the **largest planet-motion-during-passage ratios** of any celestial bodies in the solar system. This is due to their high orbital velocities and compact SOIs, resulting in a displacement ratio:
$$\text{ratio} = \frac{2 \cdot v_{\text{planet}}}{v_{\infty}}$$
which is largest at the inner planets (e.g., ~23.3 for Mercury, ~7.3 for Venus) and smallest for the outer planets (e.g., ~1.1 for Neptune).

To quantify how much this moving-planet effect shifts the state vector at the exit of the sphere of influence (compared to the frozen-planet patched-conics model), we performed three-body numerical propagations (Sun + planet gravity using time-varying SPICE ephemerides) and compared them directly against propagations where the planet's position is frozen at the periapsis epoch. 

Both propagations start from a shared state vector constructed at periapsis and run to the SOI boundary exit epoch. The resulting velocity deviation ($v_{\text{dev}}$) represents the **accumulated heliocentric velocity-vector difference at the SOI exit boundary**. Importantly, because this value is measured at the SOI exit boundary, it does not directly represent the final mission propellant $\Delta v$ penalty (which requires full multi-leg chain re-optimization to determine how much trajectory correction is needed), but it indicates a massive state divergence that the solver must resolve.


---

## Per-mission re-examination

### 1. Galileo (Venus, Earth, Earth)

*   **Closest Measured Case in Prompt 45:** 
    *   For the Earth flybys (Earth 1: $v_{\infty} = 8.95$ km/s, $r_p$ alt = 960 km; Earth 2: $v_{\infty} = 8.9$ km/s, $r_p$ alt = 303 km), the closest case measured in Prompt 45 is **Mars** ($v_{\infty} = 5.5$ km/s, $r_p$ alt = 400 km).
*   **Fidelity Deviations vs. Mission Limits:**
    *   **Mars Reference (P45):** Position deviation = $170,212.5$ km, velocity deviation = $1.663$ km/s.
    *   **Proportional Scaling Assumption:** Since Earth's gravity is larger than Mars's ($\mu_{\text{Earth}} \approx 9.3 \times \mu_{\text{Mars}}$) and the SOI is larger, we expect the integrated gravity perturbation to be significantly larger for Earth. Direct physical integration yields the following exact deviations:
        *   **Venus Flyby (1990-02-10):** Position deviation = **$206,379.6$ km**, velocity deviation = **$2.962$ km/s** (SOI crossing = $0.87$ days).
        *   **Earth 1 Flyby (1990-12-08):** Position deviation = **$287,137.5$ km**, velocity deviation = **$2.839$ km/s** (SOI crossing = $1.20$ days).
        *   **Earth 2 Flyby (1992-12-08):** Position deviation = **$312,553.7$ km**, velocity deviation = **$3.071$ km/s** (SOI crossing = $1.20$ days).
    *   **Comparison to Margins:**
        *   **Boundary State Divergence:** The actual three-body moving-planet propagation results in a heliocentric velocity-vector discrepancy at the SOI exit boundary of **$2.839\text{--}3.071$ km/s** and a spatial position shift of **$206,379.6\text{--}312,553.7$ km** compared to the patched-conics model.
        *   **Safety Altitude:** The position deviations (~$206,000$ to $312,000$ km) are thousands of times larger than the ASTRA clearance margins (`SAFE_FLYBY_ALTITUDE_KM` = $500$ km for Earth and $300$ km for Venus). 
        *   **Mission $\Delta v$ Context:** Direct comparison of these boundary state velocity deviations to the mission's budgeted propellant $\Delta v$ (e.g. Galileo's TCM/DSM budget of ~$0.125$ km/s) is not scientifically valid because the downstream targeting corrections have not yet been optimized under a moving-planet solver. However, a velocity-vector discrepancy of ~3 km/s at the exit of an early leg implies that the patched-conics solver is searching an entirely different state space than the real physics, which is a major gap.
*   **Confidence Change:** 
    *   This **reduces** confidence in the physical correctness of the original conics-based Galileo $\Delta v$ reproduction. While the conics solver converged/non-converged on its own mathematical terms, the resulting state trajectories deviate substantially from the moving-planet physics. However, the exact end-to-end effect on Galileo's mission $\Delta v$ has not yet been quantified.


---

### 2. Cassini (Venus, Venus, Earth)

*   **Closest Measured Case in Prompt 45:**
    *   For the Venus flybys (Venus 1: $v_{\infty} = 9.7$ km/s, $r_p$ alt = 284 km; Venus 2: $v_{\infty} = 9.7$ km/s, $r_p$ alt = 600 km), the closest case measured in Prompt 45 is **Mars** ($v_{\infty} = 5.5$ km/s, $r_p$ alt = 400 km).
*   **Fidelity Deviations vs. Mission Limits:**
    *   **Mars Reference (P45):** Position deviation = $170,212.5$ km, velocity deviation = $1.663$ km/s.
    *   **Proportional Scaling Assumption:** Venus is a much more massive perturber than Mars ($\mu_{\text{Venus}} \approx 7.6 \times \mu_{\text{Mars}}$). Consequently, we expect higher velocity perturbations. Direct physical integration yields:
        *   **Venus 1 Flyby (1998-04-26):** Position deviation = **$259,463.0$ km**, velocity deviation = **$4.228$ km/s** (SOI crossing = $0.74$ days).
        *   **Venus 2 Flyby (1999-06-24):** Position deviation = **$281,302.7$ km**, velocity deviation = **$4.592$ km/s** (SOI crossing = $0.74$ days).
        *   **Earth Flyby (1999-08-18):** Position deviation = **$439,807.3$ km**, velocity deviation = **$4.407$ km/s** (SOI crossing = $1.19$ days).
    *   **Comparison to Margins:**
        *   **Boundary State Divergence:** The heliocentric velocity-vector discrepancy at the SOI exit boundary is **$4.228\text{--}4.592$ km/s** (Venus) and **$4.407$ km/s** (Earth), with position shifts of **$259,463.0\text{--}439,807.3$ km**.
        *   **Safety Altitude:** The spatial deviations are orders of magnitude larger than the clearance margins of $300$ km (Venus) and $500$ km (Earth).
        *   **Mission $\Delta v$ Context:** Direct comparison to Cassini's benchmark DSM budget ($1.0$ km/s) is scientifically invalid since downstream corrections could leverage gravity assists to reduce the net penalty. Nonetheless, a ~4.5 km/s boundary state discrepancy makes the patched-conics trajectory unrepresentative of the real physical environment.
*   **Confidence Change:**
    *   This **reduces** confidence in the physical representativeness of the patched-conics validation. While conics-based trajectory optimization did not converge for Cassini VVE due to resonance limitations, the moving-planet state divergence indicates that a high-fidelity propagator is necessary to model flight-representative trajectories. The exact end-to-end impact on Cassini's mission $\Delta v$ remains unquantified.


---

### 3. MESSENGER (Earth, Venus, Venus, Mercury, Mercury, Mercury)

*   **Closest Measured Case in Prompt 45:**
    *   For the three Mercury flybys (Mercury 1: $v_{\infty} = 5.0$ km/s, $r_p$ alt = 201 km; Mercury 2: $v_{\infty} = 5.0$ km/s, $r_p$ alt = 201 km; Mercury 3: $v_{\infty} = 5.0$ km/s, $r_p$ alt = 228 km), the closest measured case is **Mercury** ($v_{\infty} = 5.0$ km/s, $r_p$ alt = 500 km).
*   **Fidelity Deviations vs. Mission Limits:**
    *   **Mercury Reference (P45):** Position deviation = **$24,092.8$ km**, velocity deviation = **$1.159$ km/s** (at $500$ km alt).
    *   **Proportional Scaling Assumption:** The actual MESSENGER Mercury flybys occurred at much lower altitudes (~$201$ to $228$ km) compared to the $500$ km test case. Because gravity scales as $1/r^2$, passing closer to the surface increases the gravitational acceleration and thus amplifies the integration error. Direct physical integration yields:
        *   **Mercury 1 Flyby (2008-01-14):** Position deviation = **$26,359.2$ km**, velocity deviation = **$1.261$ km/s** (SOI crossing = $0.26$ days).
        *   **Mercury 2 Flyby (2008-10-06):** Position deviation = **$27,470.2$ km**, velocity deviation = **$1.314$ km/s** (SOI crossing = $0.26$ days).
        *   **Mercury 3 Flyby (2009-09-29):** Position deviation = **$37,827.3$ km**, velocity deviation = **$1.803$ km/s** (SOI crossing = $0.26$ days).
    *   **Comparison to Margins:**
        *   **Boundary State Divergence:** The heliocentric velocity-vector discrepancy at the SOI exit boundary is **$1.261\text{--}1.803$ km/s** for the Mercury flybys, with position deviations of **$26,359.2\text{--}37,827.3$ km**.
        *   **Safety Altitude:** The spatial deviations ($26,000\text{--}37,800$ km) are more than 130 times the $200$ km safety clearance for Mercury.
        *   **Mission $\Delta v$ Context:** Comparing this directly to MESSENGER's total launch budget (~2.2 km/s) is not scientifically valid because boundary state errors are not the same as correction burns. However, since MESSENGER completes multiple consecutive Mercury flybys, the accumulation of these errors means the patched-conics solver operates in a substantially different state space than the physical environment.
*   **Confidence Change:**
    *   Confidence in conics-based trajectory representativeness is **reduced**. The compounding state discrepancies over consecutive encounters indicate that conics-based modeling is not representative of actual mission flight path without a high-fidelity numerical correction. The exact end-to-end impact on MESSENGER's mission $\Delta v$ has not yet been computed.


---

## Conclusion

The original validations should be re-run with the high-fidelity option once Prompt 47 builds it, because the measured state deviation at the SOI boundary is extremely large, suggesting that the patched-conics model operates in a substantially different trajectory state space than the real physics.

This recommendation is justified by the following findings:
1.  **State Vector Discrepancy:** The moving-planet model produces substantially different propagated states at the SOI exit, with heliocentric velocity-vector discrepancies of **$1.26\text{--}1.80$ km/s** (Mercury) and **$2.83\text{--}4.59$ km/s** (Earth/Venus), and position shifts of up to **$439,800$ km**.
2.  **Unquantified Mission $\Delta v$:** Since mission $\Delta v$ was not actually recomputed (only boundary state divergence was measured), the exact end-to-end effect of these physical deviations on mission-level propellant budgets has not yet been quantified. It is not scientifically valid to directly compare these boundary discrepancies to the mission DSM budgets.
3.  **Physical Representativeness:** The scale of the boundary deviations indicates that the instantaneous patched-conics model is physically inadequate for representing the real flight dynamics of Galileo, Cassini, and MESSENGER. Re-running these multi-flyby sequences with high-fidelity propagation and a proper time-varying ephemeris solver is necessary to assess the true propellant cost of the gravity assists.

