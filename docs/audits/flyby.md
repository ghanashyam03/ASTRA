# Physics Validation Audit: Earth-Venus-Mars 2032 Flyby

This document provides a rigorous physical audit of the optimized Earth-Venus-Mars 2032 flyby trajectory reported by the `flyby` optimization strategy. It evaluates the physical feasibility of the gravity assist at Venus by comparing the required orbital deflection against the maximum turn angle permitted by Venusian gravity.

---

## 1. Parameters and Vector Audit

The optimized trajectory has the following parameters at the Venus flyby epoch ($t_{\text{flyby}} = 1048958759.41$ J2000 seconds, corresponding to **2033-03-29 11:25:59 UTC**):

| Metric | Value | Equation / Source |
| :--- | :--- | :--- |
| **Incoming Excess Velocity ($v_{\infty, \text{in}}$)** | $10.0035 \text{ km/s}$ | $\|v_{\text{sc, arr}} - v_{\text{venus}}\|$ |
| **Outgoing Excess Velocity ($v_{\infty, \text{out}}$)** | $9.9269 \text{ km/s}$ | $\|v_{\text{sc, dep}} - v_{\text{venus}}\|$ |
| **Required Turn Angle ($\theta_{\text{req}}$)** | $156.85^{\circ}$ | $\arccos\left(\frac{v_{\infty, \text{in}} \cdot v_{\infty, \text{out}}}{\|v_{\infty, \text{in}}\| \|v_{\infty, \text{out}}\|}\right)$ |
| **Optimized Periapsis Altitude ($h_p$)** | $5,268.53 \text{ km}$ | Free optimization variable |
| **Venus Physical Radius ($R_{\text{venus}}$)** | $6,051.80 \text{ km}$ | Physical constant |
| **Venusian Gravitational Parameter ($\mu_{\text{venus}}$)** | $324,859.0 \text{ km}^3/\text{s}^2$ | GM constant |

---

## 2. Feasibility Calculations

### A. Turn Angle Feasibility
For a hyperbolic flyby, the deflection angle $\delta$ (total turn angle of the excess velocity vector) is related to the eccentricity $e$ of the flyby hyperbola by:
$$\delta = 2 \arcsin\left(\frac{1}{e}\right)$$

where eccentricity $e$ is determined by the periapsis radius $r_p = R_{\text{venus}} + h_p$ and the incoming speed $v_{\infty, \text{in}}$:
$$e = 1 + \frac{r_p v_{\infty, \text{in}}^2}{\mu_{\text{venus}}}$$

Using the optimized parameters:
$$r_p = 6051.80 \text{ km} + 5268.53 \text{ km} = 11,320.33 \text{ km}$$
$$e = 1 + \frac{11,320.33 \times 10.0035^2}{324,859.0} = 1 + \frac{1,132,828.62}{324,859.0} \approx 4.4874$$
$$\delta_{\text{achievable}} = 2 \arcsin\left(\frac{1}{4.4874}\right) = 2 \arcsin(0.22285) \approx 25.75^{\circ}$$

* **Required Turn Angle ($\theta_{\text{req}}$):** $156.85^{\circ}$
* **Maximum Achievable Turn Angle ($\delta_{\text{achievable}}$):** $25.75^{\circ}$
* **Comparison:** $\theta_{\text{req}} \le \delta_{\text{achievable}}$ is **False** ($156.85^{\circ} > 25.75^{\circ}$).

### B. Energy Conservation check ($v_{\infty}$ magnitude)
For a passive, unpowered flyby, the incoming excess speed must equal the outgoing excess speed in the planet's reference frame:
$$\|v_{\infty, \text{in}}\| = \|v_{\infty, \text{out}}\|$$

* **Incoming Speed:** $10.0035 \text{ km/s}$
* **Outgoing Speed:** $9.9269 \text{ km/s}$
* **Difference:** $0.0766 \text{ km/s}$ (energy is not conserved, though the difference is small).

### C. Implied Vector Correction $\Delta v$
Since the incoming and outgoing asymptotes are misaligned, a vector correction burn is required to transition the spacecraft from the incoming hyperbolic trajectory to the outgoing trajectory. The minimum impulsive burn at the patch point (center of Venus's SOI) is:
$$\Delta v_{\text{correction}} = \|v_{\infty, \text{out}} - v_{\infty, \text{in}}\| = 19.525 \text{ km/s}$$

---

## 3. Physical Achievability Analysis

The trajectory solver reports a total mission $\Delta v$ of **$8.8799 \text{ km/s}$** (consisting of $4.2884 \text{ km/s}$ at Earth departure and $4.5915 \text{ km/s}$ at Mars capture) under the assumption that the Venus flyby is unpowered and passive ($\Delta v_{\text{fly}} = 0.0$).

However:
1. To achieve a deflection of $156.85^{\circ}$ purely by gravity with an incoming speed of $10.0035 \text{ km/s}$, the required periapsis radius would be:
   $$e_{\text{needed}} = \frac{1}{\sin(156.85^{\circ}/2)} \approx 1.0207$$
   $$r_p = \frac{\mu_{\text{venus}} (e_{\text{needed}} - 1)}{v_{\infty, \text{in}}^2} = \frac{324,859.0 \times 0.0207}{10.0035^2} \approx 67.24 \text{ km}$$
2. A periapsis radius of $67.24 \text{ km}$ lies deep inside the interior of Venus (physical radius $6,051.80 \text{ km}$). The spacecraft would have to fly **$5,984.56 \text{ km}$ below the surface** of Venus, which is physically impossible.

---

## 4. Final Verdict

### **[VERDICT] C) Physically impossible**

The reported Earth-Venus-Mars trajectory is **physically impossible** as a passive gravity assist. The deflection angle required to connect the two independent Lambert transfer arcs exceeds the maximum achievable deflection angle of Venusian gravity at the specified altitude by **$131.10^{\circ}$**. Attempting this flyby passively would result in the spacecraft hitting the planet or escaping on a completely incorrect trajectory. 

To fly this mission path, the spacecraft would require a massive active correction burn of **$19.525 \text{ km/s}$** at Venus, raising the total mission $\Delta v$ from $8.8799 \text{ km/s}$ to **$28.405 \text{ km/s}$**.
