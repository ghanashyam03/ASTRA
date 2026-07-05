# ASTRA CR3BP Propagator Design Specification

## 1. Scope and Motivation

The Circular Restricted Three-Body Problem (CR3BP) models the motion of a spacecraft under the simultaneous gravitational influence of two massive bodies (e.g., Sun and planet) that orbit their common barycentre in circular orbits. Unlike patched-conics, the CR3BP applies the correct gravitational forces continuously throughout the trajectory — including inside the smaller body's SOI — making it the appropriate model for:
- High-fidelity flyby trajectory computation at inner planets (Venus, Earth, Mercury)
- Resonant transfer design (Galileo VEEGA, MESSENGER)
- Lagrange-point manifold exploitation (lunar free-return, Jupiter Trojan access)

---

## 2. Reference Frame

The CR3BP rotating frame rotates with the primaries ($P_1$ = Sun, $P_2$ = planet).
- **Origin**: The barycentre of the two-body system.
- **x-axis**: Points from $P_1$ to $P_2$.
- **z-axis**: The direction of orbital angular momentum (ecliptic north = $+z$ in ECLIPJ2000).
- **y-axis**: Completes the right-handed orthonormal basis ($\mathbf{u}_y = \mathbf{u}_z \times \mathbf{u}_x$).

Units are non-dimensionalized using the following base quantities:
- **Length unit**: $L^*$ = distance between $P_1$ and $P_2$ (semi-major axis of the planet's orbit).
- **Time unit**: $T^* = 1/n$, where $n = \sqrt{G(m_1 + m_2)/{L^*}^3}$ is the mean motion.
- **Mass unit**: $M^* = m_1 + m_2$.

Under this normalization, the universal gravitational constant $G = 1$, the sum of masses $m_1 + m_2 = 1$, and the mean motion $n = 1$.

---

## 3. Mass Parameter

The dimensionless mass parameter is defined as:
$$\mu^* = \frac{m_2}{m_1 + m_2}$$

Consequently, the non-dimensional mass of the primary $P_1$ (Sun) is $1 - \mu^*$, located at $(-\mu^*, 0, 0)$, and the mass of the secondary $P_2$ (planet) is $\mu^*$, located at $(1 - \mu^*, 0, 0)$.

Pre-computed values (derived from DE440 masses):

| Body     | a [AU]  | $\mu^*$ (Sun + Planet) |
|----------|---------|----------------------|
| Mercury  | 0.38710 | 1.6601e-7            |
| Venus    | 0.72333 | 2.4478e-6            |
| Earth    | 1.00000 | 3.0034e-6            |
| Mars     | 1.52366 | 3.2271e-7            |
| Jupiter  | 5.20336 | 9.5370e-4            |
| Saturn   | 9.53707 | 2.8578e-4            |
| Uranus   | 19.1913 | 4.3667e-5            |
| Neptune  | 30.0690 | 5.1503e-5            |

---

## 4. Equations of Motion

The state vector of the spacecraft in the rotating frame is $\mathbf{q} = [x, y, z, \dot{x}, \dot{y}, \dot{z}]^T$.

The equations of motion are:
$$\ddot{x} - 2\dot{y} = \frac{\partial \Omega}{\partial x}$$
$$\ddot{y} + 2\dot{x} = \frac{\partial \Omega}{\partial y}$$
$$\ddot{z} = \frac{\partial \Omega}{\partial z}$$

where the effective potential is:
$$\Omega(x,y,z) = \frac{1}{2}(x^2 + y^2) + \frac{1 - \mu^*}{r_1} + \frac{\mu^*}{r_2}$$

and the distances to the primaries are:
- Distance from $P_1$ (Sun): $r_1 = \sqrt{(x + \mu^*)^2 + y^2 + z^2}$
- Distance from $P_2$ (planet): $r_2 = \sqrt{(x - 1 + \mu^*)^2 + y^2 + z^2}$

Expanding the partial derivatives yields:
$$\ddot{x} = 2\dot{y} + x - \frac{(1 - \mu^*)(x + \mu^*)}{r_1^3} - \frac{\mu^*(x - 1 + \mu^*)}{r_2^3}$$
$$\ddot{y} = -2\dot{x} + y - \frac{(1 - \mu^*)y}{r_1^3} - \frac{\mu^* y}{r_2^3}$$
$$\ddot{z} = -\frac{(1 - \mu^*)z}{r_1^3} - \frac{\mu^* z}{r_2^3}$$

---

## 5. Lagrange Point Positions (Non-Dimensional, z=0)

The five equilibrium points (Lagrange points) lie in the $xy$-plane ($z=0$):
- **$L_1$** (between $P_1$ and $P_2$): $x \approx 1 - \left(\frac{\mu^*}{3}\right)^{1/3}$
- **$L_2$** (beyond $P_2$): $x \approx 1 + \left(\frac{\mu^*}{3}\right)^{1/3}$
- **$L_3$** (opposite $P_2$): $x \approx -1 - \frac{5}{12}\mu^*$
- **$L_4$** ($+60^\circ$ from $P_2$): $x = \frac{1}{2} - \mu^*$, $y = +\frac{\sqrt{3}}{2}$
- **$L_5$** ($-60^\circ$ from $P_2$): $x = \frac{1}{2} - \mu^*$, $y = -\frac{\sqrt{3}}{2}$

---

## 6. Non-Dimensionalization / Re-Dimensionalization

Conversion equations between dimensional (km, s) and non-dimensional units:

### TO Non-Dimensional:
$$x_{nd} = \frac{x_{km}}{L^*}$$
$$t_{nd} = t_s \cdot n$$
$$v_{nd} = \frac{v_{km/s}}{L^* \cdot n}$$

### FROM Non-Dimensional:
$$x_{km} = x_{nd} \cdot L^*$$
$$t_s = \frac{t_{nd}}{n}$$
$$v_{km/s} = v_{nd} \cdot (L^* \cdot n)$$

---

## 7. Rotation from Inertial (ECLIPJ2000) to CR3BP Rotating Frame

Let $R(t)$ be the coordinate rotation matrix from the inertial ECLIPJ2000 frame to the rotating frame.

### 7.1 Analytical Circular Model
In the idealized model, $P_2$ orbits the barycentre in the inertial plane with angular velocity $n$ (mean motion). The rotation angle is $\theta = n \cdot t$ (measured from the alignment epoch):
$$R(t) = \begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### 7.2 Ephemeris-Based Dynamic Model
When using actual ephemerides, the position $\mathbf{r}_{planet}$ and velocity $\mathbf{v}_{planet}$ of the planet relative to the Sun are retrieved at epoch $t$. The rotating frame's basis vectors are:
$$\mathbf{u}_x = \frac{\mathbf{r}_{planet}}{\|\mathbf{r}_{planet}\|}$$
$$\mathbf{u}_z = \frac{\mathbf{r}_{planet} \times \mathbf{v}_{planet}}{\|\mathbf{r}_{planet} \times \mathbf{v}_{planet}\|}$$
$$\mathbf{u}_y = \mathbf{u}_z \times \mathbf{u}_x$$

The rotation matrix is:
$$R(t) = \begin{bmatrix} \mathbf{u}_x^T \\ \mathbf{u}_y^T \\ \mathbf{u}_z^T \end{bmatrix}$$

### 7.3 State Transformations
Let $\mathbf{r}_{inertial}$ and $\mathbf{v}_{inertial}$ be the spacecraft position and velocity relative to the primary ($P_1$ = Sun).
Let $\boldsymbol{\omega} = [0, 0, n]^T$ be the angular velocity vector of the rotating frame.

#### Forward Transformation (Inertial to Rotating Non-Dimensional):
1. **Translate to Barycentre**:
   $$\mathbf{r}_{bary} = \mathbf{r}_{inertial} - \mu^* \mathbf{r}_{planet}$$
   $$\mathbf{v}_{bary} = \mathbf{v}_{inertial} - \mu^* \mathbf{v}_{planet}$$
2. **Rotate to Rotating Frame**:
   $$\mathbf{r}_{rot} = R(t) \mathbf{r}_{bary}$$
   $$\mathbf{v}_{rot} = R(t) \mathbf{v}_{bary} - \boldsymbol{\omega} \times \mathbf{r}_{rot}$$
3. **Non-Dimensionalize**:
   $$\mathbf{r}_{nd} = \frac{\mathbf{r}_{rot}}{L^*}$$
   $$\mathbf{v}_{nd} = \frac{\mathbf{v}_{rot}}{L^* \cdot n}$$

#### Inverse Transformation (Rotating Non-Dimensional to Inertial):
1. **Dimensionalize**:
   $$\mathbf{r}_{rot} = \mathbf{r}_{nd} \cdot L^*$$
   $$\mathbf{v}_{rot} = \mathbf{v}_{nd} \cdot (L^* \cdot n)$$
2. **Rotate back to Inertial**:
   $$\mathbf{r}_{bary} = R^T(t) \mathbf{r}_{rot}$$
   $$\mathbf{v}_{bary} = R^T(t) (\mathbf{v}_{rot} + \boldsymbol{\omega} \times \mathbf{r}_{rot})$$
3. **Translate back to Sun-Centered**:
   $$\mathbf{r}_{inertial} = \mathbf{r}_{bary} + \mu^* \mathbf{r}_{planet}$$
   $$\mathbf{v}_{inertial} = \mathbf{v}_{bary} + \mu^* \mathbf{v}_{planet}$$

---

## 8. Python Interface (to be implemented in P53)

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class CR3BPSystem:
    body: str            # planet name (e.g. "VENUS")
    mu_star: float       # mass parameter (dimensionless)
    L_star_km: float     # length unit [km]
    n_rad_per_s: float   # mean motion [rad/s]
    T_star_s: float      # time unit [s]


def cr3bp_eom(t: float, q: np.ndarray, mu: float) -> np.ndarray:
    """CR3BP equations of motion. Returns dq/dt.
    
    Parameters
    ----------
    t : float
        Dimensionless time.
    q : np.ndarray
        Dimensionless state vector [x, y, z, vx, vy, vz].
    mu : float
        Dimensionless mass parameter.
        
    Returns
    -------
    np.ndarray
        Derivative state vector dq/dt.
    """
    ...


def propagate_cr3bp(
    system: CR3BPSystem,
    q0: np.ndarray,          # initial state in rotating frame (non-dimensional)
    t_span: tuple[float, float],  # integration time span (non-dimensional)
    t_eval: np.ndarray | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate CR3BP EOM. Returns (t_array, q_array)."""
    ...


def nondimensionalize_state(
    pos_km: np.ndarray, vel_km_s: np.ndarray,
    epoch_s: float, system: CR3BPSystem, planet_pos_km: np.ndarray
) -> np.ndarray:
    """Convert inertial ECLIPJ2000 state to rotating CR3BP non-dimensional state.
    
    Note: Both pos_km and vel_km_s are Sun-centered inertial states.
    planet_pos_km is the planet's position relative to the Sun.
    """
    ...


def dimensionalize_state(
    q_nd: np.ndarray, epoch_s: float, system: CR3BPSystem, planet_pos_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert rotating CR3BP state back to inertial km and km/s relative to the Sun."""
    ...
```
