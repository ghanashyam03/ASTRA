# Mandatory Neural Surrogate Audit Checklist

Any update introducing or modifying a neural surrogate used inside an
optimizer MUST run it through `run_surrogate_audit()` and include the
resulting report in its verification section before being considered for
merge. This checklist exists because three of the four neural-acceleration
attempts in ASTRA's history failed under independent audit despite passing
unit tests — this template is how that audit happens automatically instead
of depending on someone remembering to do it by hand.

- [ ] Absolute accuracy reported (MAE, RMSE, fraction within 1 km/s) —
      disclosed even if poor. A poor regressor is not disqualifying by
      itself; an UNDISCLOSED poor regressor is.
- [ ] Ranking quality reported (Spearman, Kendall) — this is what actually
      matter if the surrogate is used for warm-starting or pruning, not
      absolute accuracy.
- [ ] Multi-seed comparison against the no-surrogate baseline on a real
      benchmark mission — single-seed results are not sufficient evidence.
- [ ] The surrogate's role in the optimizer is restricted to what its audit
      actually supports — a poor regressor must not be used to report
      absolute Δv values to a user; it may be used for ranking/ordering only.
- [ ] If the surrogate touches multi-leg flyby geometry, P41's audit layer
      (audit_trajectory_physics) must also pass on every trajectory the
      surrogate-assisted optimizer produces.
