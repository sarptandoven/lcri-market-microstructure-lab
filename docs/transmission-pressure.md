# Transmission Pressure Evaluation

Shadow absorption creates a second pressure series: `transmission_pressure`.

LCRI measures residual pressure after removing the local liquidity baseline. Transmission pressure applies an absorption haircut to that residual. The research question is whether the haircut removes false pressure without destroying useful directional information.

```python
evaluate_signals(scored, signals=["lcri", "transmission_pressure"])
```

`compare_transmission_signal` reports directional accuracy, Brier score, and rank-correlation deltas against LCRI.

## Calibration-monotonicity fracture pressure

Directional lift is not enough for a publishable market-microstructure signal.
The score must also remain calibrated and order-preserving across latent
liquidity states. `calibration_monotonicity_pressure` aligns calibration bins
with signal quantiles by ordinal rank and highlights buckets where a negative
observed-frequency slope coincides with calibration residuals.

Rows labelled `fractured_miscalibrated` are the most suspicious: the model is
not only inverted locally, it is also mispricing the same score region. Treat
those rows as candidates for latent liquidity fracture review before promoting
transmission pressure or LCRI into a release gate.

`calibration_monotonicity_pressure_summary` converts the row-level table into a
compact gate: any miscalibrated fracture blocks the fracture-pressure gate,
while shape-only fractures remain visible as review debt.
