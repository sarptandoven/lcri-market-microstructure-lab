# Transmission Pressure Evaluation

Shadow absorption creates a second pressure series: `transmission_pressure`.

LCRI measures residual pressure after removing the local liquidity baseline. Transmission pressure applies an absorption haircut to that residual. The research question is whether the haircut removes false pressure without destroying useful directional information.

```python
evaluate_signals(scored, signals=["lcri", "transmission_pressure"])
```

`compare_transmission_signal` reports directional accuracy, Brier score, and rank-correlation deltas against LCRI.
