# Queue Reversal Risk

Queue reversal risk is a pressure fragility feature.

The premise is that residual imbalance can become most dangerous when the visible signal is obvious but poorly supported. A large LCRI print with weak transmission, opposite pressure memory, and a hollow book is more likely to snap back than a smaller residual that is persistent and transmitted.

The feature combines three fragility terms:

```text
memory_disagreement = max(-(sign(lcri) * pressure_memory), 0)
transmission_gap    = max(abs(lcri) - abs(transmission_pressure), 0)
queue_reversal_risk = memory_disagreement + normalized_gap + liquidity_void_ratio
```

The signed `queue_reversal_pressure` points opposite the current residual pressure. It is not a forecast by itself. It is a warning that the current pressure state may be crowded, unsupported, or vulnerable to passive absorption.

Research use:

- down-weight publishable signals with high reversal risk
- split LCRI performance by reversal flag
- inspect whether void-heavy books produce faster pressure decay
