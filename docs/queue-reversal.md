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

## Lead-lag coupling

`add_reversal_lead_lag_coupling` turns the pressure warning into a temporal
stress feature. It asks whether transmitted pressure on this row is followed by
same-direction queue reversal pressure on the next row:

```text
aligned = sign(transmission_pressure[t]) * queue_reversal_pressure[t + 1]
coupling = max(aligned, 0) * queue_reversal_risk[t + 1] / (1 + abs(transmission_pressure[t]))
```

Positive coupling means the pressure made it through the shadow-absorption stack
and was echoed by future queue snapback. That is stronger evidence than a static
reversal flag because it links transmitted pressure to the next fragile queue
state.

## Combined fracture/reversal release gate

`fracture_reversal_release_gate` combines two independent failure channels:

- calibration fracture pressure: LCRI probability shape is broken in a quantile
- reversal stress concentration: lead-lag queue snapback clusters inside a regime

The gate blocks only when the calibration fracture is reinforced by concentrated
reversal stress. A lone fracture or lone reversal concentration is escalated to
review, not block, because it may still be a harmless artifact.

## Transition gate diagnostics

`reversal_transition_gate_diagnostics` localizes the combined gate at regime
boundaries. It groups rows where `regime_changed` is true, or where the regime
label changes from the prior row, into transition labels such as `thick->thin`.
For each transition it reports:

- total, mean, and peak reversal lead-lag coupling
- share of all transition reversal stress
- the run-level combined release decision
- a transition-level `review` flag when an active combined gate is dominated by
  that boundary

This makes the artifact more publishable: reviewers can see whether a blocked or
reviewed release is caused by a specific liquidity-state handoff rather than a
diffuse full-sample average.
