# Cost-Aware Evaluation

Directional accuracy alone can be misleading.

A signal can predict the next mid-price direction and still be unusable after spread, slippage, and minimum edge requirements. Cost-aware evaluation focuses only on rows where the transaction-cost label selects a tradable side.

The evaluator uses `cost_aware_direction`:

```text
1  long side clears cost
0  short side clears cost
-1 no side clears cost
```

Rows with `-1` are excluded from the scored sample and counted as abstentions.

This makes the metric closer to signal delivery:

- weak directional moves do not inflate accuracy
- abstention is visible as a sample-size cost
- LCRI can be compared against raw imbalance on tradable rows only

The output reports:

- rows
- abstained rows
- directional accuracy
- Brier score
- rank correlation

Research use cases:

- compare LCRI lift before and after transaction costs
- evaluate transmission pressure on tradable rows
- detect regimes where most apparent direction is not economically publishable
