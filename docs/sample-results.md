# Sample Results

The default demo uses 20,000 synthetic order book snapshots across thick, thin, stressed, and replenishing liquidity regimes.

Example command:

```bash
lcri-lab run-demo --rows 20000 --seed 7
```

Representative output:

| signal | directional accuracy | Brier score | rank correlation |
| --- | ---: | ---: | ---: |
| raw_imbalance | 0.40395 | 0.320357 | 0.024351 |
| lcri | 0.56455 | 0.265111 | 0.186961 |

The important point is not the absolute number. The controlled setup creates structural order book bias from liquidity state, then evaluates whether subtracting a liquidity-conditioned baseline recovers more useful pressure information.

In this run, LCRI improves directional accuracy, Brier score, and rank correlation versus raw imbalance.


## Cost-aware evaluation

The research workflow now separates direction from tradability.

A move can have the correct sign but still fail after spread and slippage.
Cost-aware evaluation uses `cost_aware_direction` so weak moves become
abstentions instead of hidden false positives.

This is the right comparison point for publishable signal delivery.

## Transition-conditioned evaluation

Liquidity regime changes are evaluated separately from stable periods.
`transition_conditioned_metrics` compares signal quality on rows where
`regime_changed` is active versus rows where the regime is stable.
`transition_signal_lift` condenses the same split into LCRI lift over raw
imbalance for accuracy, Brier score, and rank correlation.
`transition_robustness_summary` then checks whether directional accuracy lift
survives in both stable and transitioning periods.

This isolates whether LCRI remains useful during microstructure state changes,
where static imbalance baselines are most likely to leak regime structure.

The demo also writes heldout transition metrics and heldout transition lift.
Those files apply the same transition split only to rows excluded from fitting.
They are the safer reference when checking whether transition robustness is
surviving outside the model's training sample.
Use these files before citing transition behavior as a heldout result.
The heldout transition chart provides the same check visually for quick review.
