# Liquidity Fracture Feature Family

LCRI removes expected imbalance under the current liquidity state. The next useful question is whether the visible book is internally coherent.

A normal imbalance can come from a stable depth profile. It can also come from a fragile book where the best level is heavy but the levels behind it are hollow, or where the top-of-book signal disagrees with the full visible book. Those cases should not share the same baseline.

This project adds a liquidity fracture feature family to make that distinction explicit.

## Features

### Imbalance fracture

```text
imbalance_fracture = top_imbalance - raw_imbalance
```

This measures disagreement between the touch and the full visible ladder. A high absolute value means the top level is telling a different story than aggregate depth.

### Liquidity void ratio

```text
liquidity_void_ratio = max(max(depth_l - depth_{l+1}, 0)) / total_depth
```

This detects cliffs behind the touch. A book with large displayed size at the best level but weak backing depth should be treated as more fragile than a smoothly layered book.

### Depth convexity

```text
depth_convexity = (depth_1 + depth_L - 2 * depth_mid) / total_depth
```

This captures whether depth is concentrated at the extremes of the visible ladder rather than distributed through the middle levels.

### Resilience asymmetry

```text
resilience_asymmetry = queue_pressure * spread_ticks / (1 + replenishment_rate)
```

This is a signed pressure score amplified when spreads are wider and replenishment is weaker. It separates pressure that is likely to be absorbed from pressure that may propagate into price movement.

## Why this belongs in the baseline

These are not standalone alpha claims. They describe the market state under which raw imbalance is observed. Feeding them into the liquidity-conditioned baseline lets the residual isolate abnormal pressure after controlling for book fragility.

That keeps the core LCRI philosophy intact:

```text
observed imbalance - expected imbalance under local liquidity conditions
```

The local liquidity condition now includes fracture structure, not only spread, volatility, total depth, and replenishment.

## Synthetic sanity check

On a 5,000-row simulation with seed 31, the upgraded LCRI model produced this lift over raw imbalance:

```text
directional_accuracy_lift: 0.1336
brier_score_reduction:    0.0428
rank_correlation_lift:    0.0732
```

This is not evidence of live-market edge. It is a controlled sanity check that the fracture variables integrate cleanly into the existing simulator and preserve the expected behavior of LCRI: raw imbalance is less useful than residual pressure after conditioning on liquidity state.

## Artifact compatibility

Model artifact schema version is now `2` because fitted feature names changed. Versioned load checks reject old or incompatible artifacts instead of silently scoring with the wrong design matrix.
