# Pressure Memory

Liquidity pressure is more useful when it persists.

A single LCRI spike can be a transient queue refresh, a stale displayed book, or a short-lived sweep. A persistent residual is different. It says the current book keeps showing abnormal pressure even after the liquidity baseline has been removed.

Pressure memory adds a small time-series layer on top of scored snapshots:

```text
pressure_memory = ewma(lcri)
fracture_memory = ewma(imbalance_fracture)
pressure_decay_risk = abs(lcri - pressure_memory) / (1 + abs(lcri))
pressure_memory_half_life = bars since local memory peak decayed by 50%
pressure_memory_decay_state = inactive | persistent | fast_decay | slow_decay
```

Half-life state is a compact artifact label. Fast decay marks transient pressure that loses force quickly. Slow decay marks observed memory release only after a longer local window, which is closer to latent-liquidity stress than simple mean reversion.

The research target is not to replace LCRI. It is to separate durable residual pressure from one-shot dislocations.

Useful hypotheses:

- persistent positive pressure with low decay risk should behave differently from a single positive print
- pressure aligned with persistent fracture should be more fragile than pressure in a coherent book
- high decay risk should reduce confidence in publishable-edge gates
