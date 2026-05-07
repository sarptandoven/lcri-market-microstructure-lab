# Pressure Memory

Liquidity pressure is more useful when it persists.

A single LCRI spike can be a transient queue refresh, a stale displayed book, or a short-lived sweep. A persistent residual is different. It says the current book keeps showing abnormal pressure even after the liquidity baseline has been removed.

Pressure memory adds a small time-series layer on top of scored snapshots:

```text
pressure_memory = ewma(lcri)
fracture_memory = ewma(imbalance_fracture)
pressure_decay_risk = abs(lcri - pressure_memory) / (1 + abs(lcri))
latent_liquidity_fracture = abs(pressure_memory) * abs(fracture_memory)
pressure_memory_half_life = bars since local memory peak decayed by 50%
pressure_memory_release_velocity = (1 - decay_ratio) / half_life on decay events
pressure_memory_decay_state = inactive | persistent | fast_decay | slow_decay
```

Half-life state is a compact artifact label. Fast decay marks transient pressure that loses force quickly. Slow decay marks observed memory release only after a longer local window, which is closer to latent-liquidity stress than simple mean reversion. Release velocity keeps the same event sparse but distinguishes abrupt memory collapse from slower leakage. `pressure_memory_decay_summary` aggregates state share, event rate, mean half-life, mean release velocity, and latent fracture exposure for artifact-level comparison.

`classify_pressure_memory_artifacts` turns those state/exposure shapes into a small review catalog. A fast release with elevated latent fracture is not just mean reversion, it can indicate pressure has moved from visible imbalance into hidden ladder fragility. Persistent states with high fracture exposure are separated from benign decay so review workflows can prioritize adverse-selection phase shifts instead of treating all pressure relaxation as healthy.

`hidden_resiliency_asymmetry_summary` is the report-level guardrail for this failure mode. It compares fast-decay fracture against slow/persistent fracture and scales positive gaps by the fast-vs-slow release-velocity advantage. A positive score means apparent resiliency is asymmetric: displayed pressure unloaded quickly, but latent fracture stayed higher in the fast-release bucket. That is a warning against calling the event healthy purely because pressure memory decayed.

Artifact families:

- `fractured_fast_release`: visible pressure clears quickly while fracture stays elevated
- `latent_fracture_persistence`: pressure memory remains persistent in a fractured book
- `sticky_fracture_decay`: slow pressure release leaves elevated latent fracture behind
- `benign_decay`: release or persistence without elevated fracture exposure

The research target is not to replace LCRI. It is to separate durable residual pressure from one-shot dislocations. Latent fracture is deliberately multiplicative: high pressure memory with a calm book should not look like the same state as high memory while the ladder remains fractured.

Useful hypotheses:

- persistent positive pressure with low decay risk should behave differently from a single positive print
- pressure aligned with persistent fracture should be more fragile than pressure in a coherent book
- high decay risk should reduce confidence in publishable-edge gates
