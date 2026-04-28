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
