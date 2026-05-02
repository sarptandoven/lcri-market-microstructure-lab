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
The heldout calibration curve applies the probability calibration check only to
rows excluded from fitting.
The generated summary includes `generalization_gap.csv` so full-sample and
heldout signal quality can be compared from the main report.
The generalization gap chart shows the same comparison as a quick visual check.
Use it to spot signals whose full-sample lift does not survive heldout scoring.
The summary also includes regime-level generalization gaps so degradation can be
checked inside each liquidity condition.
That table is useful when the headline heldout result is stable but one regime
shows clear metric decay. The companion regime gap chart focuses on directional
accuracy gaps for quick visual review.
Positive bars indicate the full-sample score is higher than the heldout score.
Negative bars indicate heldout scoring matched or exceeded the full-sample result.
Read it with `regime_generalization_gap.csv` for exact metric values.
The summary also includes transition-level generalization gaps so stable and
transition segments can be checked against heldout scoring. The transition gap
chart highlights whether degradation concentrates around regime changes.
Positive bars indicate stronger full-sample performance than heldout performance.
Negative bars indicate heldout performance matched or exceeded the full-sample score.
Use `transition_generalization_gap.csv` when exact metric values are needed.
This keeps the visual artifact tied to auditable numeric output.
The markdown summary mirrors `generalization_overview.json` for a compact audit
of the generated generalization gap artifacts. It also includes the ranked
leaderboard so the largest heldout degradations are visible without opening CSVs.
`lcri_generalization_gap_leaderboard.csv` applies the same ranking after filtering
to LCRI rows only, which makes the residual signal's weakest heldout contexts
visible without scanning raw imbalance rows. `lcri_generalization_scope_summary.csv`
rolls those LCRI rows up by signal, regime, and transition scope so reviewers can
see whether degradation is broad or localized. These tables are intended as the
first triage view when reviewing a generated demo run.

`lcri_generalization_gap_delta.csv` compares raw imbalance and LCRI degradation
across signal, regime, and transition scopes. Positive `raw_minus_lcri` values
mean LCRI lost less directional accuracy than raw imbalance on heldout rows.
Negative values mark scopes where LCRI degraded more and should be inspected
before treating the residual signal as robust. `lcri_gap_delta_flags.csv` adds a
plain stability label to each row for quick filtering. The matching chart shows
the same comparison visually, while `lcri_gap_delta_summary.json` records row
counts and the strongest stability and instability contexts for automated report
checks.
