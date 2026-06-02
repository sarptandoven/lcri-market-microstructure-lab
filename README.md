# LCRI Market Microstructure Lab

A research package for liquidity-conditioned residual imbalance in limit order books.

Raw order book imbalance is usually measured as:

```text
bid_depth - ask_depth
---------------------
bid_depth + ask_depth
```

That ratio is incomplete by itself. The same imbalance can have different information content depending on spread, total depth, replenishment, volatility, the current liquidity regime, and whether the visible book is internally coherent. This package estimates the expected imbalance under local liquidity conditions, then scores the residual pressure that remains after the local baseline is removed.

## Model

Let `I_t` be raw imbalance and `X_t` be local liquidity-state features. The baseline imbalance is:

```text
E[I_t | X_t]
```

The liquidity-conditioned residual imbalance is:

```text
LCRI_t = (I_t - E[I_t | X_t]) / sigma_regime
```

where `sigma_regime` is the residual scale estimated inside comparable liquidity regimes.

Positive LCRI means bid-side pressure is high relative to the current liquidity baseline. Negative LCRI means ask-side pressure is high relative to the current liquidity baseline.

The default baseline remains ridge-regularized and inspectable, but its design
matrix includes nonlinear liquidity stress terms (`spread_ticks²`,
`volatility²`, `liquidity_void_ratio × volatility`, and inverse replenishment)
so convex book-stress effects are removed before residual pressure is scored.
`baseline_rolling_basis_comparison` audits that neutralization across rolling
chronological train/test blocks, while `baseline_rolling_basis_summary` turns
fold-level lift, winner rate, and overfit ratios into a compact publishability
check so nonlinear lift is not inferred from a single favorable holdout split.

## Inputs

The model expects order book snapshots with these columns for each level from 1 to 5 by default:

```text
bid_px_1, bid_sz_1, ask_px_1, ask_sz_1
...
bid_px_5, bid_sz_5, ask_px_5, ask_sz_5
```

Additional required state columns:

```text
timestamp
regime
mid
next_mid
spread
spread_ticks
volatility
replenishment_rate
```

The included simulator generates this schema. Real market data can be scored after being normalized into the same snapshot format.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
lcri-lab run-demo --rows 20000 --seed 7 --train-frac 0.70 --passive-fill-horizon 2
pytest -q
ruff check .
```

Generated artifacts:

```text
reports/
  lcri-model.json
  figures/
    raw_vs_lcri_scatter.png
    regime_signal_quality.png
    transition_signal_quality.png
    heldout_transition_signal_quality.png
    calibration_curve.png
    heldout_calibration_curve.png
    generalization_gap.png
    regime_generalization_gap.png
    transition_generalization_gap.png
    lcri_generalization_gap_delta.png
    lcri_generalization_severity_by_scope.png
    lcri_gap_delta_scope_summary.png
  metrics.csv
  heldout_metrics.csv
  generalization_gap.csv
  regime_metrics.csv
  heldout_regime_metrics.csv
  regime_generalization_gap.csv
  transition_metrics.csv
  heldout_transition_metrics.csv
  transition_generalization_gap.csv
  generalization_overview.json
  generalization_gap_leaderboard.csv
  lcri_generalization_gap_leaderboard.csv
  lcri_generalization_scope_summary.csv
  lcri_generalization_severity.csv
  lcri_generalization_severity_by_scope.csv
  lcri_generalization_scope_risk.csv
  lcri_generalization_scope_gate_decisions.csv
  lcri_generalization_critical_contexts.csv
  lcri_generalization_blocker_summary.json
  lcri_generalization_severity_summary.json
  lcri_worst_generalization_context.json
  lcri_generalization_gate_decision.json
  lcri_generalization_gap_delta.csv
  lcri_gap_delta_flags.csv
  lcri_gap_delta_improvements.csv
  lcri_gap_delta_regressions.csv
  lcri_gap_delta_scorecard.json
  lcri_gap_delta_scope_extremes.csv
  lcri_gap_delta_scope_summary.csv
  lcri_gap_delta_summary.json
  transition_lift.csv
  heldout_transition_lift.csv
  passive_fill_event_lead_lag_profile.csv
  heldout_passive_fill_event_lead_lag_profile.csv
  passive_fill_event_window_transition_matrix.csv
  heldout_passive_fill_event_window_transition_matrix.csv
  passive_fill_event_window_transition_scorecard.json
  heldout_passive_fill_event_window_transition_scorecard.json
  execution_adjusted_lcri_quantile_diagnostics.csv
  heldout_execution_adjusted_lcri_quantile_diagnostics.csv
  queue_position_unfilled_opportunity_curve.csv
  heldout_queue_position_unfilled_opportunity_curve.csv
  queue_position_lcri_tail_adverse_selection_surface.csv
  heldout_queue_position_lcri_tail_adverse_selection_surface.csv
  queue_position_lcri_tail_adverse_selection_release_scorecard.json
  heldout_queue_position_lcri_tail_adverse_selection_release_scorecard.json
  execution_adjusted_lcri_event_window_attribution.csv
  heldout_execution_adjusted_lcri_event_window_attribution.csv
  execution_adjusted_lcri_event_window_release_scorecard.json
  heldout_execution_adjusted_lcri_event_window_release_scorecard.json
  execution_adjusted_lcri_regime_attribution.csv
  heldout_execution_adjusted_lcri_regime_attribution.csv
  queue_position_latency_regime_surface.csv
  heldout_queue_position_latency_regime_surface.csv
  queue_position_latency_edge_survival.csv
  heldout_queue_position_latency_edge_survival.csv
  queue_position_latency_edge_survival_scorecard.json
  heldout_queue_position_latency_edge_survival_scorecard.json
  queue_position_latency_release_scorecard.json
  heldout_queue_position_latency_release_scorecard.json
  queue_position_expected_value_stress_table.csv
  heldout_queue_position_expected_value_stress_table.csv
  queue_position_expected_value_stress_summary.json
  heldout_queue_position_expected_value_stress_summary.json
  transition_robustness.json
  heldout_transition_robustness.json
  research_summary.md
  artifact_manifest.json
  artifact_metadata_summary.json
  sample_snapshots.csv
```

The execution-adjusted LCRI quantile diagnostics bucket rows by raw `abs(lcri)` and report how much signal survives passive-execution constraints, including the selected-side fill probability, selected-side adverse-fill probability, and fill-minus-adverse spread for each bucket. `add_latency_adjusted_passive_fill_probabilities` adds a live-deployment guardrail on top of the queue-position model: it discounts bid/ask fill probabilities and uplifts adverse-fill probabilities under model/routing/exchange decision latency using queue-clearance share, spread, volatility, and replenishment stress, so zero-latency paper fills can be compared against stale-queue deployable odds before execution-adjusted edge is claimed. The queue-position unfilled opportunity curve complements adverse-selection diagnostics by keeping unfilled selected-side LCRI rows in the denominator, splitting raw signal edge into captured edge versus edge stranded by queue non-fills across tail bins and event-window regimes. The queue-position LCRI tail adverse-selection surface cross-buckets raw LCRI tails with selected-side fill-probability bins so reviewers can distinguish calibrated tail fills from high-fill/high-toxicity cells where adverse-selection odds or negative realized edge make the apparent passive alpha non-publishable; its release scorecard converts toxic tail row share, candidate-weighted fill-minus-adverse, and tail fill-residual breaches into pass/review/block labels for demo and CI consumers. The event-window attribution companion cross-tabulates raw LCRI strength with passive-fill event-window regimes so reviewers can spot high-LCRI pockets where fills are available but adverse selection erases the edge; its release scorecard converts those high-LCRI event-window toxicity, signal-survival, and fill-minus-adverse checks into pass/review/block labels for demo and CI consumers. The event-window transition matrix keeps the scored-frame chronology intact by measuring pre_event→event, event→post_event, and other one-step paths with edge decay, next-state toxicity, and negative-edge share, so demo reviewers can see whether tradable fill rows are immediately followed by execution decay rather than relying on static row buckets; its transition scorecard promotes toxic event→post_event decay into pass/review/block release labels with worst-path and candidate-weighted toxicity fields for CI consumers. The regime attribution companion groups by liquidity regime and raw LCRI side so reviewers can see where queue-aware execution preserves, abstains, or inverts the residual imbalance before treating an LCRI result as publishable. The queue-position latency regime surface then replays selected-side realized fills at later snapshot latencies inside each passive-fill event-window regime, exposing cases where apparent passive alpha only works with zero-latency queue state. The latency edge-survival artifact prices that same stale-queue replay in realized execution-adjusted ticks, so lost high-edge fills are not hidden by aggregate fill-rate averages; its scorecard gates releases on candidate-weighted edge survival, worst delayed-edge gap, and fragile-edge candidate share. The latency release scorecard condenses the fill-rate surface into candidate-weighted fill decay, fragile-candidate share, worst latency/regime pointer, and pass/review/block labels for API and CI consumers.

## Python usage

```python
from lcri_lab.model import LCRIModel
from lcri_lab.simulator import SimulationConfig, simulate_order_books

snapshots = simulate_order_books(SimulationConfig(rows=5000, seed=42))
train = snapshots.iloc[:3500]
test = snapshots.iloc[3500:]

model = LCRIModel().fit(train)
scored = model.score_frame(test)
model.save("reports/lcri-model.json")

print(scored[["timestamp", "raw_imbalance", "lcri", "lcri_probability"]].head())
```

Load a persisted model:

```python
from lcri_lab.model import LCRIModel

model = LCRIModel.load("reports/lcri-model.json")
scored = model.score_frame(order_book_snapshots)
```

Persisted models include a `schema_version` field so incompatible artifact changes fail fast.

### Trade-confirmed passive fill labels

When order-level queue messages are available, `add_event_level_trade_confirmed_fill_proxy`
separates same-price trade depletion from cancel/delete queue advancement. A passive
fill is only marked after cumulative queue advance reaches the queue-ahead or
child-order-clearance threshold on a confirming trade; cancel-only queue clears are
flagged as `*_queue_advance_without_trade` so execution-adjusted LCRI studies do
not mistake queue movement for a tradable fill.

```python
from lcri_lab import (
    add_event_level_trade_confirmed_fill_proxy,
    trade_confirmed_passive_fill_latency_summary,
)

fills = add_event_level_trade_confirmed_fill_proxy(
    queued_snapshots,
    order_events,
    horizon=0.250,
    group_cols=("symbol", "session"),
)
latency_review = trade_confirmed_passive_fill_latency_summary(
    fills,
    max_mean_latency=0.300,
    max_cancel_only_clear_rate=0.025,
)
```

`trade_confirmed_passive_fill_latency_summary` returns typed side-level review
rows for `bid`, `ask`, and `all`, including trade-confirmed fill rate,
cancel-only clear rate, mean/p95 fill latency, mean trade/cancel depletion, and a
bounded `review_label`. The companion report verifier accepts both in-sample and
heldout `*_trade_confirmed_passive_fill_latency_summary.csv` artifacts so API
users can gate publication on whether fills are promptly trade-confirmed rather
than inferred from cancel-only queue movement.

### Queue-position drawdown episodes

Execution-aware demos can turn a scored passive path into concrete underwater runs
instead of relying only on aggregate drawdown. `queue_position_path_drawdown_episodes`
replays non-abstain execution rows, keeps optional session/symbol grouping intact,
and reports each episode's start/end row, trough row, recovery edge, side-turnover
count, dominant passive-fill event-window regime, and whether the drawdown recovered
or remained open. `queue_position_path_drawdown_summary` condenses those episodes
into a release-facing artifact that flags unrecovered drawdown share, severe-episode
share, top path concentration, and passive-fill event-window regimes that dominate
queue-position losses.

```python
from lcri_lab import queue_position_path_drawdown_episodes, queue_position_path_drawdown_summary

episodes = queue_position_path_drawdown_episodes(
    scored,
    group_cols="session",
    event_window_col="passive_fill_event_window_regime",
)
summary = queue_position_path_drawdown_summary(episodes)
```

`lcri-lab run-demo` now writes both in-sample and heldout
`queue_position_path_drawdown_episodes.csv` plus
`queue_position_path_drawdown_summary.json` artifacts, and `verify-report` checks
that drawdown magnitudes, open/recovered episode labels, and summary release labels
remain coherent.

### Queue-position EV stress summaries

After selecting queue-position EV policies, `queue_position_expected_value_stress_table`
reprices each selected regime under latency and toxicity haircuts. The companion
`queue_position_expected_value_stress_summary` candidate-weights those stressed EV
rows, reports the fragile candidate share, names the weakest scenario/regime, and
returns a pass/review/block release label so demos cannot publish a passive policy
that only works at optimistic queue-priority assumptions.

```python
from lcri_lab import (
    queue_position_expected_value_policy_selection,
    queue_position_expected_value_stress_summary,
    queue_position_expected_value_stress_table,
)

selection = queue_position_expected_value_policy_selection(ev_frontier)
stress = queue_position_expected_value_stress_table(selection)
stress_gate = queue_position_expected_value_stress_summary(stress)
```

`lcri-lab run-demo` emits both `queue_position_expected_value_stress_table.csv` and
`queue_position_expected_value_stress_summary.json` for in-sample and heldout splits.

### Nonlinear regularization audit

API users can test whether nonlinear liquidity-neutralization lift survives across a
ridge path instead of depending on one tuning value:

```python
from lcri_lab import (
    baseline_nonlinear_regularization_path,
    baseline_nonlinear_regularization_summary,
    compute_features,
)

features = compute_features(order_book_snapshots)
path = baseline_nonlinear_regularization_path(
    features,
    ridges=(0.0, 1e-6, 1e-4, 1e-2, 1.0),
    train_fraction=0.60,
    min_lift=0.0,
)
summary = baseline_nonlinear_regularization_summary(
    path,
    min_supported_ridges=2,
    min_median_lift=0.0,
)
```

`path` returns one row per `(ridge, basis)` with chronological train/test RMSE,
lift versus the core linear basis, residual bias, coefficient norms, and a
`support_label`. `summary` converts that audit into a typed dictionary suitable
for release gates: supported ridge count, best ridge, median lift, coefficient
norm bound, `publishable`, and `review_note`. Treat this as a robustness screen
for an original nonlinear LCRI hypothesis, not as proof that the basis is novel
or universally stable.

## CLI usage

Normalize flat L2 snapshots before fitting or scoring:

```bash
lcri-lab normalize \
  --input raw_l2.csv \
  --output snapshots.csv \
  --tick-size 0.01 \
  --levels 5 \
  --derive-state
```

Fit a model from normalized snapshots:

```bash
lcri-lab fit --input snapshots.csv --model reports/lcri-model.json
```

Inspect a fitted artifact:

```bash
lcri-lab describe-model --model reports/lcri-model.json
```

Score snapshots:

```bash
lcri-lab score \
  --input new_snapshots.csv \
  --model reports/lcri-model.json \
  --output reports/scored_snapshots.csv \
  --columns timestamp,raw_imbalance,lcri,lcri_probability
```

Run the synthetic research workflow with a reproducible training split:

```bash
lcri-lab run-demo --rows 20000 --seed 7 --train-frac 0.70 --passive-fill-horizon 2
```

## Evaluation

The default research workflow compares raw imbalance against LCRI using:

- directional accuracy
- Brier score
- rank correlation
- regime-stratified metrics
- transition-conditioned metrics
- calibration curves

Representative synthetic result from the default seed:

| signal | directional accuracy | Brier score | rank correlation |
| --- | ---: | ---: | ---: |
| raw_imbalance | 0.40395 | 0.320357 | 0.024351 |
| lcri | 0.56455 | 0.265111 | 0.186961 |

The controlled simulation includes structural liquidity bias, spread changes, depth variation, and regime-specific pressure sensitivity. LCRI is useful when raw depth imbalance contains a mixture of persistent liquidity structure and short-horizon pressure.

## Project layout

```text
src/lcri_lab/
  simulator.py      synthetic order book generation
  features.py       imbalance, liquidity-state, and fracture features
  baseline.py       liquidity-conditioned baseline estimator
  model.py          fit, score, save, and load interface
  evaluation.py     metrics and regime-stratified analysis
  execution.py      queue-position fill probabilities and execution calibration
  labels.py         transaction-cost-aware labels
  publishability.py publishable edge gate
  memory.py         rolling pressure persistence features
  absorption.py     pressure absorption and transmission features
  plotting.py       report figures
  cli.py            command-line interface

examples/
  fit_and_score.py

tests/
  test_features.py
  test_baseline.py
  test_model.py
  test_demo.py
```

## Research notes

- [Artifact catalog](docs/artifact-catalog.md)
- [LCRI generalization gates](docs/lcri-generalization-gates.md)
- [Report verification workflow](docs/report-verification.md)

## Current limitations

- The included workflow uses synthetic data.
- Real order book feeds must be normalized into the snapshot schema before scoring.
- The current baseline is transparent and ridge-regularized; nonlinear stress enters through explicit basis terms rather than a black-box estimator.
- Queue-position-aware passive fill probability is snapshot-proxy based for demos, but the execution module also includes an event-level realized-fill adapter that maps configurable venue-specific trade/cancel event-type and side aliases into side-specific `bid_realized_fill`/`ask_realized_fill` labels with symbol/session grouping. Demo calibration labels now use configurable-horizon visible best-level depletion versus estimated queue-ahead instead of only next-mid touches.
- Passive-fill calibration curves can compare that proxy to side-specific realized fill flags, but event-level add/cancel/trade data remains the live-calibration target.
- Passive-fill event windows now surface side-specific post-fill drift by regime and are emitted as demo/report artifacts with calibration summaries. Lead/lag profiles (`passive_fill_event_lead_lag_profile.csv` and heldout counterpart) expose offset-level realized-edge toxicity around high-probability fills so pre-fill warning, event-row selection, and post-fill decay are not hidden inside aggregate window sums.

## Next steps

- Add a real-data adapter for normalized TAQ or crypto L2 snapshots.
- Wire the event-level realized-fill adapter into a real-data ingestion example with venue-specific trade/cancel side conventions.
- Evaluate liquidity fracture features on real L2 data against raw LCRI.
- Test whether pressure memory improves publishable-edge filtering.
- Compare publishable-edge hit rate across shadow absorption regimes.
- Track residual tail diagnostics by side, threshold, and absorption state.
- Audit feature stability by liquidity regime before model promotion.
- Compare directional metrics against cost-aware tradable labels.
- Add model cards with fitted coefficients and residual scales.
