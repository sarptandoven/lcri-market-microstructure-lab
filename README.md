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
lcri-lab run-demo --rows 20000 --seed 7 --train-frac 0.70
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
  metrics.csv
  heldout_metrics.csv
  regime_metrics.csv
  heldout_regime_metrics.csv
  transition_metrics.csv
  heldout_transition_metrics.csv
  transition_lift.csv
  heldout_transition_lift.csv
  transition_robustness.json
  heldout_transition_robustness.json
  research_summary.md
  artifact_manifest.json
  sample_snapshots.csv
```

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
lcri-lab run-demo --rows 20000 --seed 7 --train-frac 0.70
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

## Current limitations

- The included workflow uses synthetic data.
- Real order book feeds must be normalized into the snapshot schema before scoring.
- The current baseline is linear and transparent by design.
- Queue position and fill probability are not implemented yet.

## Next steps

- Add a real-data adapter for normalized TAQ or crypto L2 snapshots.
- Evaluate liquidity fracture features on real L2 data against raw LCRI.
- Test whether pressure memory improves publishable-edge filtering.
- Compare publishable-edge hit rate across shadow absorption regimes.
- Track residual tail diagnostics by side, threshold, and absorption state.
- Audit feature stability by liquidity regime before model promotion.
- Compare directional metrics against cost-aware tradable labels.
- Add nonlinear baseline estimators with identical `fit` and `score_frame` semantics.
- Extend the publishability gate with queue-position-aware fill probability.
- Add event-window regime tagging.
- Add model cards with fitted coefficients and residual scales.
