# LCRI Market Microstructure Lab

A compact research system for studying liquidity-conditioned residual imbalance in synthetic limit order books.

Most retail trading demos treat order book imbalance as a raw ratio:

```text
bid_depth - ask_depth
---------------------
bid_depth + ask_depth
```

That misses the central microstructure issue. The same raw imbalance can mean different things in a thick, stable book than it means in a thin, stressed, or rapidly replenishing book. This project tests that distinction directly.

The lab simulates order book states across liquidity regimes, computes raw imbalance, estimates the expected imbalance given local liquidity conditions, and evaluates whether the residual imbalance is more informative for short-horizon price moves.

## Research question

Does imbalance become more useful when measured relative to the local liquidity baseline?

The project compares two signals:

- `raw_imbalance`: visible depth pressure at the top levels of the book.
- `lcri`: liquidity-conditioned residual imbalance, standardized by regime-specific residual scale.

The demo is intentionally synthetic. The goal is not to claim market profitability. The goal is to demonstrate clean market-structure reasoning, controlled experiments, and disciplined evaluation.

## What the demo does

- Simulates order book snapshots under thick, thin, stressed, and replenishing regimes.
- Builds depth, spread, volatility, replenishment, and queue-pressure features.
- Computes raw imbalance from visible bid and ask depth.
- Fits a deterministic liquidity baseline for imbalance.
- Computes LCRI as a standardized residual.
- Simulates next-horizon mid-price moves with regime-dependent sensitivity.
- Evaluates raw imbalance vs LCRI using directional accuracy, Brier score, rank correlation, and regime-stratified performance.
- Writes figures and result tables under `reports/`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
lcri-lab run-demo --rows 20000 --seed 7
pytest
```

Generated artifacts:

```text
reports/
  figures/
    raw_vs_lcri_scatter.png
    regime_signal_quality.png
    calibration_curve.png
  metrics.csv
  regime_metrics.csv
  sample_snapshots.csv
```

## Core formula

Let `I_t` be raw imbalance and `X_t` be local liquidity state features. The baseline imbalance is:

```text
E[I_t | X_t]
```

The liquidity-conditioned residual imbalance is:

```text
LCRI_t = (I_t - E[I_t | X_t]) / sigma_regime
```

where `sigma_regime` is the residual scale estimated inside comparable liquidity regimes.

## Why this is recruiter-relevant

This is not a trading bot. It is a controlled microstructure research environment. The project is designed to show:

- market-state conditioning rather than flat feature engineering
- awareness of liquidity, spread, and queue effects
- post-signal evaluation by regime
- reproducible simulation and metrics
- clean engineering boundaries between simulation, features, modeling, and reporting

## Project layout

```text
src/lcri_lab/
  simulator.py      synthetic limit order book generation
  features.py       imbalance and liquidity-state features
  baseline.py       liquidity-conditioned baseline model
  evaluation.py     metrics and regime-stratified analysis
  plotting.py       report figures
  cli.py            command-line demo runner

tests/
  test_features.py
  test_baseline.py
  test_demo.py
```

## Limitations

- The market data is synthetic.
- The fill model is not implemented yet.
- The baseline is intentionally transparent instead of heavily optimized.
- No claim is made about live trading performance.

## Next milestones

- Add a price-time priority matching engine.
- Add queue-position simulation.
- Add event shock regimes.
- Add transaction-complete labels after spread and slippage.
- Add comparison against a learned nonlinear baseline.
