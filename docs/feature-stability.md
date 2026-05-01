# Feature Stability Report

Research features should be audited before they are trusted.

`feature_stability_report` summarizes selected feature distributions by regime. The goal is to catch fragile inputs before they are used in scoring, calibration, or publishability gates.

For each feature and regime it reports:

- rows
- finite rate
- mean
- standard deviation
- 5th percentile
- 95th percentile

This is useful for microstructure research because feature quality is rarely uniform across liquidity regimes. A variable that behaves cleanly in thick books can become unstable during stressed or thin regimes.

Initial use cases:

- compare fracture features across regimes
- inspect whether pressure memory becomes unstable during stressed books
- detect non-finite or collapsed feature series before fitting a baseline
