# Absorption Regime Metrics

Shadow absorption assigns each scored snapshot to an absorption state:

```text
transmitted
contested
absorbed
```

The regime label is only useful if it can be evaluated. `absorption_regime_metrics` splits the frame by absorption regime and evaluates both `lcri` and `transmission_pressure` inside each group.

This answers a direct research question:

```text
Does the absorption haircut improve signal behavior specifically in absorbed or contested books?
```

The output keeps the same metric columns as the main evaluation table:

- directional accuracy
- Brier score
- rank correlation
- mean absolute score

This makes absorption a measurable market-state partition instead of a narrative tag.
