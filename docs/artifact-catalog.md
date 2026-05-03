# Artifact catalog

This catalog maps generated report files to their research purpose. It is meant
for reviewers who receive a report directory without the surrounding run logs.

## Core run files

- `sample_snapshots.csv`: first scored snapshots for quick schema inspection.
- `metrics.csv`: full-sample signal quality for raw imbalance and LCRI.
- `heldout_metrics.csv`: heldout signal quality for the same signals.
- `lcri-model.json`: transparent fitted baseline and model configuration.

## Generalization files

- `generalization_gap.csv`: signal-level full-sample minus heldout gaps.
- `regime_generalization_gap.csv`: gap table split by liquidity regime.
- `transition_generalization_gap.csv`: gap table split by transition segment.
- `generalization_overview.json`: row counts and max gap values for quick audit.
- `generalization_gap_leaderboard.csv`: largest directional-accuracy gaps across
  all scopes and signals.

## LCRI gate files

- `lcri_generalization_gap_leaderboard.csv`: LCRI-only gap leaderboard.
- `lcri_generalization_severity.csv`: stable/warning/critical labels per LCRI
  gap row.
- `lcri_generalization_scope_risk.csv`: warning and critical shares by scope.
- `lcri_generalization_gate_decision.json`: compact pass/block result.
- `lcri_generalization_blocker_summary.json`: affected scopes and worst blocker
  context when critical rows exist.

## Stability comparison files

- `lcri_generalization_gap_delta.csv`: raw imbalance gap minus LCRI gap.
- `lcri_gap_delta_scorecard.json`: mean, median, and share-level stability edge.
- `lcri_gap_delta_scope_summary.csv`: per-scope mean, min, max, and share-level
  stability deltas.
- `lcri_gap_delta_flags.csv`: categorical row labels for relative stability.
- `lcri_gap_delta_improvements.csv`: rows where LCRI degraded less than raw
  imbalance, sorted best first.
- `lcri_gap_delta_regressions.csv`: rows where LCRI degraded more than raw
  imbalance, sorted worst first.
- `lcri_gap_delta_summary.json`: strongest LCRI stability and instability
  contexts.

## Figures

Figures mirror the most important CSV/JSON artifacts for visual review. Treat
CSV and JSON outputs as source-of-truth for automation, and figures as reviewer
aids.

- `figures/lcri_generalization_gap_delta.png`: row-level raw-minus-LCRI gap
  comparison across all scopes.
- `figures/lcri_generalization_severity_by_scope.png`: stacked stable, warning,
  and critical counts by scope.
- `figures/lcri_gap_delta_scope_summary.png`: mean stability edge by scope.

Use the figures to spot obvious concentration before opening the corresponding
CSV files for exact values.
