# LCRI generalization gates

LCRI gates are release-review artifacts for checking whether residual imbalance
quality survives heldout evaluation. They are intentionally small, deterministic,
and easy to inspect in CI logs.

## Inputs

- `lcri_generalization_gap_leaderboard.csv` ranks LCRI directional-accuracy gaps
  across signal, regime, and transition scopes.
- `lcri_generalization_severity.csv` labels each LCRI gap as `stable`,
  `warning`, or `critical`.
- `lcri_generalization_severity_by_scope.csv` rolls those labels up by scope so
  reviewers can see whether degradation is localized or broad.
- `lcri_generalization_scope_risk.csv` converts those scope counts into warning
  and critical shares for quick risk ranking.
- `lcri_generalization_scope_gate_decisions.csv` assigns each scope a `pass`,
  `warn`, or `block` decision with a compact reason.
- `lcri_generalization_critical_contexts.csv` lists only blocking critical rows,
  sorted by largest directional-accuracy gap first.
- `lcri_generalization_blocker_summary.json` compresses critical rows into row
  count, affected scopes, and worst blocker context.
- `lcri_worst_generalization_context.json` records the single largest LCRI gap.

## Gate outputs

- `lcri_generalization_severity_summary.json` counts all severity labels and
  exposes `passes_lcri_generalization_gate`.
- `lcri_generalization_gate_decision.json` turns the severity summary and worst
  context into a compact `pass` or `block` decision with a reason string.
- `figures/lcri_generalization_severity_by_scope.png` mirrors the scope rollup
  for visual review.

## Reading a block

A blocked run means at least one LCRI generalization row crossed the critical
directional-accuracy gap threshold. Start with the gate decision reason, then
inspect `lcri_generalization_critical_contexts.csv` and the worst scope/context
before changing model or feature code.

If the block is regime-localized, compare the regime gap table against feature
stability. If the block is transition-localized, inspect transition robustness
and the transition-conditioned heldout metrics. Use the scope risk and scope decision tables when multiple scopes have warnings
and you need to prioritize follow-up work.

## Reading a warning-only pass

A warning-only pass means no critical rows were found, but one or more scopes had
non-trivial heldout degradation. Treat this as acceptable for exploratory runs,
but keep the warning scopes visible in release notes or follow-up experiments.
