# LCRI generalization gates

LCRI gates are release-review artifacts for checking whether residual imbalance
quality survives heldout evaluation. They are intentionally small, deterministic,
and easy to inspect in CI logs.

## Inputs

- `lcri_generalization_gap_leaderboard.csv` ranks LCRI directional-accuracy gaps
  across signal, regime, and transition scopes.
- `lcri_generalization_severity.csv` labels each LCRI gap as `stable`,
  `warning`, or `critical`.
- `lcri_fragility_gate_alignment.csv` joins those LCRI gate labels to heldout
  fragility diagnostics so reviewers can distinguish deterministic blockers
  from gaps that are large relative to heldout uncertainty.
- `lcri_fragility_gate_scorecard.json` counts alignment contradictions and review
  rows so release dashboards can track whether a gate block is deterministic,
  uncertainty-driven, or aligned.
- `lcri_generalization_severity_by_scope.csv` rolls those labels up by scope so
  reviewers can see whether degradation is localized or broad. Verification
  recomputes this rollup from the row-level severity table to catch stale scope
  dashboards.
- `lcri_generalization_scope_risk.csv` converts those scope counts into warning
  and critical shares for quick risk ranking.
- `lcri_generalization_scope_gate_decisions.csv` assigns each scope a `pass`,
  `warn`, or `block` decision with a compact reason.
- `lcri_generalization_scope_gate_decision_summary.json` counts pass, warn, and
  block scopes and lists the scope names requiring follow-up.
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
- `lcri_generalization_scope_gate_decision_summary.json` is the quickest payload
  for dashboards that need counts before loading the full decision table.
- `figures/lcri_generalization_severity_by_scope.png` mirrors the scope rollup
  for visual review.

## Reading a block

A blocked run means at least one LCRI generalization row crossed the critical
directional-accuracy gap threshold. Start with the gate decision reason, then
inspect `lcri_generalization_critical_contexts.csv` and the worst scope/context
before changing model or feature code.

If the block is regime-localized, compare the regime gap table against feature
stability. If the block is transition-localized, inspect transition robustness
and the transition-conditioned heldout metrics.

Use the scope risk, scope decision table, and scope decision summary when
multiple scopes have warnings and you need to prioritize follow-up work. The
summary is intentionally redundant with the CSV so that automated release checks
can fail fast without parsing table rows. `lcri-lab verify-report` also
recomputes the summary counts and scope lists from the CSV, which catches stale
or hand-edited dashboard payloads before reviewers rely on them.

Use `lcri_fragility_gate_alignment.csv` when a block or warning looks surprising.
Use `lcri_ci_gate_contradiction_diagnostics.csv` when the question is narrower:
whether a deterministic LCRI stable/warning/critical label agrees with the heldout
confidence interval half-width for the same scope and context. The companion
`figures/lcri_ci_gate_contradiction_diagnostics.png` is the fast visual queue:
absolute gap bars show the deterministic evidence, CI half-width markers show the
uncertainty band, and row colors encode the CI/gate label.
`gate_blocks_inside_ci` and `gate_warns_inside_ci` mean the deterministic gate
crossed a severity threshold even though the gap sits inside the heldout CI
half-width. `stable_gap_outside_ci` means a stable deterministic gate still
exceeds the CI half-width, so treat dismissal as a review decision rather than a
purely thresholded pass.
Use `lcri_ci_gate_contradiction_summary.json` for the compact audit count of
those CI/gate review-required rows and the worst review context. Use
`lcri_ci_confidence_coverage_scorecard.csv` when you need scope-level triage: it
combines wide heldout interval counts, gap-outside-CI counts, CI/gate
contradiction rows, and high-priority gate disagreements into a single review
queue. Its JSON summary is the dashboard-friendly count of review scopes and the
worst CI confidence coverage scope, while
`figures/lcri_ci_confidence_coverage_scorecard.png` ranks those same scopes for
visual audit. Use
`lcri_contradiction_review_packet.csv` when scope-level stability contradictions
need row-level evidence: it attaches the worst gate context, worst relative
LCRI-vs-raw delta context, worst fragility review context, and review priority for
each scope. Use `lcri_contradiction_review_packet_summary.json` for the compact
review queue counts and worst-scope pointers derived from that packet. Use
`lcri_uncertainty_weighted_review_priority.csv` when owners need one queue that
weights contradiction priority by fragility ratios, CI width, wide-CI share, and
CI/gate disagreement counts; its JSON summary gives the compact label counts and
worst weighted scope. Use `lcri_cross_artifact_evidence_index.csv` for the final
scope-level owner handoff: it joins gate decision, severity counts, LCRI-vs-raw
stability shares, fragility review counts, CI/gate contradiction counts, and the
uncertainty-weighted priority into one sortable urgent/review/monitor/aligned
index. Its JSON summary is the compact release-review checkpoint. The companion
`lcri_evidence_release_checklist.csv` translates the same index into owner
checklist rows with blocked, needs-review, monitor, or ready status plus a
required action; `lcri_evidence_release_checklist_summary.json` exposes the
boolean `release_ready` checkpoint for automation. `lcri_owner_handoff_packet.csv`
and `lcri_owner_handoff_packet.md` are the final owner queue surfaces: the CSV is
machine-readable, and the markdown packet is the concise waive/fix/review/sign-off
handoff with source-artifact lineage. `figures/lcri_cross_artifact_evidence_index.png`
is the visual owner handoff for spotting the highest-score scope before opening
the CSV, while `figures/lcri_evidence_release_checklist.png` is the release-signoff
view ranked by blocker status and evidence score with required owner actions
overlaid. `lcri_evidence_lineage_map.csv` and
`figures/lcri_evidence_lineage_map.png` close the loop by showing whether each
scope's evidence row still has a complete, non-stale path through checklist and
handoff surfaces.

## Reading a warning-only pass

A warning-only pass means no critical rows were found, but one or more scopes had
non-trivial heldout degradation. Treat this as acceptable for exploratory runs,
but keep the warning scopes visible in release notes or follow-up experiments.
