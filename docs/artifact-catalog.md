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
- `generalization_stability_confidence_intervals.csv`: heldout directional
  accuracy confidence intervals and gap-vs-interval flags for each fragility row.
- `generalization_stability_confidence_summary.json`: compact interval width and
  gap-exceeds-interval counts for uncertainty review.

## LCRI gate files

- `lcri_generalization_gap_leaderboard.csv`: LCRI-only gap leaderboard.
- `lcri_generalization_severity.csv`: stable/warning/critical labels per LCRI
  gap row.
- `lcri_generalization_scope_risk.csv`: warning and critical shares by scope.
- `lcri_generalization_scope_gate_decision_summary.json`: pass, warn, and block
  counts across scopes for dashboard checks.
- `lcri_generalization_gate_decision.json`: compact pass/block result.
- `lcri_generalization_blocker_summary.json`: affected scopes and worst blocker
  context when critical rows exist.
- `lcri_ci_gate_contradiction_diagnostics.csv`: LCRI-only severity rows joined to
  heldout confidence intervals, with labels for blockers or warnings that sit
  inside the CI half-width and stable rows that exceed it.
- `lcri_ci_gate_contradiction_summary.json`: compact CI-vs-gate contradiction
  counts and the highest-priority review context.
- `lcri_ci_confidence_coverage_scorecard.csv`: scope-level CI coverage audit
  combining interval width, gap-outside-CI counts, CI/gate contradictions, and
  high-priority review rows.
- `lcri_ci_confidence_coverage_summary.json`: compact review-scope counts and
  worst CI confidence coverage scope.
- `figures/lcri_ci_confidence_coverage_scorecard.png`: visual owner audit of
  scope-level max/mean CI width, wide-CI share, gap-outside-CI share, and
  CI/gate contradiction counts.

## Stability comparison files

- `lcri_generalization_gap_delta.csv`: raw imbalance gap minus LCRI gap.
- `lcri_gap_delta_scorecard.json`: mean, median, and share-level stability edge.
- `lcri_gap_delta_scope_extremes.csv`: best and worst stability edge context per
  scope.
- `lcri_gap_delta_scope_summary.csv`: per-scope mean, min, max, and share-level
  stability deltas.
- `lcri_gap_delta_dominant_scopes.json`: strongest mean stability edge and drag
  across signal, regime, and transition scopes.
- `lcri_gap_delta_flags.csv`: categorical row labels for relative stability.
- `lcri_gap_delta_improvements.csv`: rows where LCRI degraded less than raw
  imbalance, sorted best first.
- `lcri_gap_delta_regressions.csv`: rows where LCRI degraded more than raw
  imbalance, sorted worst first.
- `lcri_gap_delta_summary.json`: strongest LCRI stability and instability
  contexts.
- `lcri_scope_stability_contradictions.csv`: scope-level audit joining gate
  decisions, relative LCRI-vs-raw stability shares, and fragility review counts
  to flag cross-scope report contradictions.
- `lcri_scope_stability_contradiction_summary.json`: compact counts of aligned
  and contradiction scopes plus the highest-priority contradiction context.
- `lcri_contradiction_review_packet.csv`: evidence packet that links each
  scope-level gate/stability contradiction to the worst deterministic gate row,
  worst relative LCRI-vs-raw delta row, and worst fragility/gate review row.
- `lcri_contradiction_review_packet_summary.json`: compact priority counts and
  worst-scope pointers derived from the contradiction review packet.
- `lcri_uncertainty_weighted_review_priority.csv`: owner-review queue that
  combines contradiction packet priority with CI coverage width, CI/gate
  disagreement counts, and fragility uncertainty.
- `lcri_uncertainty_weighted_review_priority_summary.json`: compact label counts
  and worst-scope pointer for the uncertainty-weighted review queue.
- `lcri_cross_artifact_evidence_index.csv`: scope-level evidence index that joins
  gate decisions, severity counts, LCRI-vs-raw stability shares, fragility review
  counts, CI/gate contradictions, and uncertainty-weighted owner priority.
- `lcri_cross_artifact_evidence_index_summary.json`: compact urgent/review/
  monitor/aligned counts and worst-scope pointer for the cross-artifact index.
- `lcri_evidence_release_checklist.csv`: evidence-index-derived owner checklist
  that maps each scope to blocked, needs-review, monitor, or ready status with a
  required action before release sign-off.
- `lcri_evidence_release_checklist_summary.json`: compact checklist counts,
  worst-scope pointer, and boolean `release_ready` checkpoint.
- `lcri_owner_handoff_packet.csv`: one-file owner queue that joins release
  checklist status with the supporting evidence index fields needed for
  waive/fix/review/sign-off decisions.
- `lcri_owner_handoff_packet_summary.json`: compact handoff queue counts,
  top-scope pointer, and boolean `handoff_clear` checkpoint.
- `lcri_owner_handoff_packet.md`: owner-facing markdown packet with the handoff
  summary, top queue rows, and source-artifact lineage for waive/fix/review/sign-off.
- `lcri_evidence_lineage_map.csv`: scope-level lineage audit that traces each
  evidence index row through the release checklist and owner handoff packet,
  flagging missing or stale source-artifact links.
- `lcri_evidence_lineage_map_summary.json`: compact lineage-health counts,
  worst-scope pointer, and boolean `lineage_clear` checkpoint.
- `figures/lcri_cross_artifact_evidence_index.png`: visual owner handoff of the
  cross-artifact evidence score with gate, CI, and uncertainty markers.
- `figures/lcri_evidence_release_checklist.png`: visual release-owner checklist
  ranked by blocker status and evidence score, with required actions overlaid.
- `figures/lcri_owner_handoff_packet.png`: visual owner queue ranked by
  handoff rank, with evidence score, CI/gate priority, and release-blocker
  markers.
- `figures/lcri_evidence_lineage_map.png`: visual lineage health map ranked by
  stale source references, missing owner surfaces, and evidence score.

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
