# Artifact catalog

This catalog maps generated report files to their research purpose. It is meant
for reviewers who receive a report directory without the surrounding run logs.

## Core run files

- `sample_snapshots.csv`: first scored snapshots for quick schema inspection.
- `metrics.csv`: full-sample signal quality for raw imbalance and LCRI.
- `heldout_metrics.csv`: heldout signal quality for the same signals.
- `lcri-model.json`: transparent fitted baseline and model configuration.
- `artifact_manifest.json`: final reproducibility manifest with planned report
  paths plus size and SHA-256 metadata for generated artifacts.
- `artifact_metadata_summary.json`: compact metadata-footprint audit derived
  from manifest metadata.
- `artifact_coverage_matrix.csv`: manifest-level artifact classification by
  family, extension, research-summary exposure, figure status, metadata
  tracking, and verification role.
- `artifact_coverage_summary.json`: compact counts for total artifacts,
  summary-backed artifacts, figures, metadata tracking, families, and every
  verification role used by release review.

## Verification roles

`artifact_coverage_matrix.csv` gives every artifact one review role so release
readiness checks can distinguish narrative, audit, and machine-verification
surfaces without opening every file:

- `manifest_audit`: manifest, metadata, and coverage files that prove the bundle
  is complete and reproducible.
- `transition_verification`: transition-conditioned metrics, lift,
  robustness, and reversal-transition gate outputs that test whether latent
  liquidity fracture pressure survives regime changes.
- `lcri_release_evidence`: LCRI gate, gap-delta, contradiction, calibration,
  release-checklist, and lineage artifacts that support publishability review.
- `owner_readiness`: `research_summary.md` and
  `lcri_owner_handoff_packet.md`, the markdown surfaces intended for human
  sign-off.
- `visual_evidence`: generated PNG figures for reviewer inspection.
- `supporting_evidence`: ordinary CSV/JSON support files that are not one of the
  stricter release or audit surfaces.

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
- `alpha_event_windows.csv`: deterministic event-window table for phase-shift
  alpha threshold crossings, including pre/post return drift and event regime.
- `alpha_event_regime_summary.csv`: regime aggregation of alpha event drift,
  sorted by adverse post-event drift share and worst tail drift.
- `alpha_event_window_summary.json`: compact event count, adverse share, mean
  drift, worst event, and max score for release review.
- `alpha_event_score_weighted_drift.json`: score-weighted companion diagnostic
  that checks whether high-score alpha events carry adverse drift.
- `alpha_event_drift_gate.json`: release gate for adverse alpha event drift.
- `alpha_event_release_review_packet.csv`: one-row reviewer packet joining the
  drift gate, score-weighted diagnostic, worst regime, and release note.
- `alpha_event_review_verification_summary.json`: compact verification health
  payload with present/missing counts, pass/fail status, release decision,
  review priority, categorized blocking-error excerpts, and next owner action
  for dashboard triage.
- `execution_adjusted_edge_summary.json`: compact passive-fill-adjusted tradable
  edge summary with abstain share, adverse-fill drag, and publishable-side conflicts.
- `execution_publishability_review_packet.csv`: reviewer packet joining original
  publishability side with best execution side, fill probabilities, edge drag, and
  conflict priority.
- `execution_publishability_release_gate.json` (when emitted by downstream
  report scripts): owner-facing gate that combines the review packet with queue
  execution quality and capacity-stability labels into `pass`/`review`/`block`,
  preventing demo sign-off when passive alpha depends on fragile queue capacity or
  high-priority execution conflicts.
- `passive_fill_event_windows.csv`: high passive-fill-probability event windows
  with side-specific pre/post realized edge drift plus modal pre/post memory-decay
  regime transitions.
- `passive_fill_event_lead_lag_profile.csv`: event-regime x relative-offset
  realized-edge profile around high-probability passive fills, exposing whether
  adverse selection appears before, on, or after the fill event rather than only
  inside aggregate pre/post sums.
- `passive_fill_event_regime_summary.csv`: regime aggregation of passive-fill
  event-window drift, adverse-post-event share, and worst drift.
- `passive_fill_event_transition_summary.csv`: transition-path aggregation of
  passive-fill event toxicity, highlighting boundary states such as calm→thin or
  thin→stress where fills may become adverse despite high predicted fill odds.
- `passive_fill_calibration_curve.csv`: side-specific passive-fill calibration
  curve that bins predicted fill probability against realized fill proxy by regime.
- `passive_fill_calibration_summary.json`: row-weighted passive-fill calibration
  health summary with expected calibration error, weighted Brier score, and worst regime.
- `queue_position_fill_surface.csv`: queue-depth x predicted-fill calibration grid
  that surfaces whether passive execution remains calibrated away from front queue.
- `queue_position_capacity_frontier.json`: thresholded queue-placement capacity
  frontier for the deepest viable passive queue fraction.
- `queue_position_capacity_stability.json`: full-sample versus heldout capacity
  comparison with queue-depth, edge, tradable-share, and side-stability labels.
- `queue_position_edge_decay.csv`: regime-level queue-depth decay summary for fill
  rate, calibration error widening, and execution-adjusted edge.
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
- `lcri_calibration_fracture_pressure.csv`: calibration bins aligned to signal
  quantiles, with residual-weighted monotonicity fracture pressure per bucket.
- `heldout_lcri_calibration_fracture_pressure.csv`: heldout version of the
  calibration-shape pressure table for out-of-sample fracture review.
- `lcri_calibration_fracture_pressure_summary.json`: compact full-sample
  fracture-pressure gate, including miscalibrated fracture counts.
- `heldout_lcri_calibration_fracture_pressure_summary.json`: compact heldout
  gate for deciding whether shape/calibration fractures survive out of sample.
- `phase_shift_artifact_review.csv`: adverse-selection phase-shift rows ranked
  by fracture-weighted review priority for memory-decay artifact triage.
- `heldout_phase_shift_artifact_review.csv`: heldout artifact triage for checking
  whether phase-shift artifacts survive out of sample.
- `lcri_calibration_fracture_gate.json`: combined full-sample plus heldout
  release gate for miscalibrated monotonicity fractures.
- `figures/lcri_cross_artifact_evidence_index.png`: visual owner handoff of the
  cross-artifact evidence score with gate, CI, and uncertainty markers.
- `figures/lcri_evidence_release_checklist.png`: visual release-owner checklist
  ranked by blocker status and evidence score, with required actions overlaid.
- `figures/lcri_owner_handoff_packet.png`: visual owner queue ranked by
  handoff rank, with evidence score, CI/gate priority, and release-blocker
  markers.
- `figures/lcri_evidence_lineage_map.png`: visual lineage health map ranked by
  stale source references, missing owner surfaces, and evidence score.
- `figures/lcri_calibration_fracture_pressure.png`: visual rank of the highest
  residual-weighted calibration-monotonicity fracture-pressure buckets.

## Figures

Figures mirror the most important CSV/JSON artifacts for visual review. Treat
CSV and JSON outputs as source-of-truth for automation, and figures as reviewer
aids.

- `figures/generalization_stability_confidence_intervals.png`: heldout
  directional-accuracy intervals ranked by gap-vs-interval and interval width.
- `figures/lcri_generalization_gap_delta.png`: row-level raw-minus-LCRI gap
  comparison across all scopes.
- `figures/lcri_generalization_severity_by_scope.png`: stacked stable, warning,
  and critical counts by scope.
- `figures/lcri_ci_gate_contradiction_diagnostics.png`: CI-vs-gate rows ranked
  by review priority, with absolute gap bars and CI half-width markers.
- `figures/lcri_gap_delta_scope_summary.png`: mean stability edge by scope.
- `figures/lcri_contradiction_review_packet.png`: priority-ranked contradiction
  evidence with fragility, absolute gate-gap, and relative stability-edge markers.
- `figures/lcri_uncertainty_weighted_review_priority.png`: owner-review queue
  ranked by uncertainty-weighted priority, with base priority, CI-width, and
  wide-CI share markers.
- `figures/lcri_cross_artifact_evidence_index.png`: scope-level evidence index
  ranked by final score, with critical gate row, CI/gate contradiction, and
  uncertainty-priority markers.
- `figures/lcri_evidence_release_checklist.png`: owner release checklist ranked
  by blocked/review/monitor/ready status and evidence score.
- `figures/lcri_owner_handoff_packet.png`: final owner handoff queue ranked by
  decision urgency, with release-blocker and CI/gate priority overlays.
- `figures/lcri_evidence_lineage_map.png`: evidence-to-checklist-to-handoff
  chain health, with stale-reference and missing-surface markers.
- `figures/lcri_calibration_fracture_pressure.png`: calibration-shape fracture
  pressure ranked by quantile, with residual signs annotated on the bars.

Use the figures to spot obvious concentration before opening the corresponding
CSV files for exact values.
