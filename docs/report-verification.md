# Report verification workflow

The demo report is designed to fail loudly when an expected research artifact is
missing or stale. Use `lcri-lab verify-report` after copying, regenerating, or
post-processing report directories.

## What gets checked

Verification combines three layers:

1. The manifest lists expected files and optional size/hash metadata.
2. Manifest metadata coverage is checked when metadata is present: entries must
   correspond to listed relative artifacts, include size and SHA-256 fields, and
   cover every listed artifact.
3. Artifact coverage matrices are recomputed from manifest paths so audit files
   cannot silently omit research-summary, figure, metadata, or family coverage.
4. Manifest metadata summaries are recomputed from manifest size/hash records.
5. Manifest-listed PNG figures are checked for a valid PNG signature, positive
   IHDR dimensions, and an IEND trailer.
6. Artifact-specific verifiers check required CSV columns or JSON keys, including
   pressure-memory decay-state and release-velocity bounds.
7. Research-summary sections are checked against generated CSV/JSON artifacts so
   owner-facing markdown cannot silently retain stale placeholders, omit keys, or
   carry stale numerical values after partial regeneration.
8. Cross-artifact consistency checks recompute compact summaries from source
   tables where dashboard payloads mirror CSV artifacts.
9. The CLI returns a non-zero exit code when any required artifact is absent,
   structurally incomplete, or internally inconsistent.

This is intentionally lighter than a statistical acceptance test. It confirms
that downstream reviewers and dashboards can read a run before they interpret the
numbers.

The scope gate decision summary is verified as JSON instead of CSV so automated
checks can read the high-level release posture without loading the full decision
table. Use the CSV when you need per-scope reasons, and the summary when you only
need pass, warn, and block counts. Verification now recomputes those counts and
scope-name lists from `lcri_generalization_scope_gate_decisions.csv` so a stale
summary cannot silently disagree with the release-review table.

## Generalization artifacts

The LCRI generalization gate artifacts are verified separately because they feed
release decisions:

- severity rows must include scope, context, gap, and severity columns
- fragility/gate alignment rows must include heldout uncertainty, severity,
  alignment labels, and review notes
- fragility/gate scorecards must match alignment-label review counts and worst
  review context
- scope rollups must include stable, warning, and critical row counts
- severity summaries and scope rollups must match the row-level severity table
- scope risk tables must include warning-or-critical and critical shares
- scope gate decision summaries must include pass, warn, and block counts
- scope gate decision summaries must match the decision CSV counts and scope lists
- gate decision JSON must include the pass/block decision and reason
- gate decision JSON must match the severity summary and worst-context payload
- critical context CSVs must contain exactly the critical severity rows
- blocker summaries must include the affected scopes and worst blocker context
- blocker summaries must match the critical context CSV counts, scopes, and worst blocker
- fragility/gate alignment labels must match the severity and heldout fragility pair
- heldout confidence intervals must recompute lower/upper bounds, interval width,
  confidence level, and gap-exceeds-half-width flags from fragility uncertainty
- scope stability contradiction rows must reconcile gate decisions, LCRI-vs-raw
  gap-delta scope shares, and fragility review counts
- contradiction review packets must link each scope to the recomputed worst gate,
  worst relative delta, worst fragility review evidence, and priority label
- contradiction review packet summaries must match the packet's priority counts,
  total fragility review rows, and worst-scope pointers
- uncertainty-weighted review priorities must match the contradiction packet and
  CI confidence scorecard, including label counts and worst-scope summary fields
- cross-artifact evidence index rows must recompute from severity-by-scope,
  scope gates, gap-delta scope shares, stability contradictions, CI confidence
  coverage, and uncertainty-weighted priorities, with a manifest/PNG-verified
  score plot for owner review
- evidence-derived release checklist rows must recompute from the cross-artifact
  evidence index, including blocked/review/monitor/ready status, required owner
  action, summary counts, the boolean release-ready checkpoint, and a
  manifest/PNG-verified release checklist plot
- owner handoff packet rows must recompute from the cross-artifact evidence
  index and release checklist, including handoff status, queue text, evidence
  context fields, summary counts, the boolean handoff-clear checkpoint, a
  stale-checked owner-facing markdown handoff packet, and a manifest/PNG-verified
  owner queue plot
- evidence lineage map rows must recompute from the evidence index, release
  checklist, and owner handoff packet, including source-artifact links, lineage
  health labels, summary counts, and the boolean lineage-clear checkpoint
- CI/gate contradiction diagnostics must recompute from LCRI severity rows and
  heldout confidence interval flags, including review priorities and summary counts
  plus a ranked visual audit figure
- alpha event release review artifacts must recompute the window summary,
  score-weighted drift JSON, drift gate JSON, and one-row release packet from
  `alpha_event_windows.csv`, so partial regeneration cannot leave stale owner
  notes or high-score adverse-drift flags in place

## Heldout fragility artifacts

Demo runs now include `generalization_fragility_diagnostics.csv`,
`generalization_fragility_summary.json`,
`generalization_stability_confidence_intervals.csv`,
`generalization_stability_confidence_summary.json`,
`lcri_fragility_gate_alignment.csv`,
`figures/generalization_fragility_diagnostics.png`, and
`figures/generalization_stability_confidence_intervals.png`. The diagnostics compare
full-sample and heldout directional accuracy by signal, regime, and transition
scope, then scale each gap by the heldout binomial standard error. The resulting
`stable`, `watch`, and `fragile` labels are uncertainty diagnostics, not release
gates. Confidence interval artifacts expose the raw heldout interval behind each
fragility row, including whether the full-to-heldout gap exceeds the interval
half-width; the companion figure ranks those intervals by review urgency for
visual audit. The alignment table joins LCRI-only fragility rows to deterministic gate
severity so reviewers can spot rows where a critical gate is statistically stable
or where a non-critical gap is fragile relative to heldout uncertainty. The
scorecard condenses those rows into aligned, review-required, deterministic-block,
and uncertainty-fragile counts for dashboards.
Verification checks the expected columns, summary keys, PNG integrity,
non-negative uncertainty scales, ratio math, confidence interval bounds,
threshold labels, summary counts, alignment labels, and scorecard counts so the report always exposes a consistent
heldout sample size and uncertainty scale behind each generalization gap.

CI/gate contradiction artifacts also include
`figures/lcri_ci_gate_contradiction_diagnostics.png`, which ranks the LCRI-only
CI-vs-gate rows by review priority and overlays CI half-width markers on the
absolute deterministic gap bars. The verifier treats the CSV/JSON diagnostics as
the numeric source of truth and checks the PNG through the manifest/figure layer.
The CI confidence coverage scorecard is recomputed from the heldout interval
artifact and CI/gate diagnostics, then its summary is checked for review-scope,
wide-CI, contradiction, and worst-scope counts so dashboard triage cannot drift
from the row-level uncertainty evidence. Its companion
`figures/lcri_ci_confidence_coverage_scorecard.png` is manifest/PNG verified as a
visual audit surface for the same scope-level uncertainty queue. The
uncertainty-weighted review queue then joins that CI scorecard back to the
contradiction packet, producing a single owner-facing priority score and summary
that verification recomputes from both source artifacts. The cross-artifact
evidence index takes the final step by joining all scope-level gate, stability,
fragility, CI, and priority surfaces into one sortable review table; its verifier
rebuilds the table and summary from source artifacts to catch stale owner-review
packets. The evidence-derived release checklist then maps the same scope rows to
explicit release-owner actions and a `release_ready` boolean; verification
recomputes both CSV and JSON checklist artifacts from the evidence index so stale
sign-off status cannot survive partial regeneration. The final owner handoff
markdown packet is checked against the CSV/JSON handoff queue so copied report
bundles cannot retain stale owner-facing bullets or top queue rows.

## Gap-delta artifacts

LCRI-vs-raw gap-delta scorecards are checked against
`lcri_generalization_gap_delta.csv` as the source of truth:

- scorecard and summary counts, shares, means, medians, and edge contexts are recomputed
- improvement and regression CSVs must contain exactly the positive and negative delta rows
- scope summary, scope extremes, and dominant-scope JSON must match recomputed scope rollups
- flag rows must match the expected `lcri_more_stable`, `lcri_less_stable`, or
  `lcri_equal_stability` classification for each scope/context

Demo runs also emit `lcri_scope_stability_contradictions.csv` and
`lcri_scope_stability_contradiction_summary.json`. These artifacts detect
cross-scope report contradictions, for example a blocked absolute gate in a
scope where LCRI is usually more stable than raw imbalance, a passing scope with
relative LCRI regressions, or a warning scope with broad relative regression.
Verification recomputes labels and summary counts from the gate decision,
gap-delta scope summary, and fragility/gate alignment artifacts so the owner
digest cannot mix stale gate posture with fresh stability deltas. The companion
`lcri_contradiction_review_packet.csv` is also recomputed from row-level severity,
gap-delta, and fragility/gate evidence so review packets cannot point at stale
worst contexts after partial regeneration. The companion
`lcri_contradiction_review_packet_summary.json` is recomputed from the packet so
priority counts and worst-scope pointers stay aligned with the row-level review
queue.

## Research summary sections

`research_summary.md` is verified as the concise owner-facing digest of the run.
When a CSV or JSON artifact exists, its mapped markdown section must also exist,
must not say `_Not generated._`, and must expose the artifact's columns or JSON
keys. Verification also compares rendered CSV cell values and JSON bullet values
using the same formatting as the generated summary. This catches stale summaries
after partial report regeneration, especially for LCRI gate, fragility, and
gap-delta sections that are redundant with machine-readable artifacts.

## Transition verification consistency

Verification treats `lcri_reversal_transition_gate.csv` and its heldout sibling as
localized explanations of the combined fracture/reversal release gate. Each row
must echo the run-level release decision, keep transition stress share within
`[0, 1]`, use non-negative finite reversal coupling, and reserve `review` for
active release gates where a transition carries at least half of transition-local
reversal stress. This prevents the transition gate from claiming a clean pass
when latent-liquidity fracture pressure is actually concentrated at a regime
boundary, or claiming review when there is no active combined-release support.

## Artifact coverage matrix

Demo runs write `artifact_coverage_matrix.csv` and
`artifact_coverage_summary.json` as a compact audit surface for the report
bundle. The matrix classifies every manifest artifact by family, extension,
research-summary exposure, figure status, manifest-metadata tracking, and
verification role. Roles separate manifest audits, visual evidence, transition
verification, LCRI release evidence, owner readiness, and ordinary supporting
evidence.

The summary counts total artifacts, summary-backed artifacts, figures,
metadata-tracked artifacts, distinct families, and every verification role. In
particular, transition verification artifacts are counted separately from
owner-readiness markdown, release-evidence tables, visual evidence, and ordinary
supporting evidence, so a report can show
whether the latent-liquidity-fracture, transmitted-pressure, reversal-stress, and
transition-gate chain has enough machine-readable support before reviewers read
the narrative. Verification recomputes both from the manifest so stale coverage
dashboards fail before reviewers rely on them.

## When verification fails

Treat missing artifacts as a run-production issue first, not a model finding.
Regenerate the report from the same seed and training fraction before changing
model code. If only hashes differ, check whether the artifact was intentionally
regenerated or post-processed.

## Useful commands

```bash
lcri-lab run-demo --rows 20000 --seed 7 --train-frac 0.70 --output reports
lcri-lab verify-report reports
```

For CI, run verification after tests so structural report failures stay visible
even when unit coverage passes.

## Failure summaries

When verification fails, the CLI includes a compact error-family summary in the
exception text. The summary separates manifest mismatches, generalization table
issues, LCRI gate artifacts, figures, and uncategorized errors. This keeps CI
logs readable when several artifacts are missing at once.

A clean run prints the same summary with `passes_verification: True`, which is
useful when storing command output next to generated reports.

## Figure artifacts

Figure verification is intentionally format-level, not visual. It catches empty,
truncated, or non-PNG files listed in the manifest before reviewers open broken
charts, while leaving statistical interpretation to the generated CSV and JSON
artifacts.

## Metadata footprint

Demo runs also write `artifact_metadata_summary.json`. This file reports how
many artifacts have manifest metadata, their total byte footprint, and the
largest generated artifact. Verification first hardens the manifest metadata
itself, rejecting unsafe paths, unlisted metadata entries, incomplete metadata
records, and partial coverage when any metadata is present. It then recomputes
this summary payload from the manifest metadata, excluding the summary file
itself because it is written before the final manifest. It is meant for
lightweight audit logs, not for statistical interpretation.
