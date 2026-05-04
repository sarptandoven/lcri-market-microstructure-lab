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
6. Artifact-specific verifiers check required CSV columns or JSON keys.
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

## Metadata footprint

Demo runs also write `artifact_metadata_summary.json`. This file reports how
many artifacts have manifest metadata, their total byte footprint, and the
largest generated artifact. It is meant for lightweight audit logs, not for
statistical interpretation.
