# Preview Ladder DiT

Preview Ladder DiT is a benchmark and method scaffold for masked video replacement where the fast preview is not a disposable draft. The preview becomes an explicit structural contract for the final render: accepted layout, mask occupancy, trajectory, boundary confidence, and low-frequency appearance are measured, stored, and reused.

## Why this exists

Most video-generation benchmarks ask whether a final video looks good. Production replacement workflows fail earlier: a creator approves a fast preview, waits for a final render, then discovers that the object moved, boundaries changed, the background leaked, or temporal flicker increased. The real product metric is time-to-acceptable-edit, not only final-render latency.

## What this repo provides

1. A public report schema for preview-final consistency.
2. Deterministic synthetic fixtures with known failure modes.
3. Lower-is-better metrics for drift, boundaries, flicker, background preservation, trajectory, mask occupancy, and edge jitter.
4. A CLI benchmark contract suitable for CI and public submissions.
5. An integration path for LTX-2.3 preview and two-stage final pipelines.

## Quick start

```bash
python3 -m pytest -q
python3 -m preview_ladder_dit.cli run-synthetic --out /tmp/pld-run
python3 -m preview_ladder_dit.cli validate-report /tmp/pld-run/report-clean.json
```

## The public claim

Hypothesis: for masked replacement, final renders conditioned on accepted preview commitments will reduce user-visible structural surprises at equal or better time-to-acceptable-edit than independent preview/final sampling. This repo is structured so that claim can be falsified with reports, media artifacts, and ablations.
