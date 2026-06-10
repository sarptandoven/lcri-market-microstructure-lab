# Experiment Plan

## Phase 1: deterministic benchmark scaffold

Use synthetic videos represented as frame-major arrays. Build preview/final pairs with controlled drift:

- clean preview-final match
- identity drift inside the mask
- boundary halo around the mask
- background leakage outside the mask
- temporal flicker inside the mask
- motion trajectory mismatch
- occlusion failure
- mask instability
- thin structure loss at the boundary
- coherent illumination pulse inside the replacement
- shadow leak into locked background
- parallax or row-stable motion mismatch

Implemented entry points:

```bash
python3 -m preview_ladder_dit.cli make-fixtures --out /tmp/pld-fixtures
python3 -m preview_ladder_dit.cli run-synthetic --out /tmp/pld-run
python3 -m preview_ladder_dit.cli validate-report /tmp/pld-run/report-clean.json
python3 -m preview_ladder_dit.cli make-benchmark --out /tmp/pld-bench
python3 -m preview_ladder_dit.cli validate-benchmark /tmp/pld-bench/benchmark_manifest.json
python3 -m preview_ladder_dit.cli run-benchmark --manifest /tmp/pld-bench/benchmark_manifest.json --out /tmp/pld-bench-run
```

Each synthetic task now carries expected metric bands. These are not publication results. They are calibration assertions: clean should be zero, each known-bad fixture should excite the metric family it was designed to stress, and future perceptual or LTX-backed metrics should preserve the same failure taxonomy.

## Phase 2: report schema and latency contract

Every run writes a JSON report with:

- `schema_version`
- `run_id` and `task_id`
- preview and final artifact metadata when real media exists
- lower-is-better metrics
- monotonic latency events
- model metadata
- environment metadata

The deterministic harness writes synthetic reports without real media artifacts. Real LTX runs should add SHA-256, dimensions, frame count, fps, duration, model IDs, seeds, step counts, and device metadata.

## Phase 3: LTX 2.3 dry-run integration

Use existing LTX pipeline hooks where possible:

- preview path: distilled or low-resolution run
- final path: two-stage or HQ run
- mask path: source video mask track, dilated boundary, and temporal confidence
- report path: JSON metrics and latency logs

The current `ltx_adapter.py` fixes the request/result contract and LTX latent shape helper while deferring real LTX imports. The next code step is a backend subclass that calls LTX only when optional dependencies and checkpoints are present.

## Phase 4: preview-conditioned final pass

Prototype commitment extraction:

- preview keyframes
- replacement bounding boxes or masks
- boundary confidence map
- temporal trajectory summary
- low-frequency color/material statistics

Condition final generation on these commitments. Measure drift against an independent final render.

## Phase 5: latency-quality Pareto curves

Report:

- time-to-first-preview
- final render latency
- preview-to-final drift
- boundary consistency
- temporal flicker delta
- background preservation
- trajectory center drift
- mask occupancy delta
- temporal edge jitter
- human or model-assisted preference when available

Primary comparisons should include at least four methods: independent preview/final generation as the negative control, preview latent or keyframe reuse, commitment-conditioned final rendering, and an oracle or human-selected preview upper bound. Publish all seeds and use paired uncertainty estimates over task-level preview-trust deltas before claiming a method is better.

## First implementation target

Keep the public report call stable:

```python
preview_final_consistency_report(source=source, preview=preview, final=final, mask=mask)
```

Return lower-is-better metrics with clear units and no dependency on LTX internals.
