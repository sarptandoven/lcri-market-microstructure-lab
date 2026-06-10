# Preview Ladder DiT

Preview-to-final consistency for latency-aware video replacement.

## Research goal

Build a public research paper and benchmark for masked video replacement where fast previews are not throwaway samples. The preview should predict and constrain the final render so creators can trust early feedback and reach an acceptable edit in fewer iterations.

## Selected topic

**Preview-Final Consistent Video Replacement**

The core claim: for production video editing, the speed problem is not only final render latency. It is the time wasted discovering that a prompt, mask, replacement, or motion path will fail. A useful diffusion video system should produce a low-latency preview that is structurally predictive of the final render, then reuse accepted preview structure during final generation.

## First implementation tranche

This repo now contains a dependency-light benchmark scaffold:

- `preview_ladder_dit/api.py`: public model-agnostic evaluation API that computes metrics, extracts commitments, builds a scorecard, and emits a validated v0.3 run report from decoded source/preview/final/mask arrays.
- `preview_ladder_dit/metrics.py`: public preview-final metric contract, now including L1 drift, boundary drift, temporal flicker, background preservation, trajectory center drift, mask occupancy drift, temporal edge jitter, confidence-weighted L1, occupancy IoU error, low-frequency drift, local temporal residual flicker, and commitment-weighted error.
- `preview_ladder_dit/commitments.py`: dependency-light preview commitment packet extraction with bbox, centroid, occupancy, low-frequency appearance, boundary uncertainty, release fraction, and lock budget.
- `preview_ladder_dit/fixtures.py`: deterministic synthetic fixture generator for clean and known-bad masked replacement cases, now with expected metric bands for CI calibration.
- `preview_ladder_dit/benchmark.py`: public synthetic benchmark manifest writer, manifest validator, benchmark runner, aggregate JSON summary, and CSV metric export.
- `preview_ladder_dit/schema.py`: v0.3 report JSON contract dataclasses and validators for submission identity, artifact hashes, metrics, artifacts, commitments, uncertainty, model metadata, environment metadata, task metadata, and preview/final latency milestones.
- `preview_ladder_dit/harness.py`: synthetic experiment runner that writes validated v0.3 reports with commitment packets.
- `preview_ladder_dit/product.py`: product scorecard gates that turn metrics into a preview-trust score and failure diagnosis.
- `preview_ladder_dit/cli.py`: CLI for fixture generation, synthetic runs, evaluation, submission validation, paper figures, report inspection, scorecards, and report validation.
- `preview_ladder_dit/ltx_adapter.py`: dependency-light LTX adapter contract and shape helpers. Real LTX imports remain deferred.

## Run checks

```bash
python3 -m pytest -q
python3 -m compileall -q preview_ladder_dit tests
```

## CLI examples

```bash
python3 -m preview_ladder_dit.cli make-fixtures --out /tmp/pld-fixtures
python3 -m preview_ladder_dit.cli run-synthetic --out /tmp/pld-run
python3 -m preview_ladder_dit.cli validate-report /tmp/pld-run/report-clean.json
python3 -m preview_ladder_dit.cli scorecard --report /tmp/pld-run/report-clean.json
python3 -m preview_ladder_dit.cli make-benchmark --out /tmp/pld-bench
python3 -m preview_ladder_dit.cli run-benchmark --manifest /tmp/pld-bench/benchmark_manifest.json --out /tmp/pld-bench-run
python3 -m preview_ladder_dit.cli run-paper-demo --out /tmp/pld-paper-demo
python3 -m preview_ladder_dit.cli export-paper-tables --reports /tmp/pld-paper-demo/reports --out /tmp/pld-paper-demo/tables
```

See `docs/fixture-taxonomy.md` for the synthetic task taxonomy, declared failure axes, detector metrics, and the boundary between synthetic calibration evidence and future LTX-backed evidence.

## First public API

```python
from preview_ladder_dit import PreviewLadderConfig, evaluate_preview_final

result = evaluate_preview_final(
    source=source_video,
    preview=fast_preview_video,
    final=final_render_video,
    mask=edit_mask,
    config=PreviewLadderConfig(
        run_id="run-001",
        task_id="replace-001",
        prompt="replace the masked object while preserving motion and background",
        model={"backend": "ltx-2.3-distilled-preview-to-hq-final", "ltx_optional": True},
    ),
)
result.write_report_json("report.json")
print(result.scorecard.trust_score)
```

For lower-level callers, `preview_final_consistency_report`, `extract_commitments`, and `build_preview_scorecard` remain importable directly.

All metrics are lower-is-better:

- `preview_final_l1`
- `boundary_consistency_error`
- `temporal_flicker_delta`
- `background_preservation_error`
- `trajectory_center_error`
- `mask_occupancy_delta`
- `temporal_edge_jitter`
- `confidence_weighted_l1`
- `occupancy_iou_error`
- `low_frequency_drift`
- `local_temporal_residual_flicker`
- `commitment_weighted_error`

## Key research direction

The next tranche should turn a fast LTX 2.3 preview into uncertainty-aware commitments: keyframes, mask occupancy, trajectory, low-frequency appearance, and boundary confidence. The final two-stage or HQ LTX render should preserve stable commitments while allowing high-uncertainty boundary and occlusion regions to improve.

## Benchmark framing

The strongest current thesis is that preview trust is a measurable systems objective. A method should not be rewarded merely for producing a fast preview or a high-quality final. It must make the preview a falsifiable commitment object: accepted layout, trajectory, occupancy, low-frequency appearance, boundary confidence, and background locks should either survive final rendering or be explicitly marked as uncertain and released. The synthetic benchmark is the dependency-free calibration layer for that claim. Real LTX and other video-DiT backends can plug into the same report schema later.
