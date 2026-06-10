# Paper Draft: Preview Ladder DiT

## Working title

Preview Ladder DiT: Preview-Final Consistent Video Replacement

## Abstract draft

Video diffusion editing systems often expose fast previews, but those previews are usually operational conveniences rather than reliable commitments. In masked video replacement, this creates a workflow latency problem: creators spend time approving prompts, masks, and motion paths only to discover that the final render changes identity, boundary behavior, temporal stability, or source background preservation. Preview Ladder DiT frames preview trust as a first-class research objective. We introduce a benchmark scaffold for measuring preview-final consistency and propose a preview-conditioned final renderer that preserves accepted low-latency structure while refining uncertain regions. The initial implementation provides deterministic synthetic fixtures, a JSON report contract, latency logging, and dependency-light metrics for preview-final drift, boundary consistency, temporal flicker, background preservation, trajectory drift, mask occupancy drift, and temporal edge jitter. The planned LTX 2.3 integration uses distilled or low-resolution preview paths and two-stage final generation, with uncertainty-aware commitments passed through latent, keyframe, mask, and denoising-loop hooks.

## Thesis

For masked video replacement, the latency that matters is time-to-acceptable-output, not only final render wall clock. A preview is useful when it is predictive enough to support user decisions and reusable enough to constrain the final render.

## Contributions

1. A public benchmark contract for preview-final consistency in masked video replacement.
2. Synthetic fixtures with controlled failure modes: identity drift, boundary halo, background leak, motion mismatch, flicker, occlusion failure, mask instability, thin structure loss, illumination pulse, shadow leak, and parallax mismatch.
3. Deterministic scalar metrics and expected metric bands that can run in CI and later be complemented by perceptual, flow, and embedding metrics.
4. A v0.2 report JSON contract with explicit metrics, commitment packets, uncertainty artifacts, task metadata, model metadata, environment metadata, and latency events.
5. A benchmark manifest and CLI harness that emit validated per-task reports, aggregate JSON summaries, and CSV tables for paper figures.
6. A product scorecard API that maps preview-final metrics to trust gates, diagnoses, and a creator-facing pass/fail signal.
7. A model-agnostic `evaluate_preview_final` public API that turns decoded source, preview, final, and mask arrays into metrics, commitments, scorecard, and a validated v0.2 report without importing LTX or other heavy backends.
8. An LTX 2.3 adapter plan that wraps existing distilled preview, retake, and two-stage final pipelines rather than forking LTX.

## Method sketch

1. Generate a cheap preview with a low-resolution, few-step, distilled, or sparse pipeline.
2. Extract preview commitments: object layout, mask occupancy, trajectory, keyframes, low-frequency color/material statistics, and boundary bands.
3. Estimate uncertainty from soft boundaries, occlusions, fast motion, thin structures, and preview instability.
4. Run final generation conditioned on accepted commitments.
5. Release high-uncertainty regions for additional compute while preserving stable background and replacement structure.
6. Report preview-final consistency, latency to first preview, final latency, and latency-quality Pareto position.

## LTX 2.3 implementation path

The adapter should remain a thin wrapper. Public LTX-2.3 exposes useful split points: distilled preview pipelines, one-stage and two-stage generation, retake temporal masks, `DiffusionStage.run`, `ModalitySpec.initial_latent`, latent and keyframe conditioning, and denoising masks. The first real backend should run two modes:

- `preview_only`: distilled or low-resolution preview with latency and shape logging.
- `preview_then_final`: final two-stage or HQ render with report JSON comparing decoded preview and final arrays.

A later backend should downsample the user mask to latent token space and use preview commitments to freeze trusted background/replacement tokens while allowing uncertain boundary and occlusion regions to denoise.

## Evaluation protocol

Report the full Pareto frontier over quality, preview trust, time-to-first-preview, final latency, and total time-to-acceptable-output. The benchmark records both raw lower-is-better metrics, expected metric band calibration checks, and product gates. The current no-dependency gate set covers preview-final L1, boundary consistency, temporal flicker, background preservation, trajectory drift, mask occupancy, temporal edge jitter, and commitment-weighted error. Keep a scalar utility only as a secondary leaderboard convenience. Reject runs that violate artifact shape consistency, missing metric fields, non-monotonic latency events, unreported seeds/model IDs, or manifest tasks with unknown fixture cases.

## Public API draft

The public surface should remain model-agnostic:

```python
from preview_ladder_dit import PreviewLadderConfig, evaluate_preview_final

result = evaluate_preview_final(
    source=source,
    preview=preview,
    final=final,
    mask=mask,
    config=PreviewLadderConfig(run_id=run_id, task_id=task_id, prompt=prompt),
)
metrics = result.metrics.to_dict()
packet = result.commitments.to_dict()
scorecard = result.scorecard.to_dict()
report = result.report.to_dict()
```

The API deliberately consumes decoded scalar features or arrays rather than model objects. A product backend can run LTX, ComfyUI, Magic Hour-style hosted jobs, or another video DiT, then use the same call to produce a validated report and trust score.

The optional LTX backend should accept the same concepts through `LTXReplacementRequest`: preview steps, final steps, scheduler, mask policy, commitment strength, and backend options. Real LTX imports stay optional so the benchmark, fixtures, schema validation, CLI, and paper figures remain runnable in CPU-only CI.
