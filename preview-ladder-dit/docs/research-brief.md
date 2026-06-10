# Preview Ladder DiT: Preview-Final Consistent Video Replacement

## Selected topic

**Preview-Final Consistent Video Replacement**: a latency-aware video diffusion editing method and benchmark where a fast preview is not a disposable low-quality render. It becomes a structural commitment that the final render must preserve unless the system explicitly reports uncertainty or asks for another iteration.

## One-line thesis

For masked video replacement, the core latency problem is not only final render speed. It is the cost of discovering whether an edit will work. A video diffusion system should produce cheap previews that predict final renders, then reuse accepted preview structure so time-to-acceptable-output falls sharply.

## Why this fits Sarp and Magic Hour-style workflows

This targets production replacement workflows: masks, boundaries, object identity, lighting, flicker, and user iteration. It is close to Magic Hour-style video editing and General Replace-style systems, but the research framing is broader and public: every developer building AI video editing needs a way to measure and improve preview trust.

## What is novel

Most diffusion acceleration work optimizes final-frame throughput: caching, sparse attention, token merging, distillation, and few-step samplers. Products often expose previews, but public research rarely treats preview-to-final consistency as a first-class objective. The novel claim is not "make video DiTs faster." The claim is: **make latency useful by guaranteeing that fast previews are predictive and reusable for final masked edits.**

## Proposed contributions

1. A public benchmark for preview-to-final consistency in masked video replacement.
2. Metrics for semantic, spatial, temporal, boundary, and background consistency between preview and final.
3. A preview-conditioned final renderer that locks accepted low-latency structure while refining detail.
4. A latency-quality protocol measuring time-to-first-preview, preview-to-final drift, and time-to-acceptable-output.
5. Optional LTX 2.3 integration using low-resolution/distilled preview stages and higher-resolution final stages.

## Method sketch

1. Generate a cheap preview with a low-resolution, few-step, or sparse LTX pipeline.
2. Convert the preview into commitments: object layout, mask occupancy, trajectory, keyframes, color/material statistics, and boundary bands.
3. Estimate uncertainty: regions likely to drift between preview and final, such as thin structures, occlusions, fast motion, and soft mask boundaries.
4. Run final generation conditioned on the preview commitments.
5. Recompute high-uncertainty regions with higher compute, while preserving accepted regions.
6. Report preview-final consistency and latency metrics.

## Initial hypothesis

A preview-conditioned final pass can reduce user-visible iteration cost more than a uniformly faster sampler, because it makes the first preview actionable. Even if final wall-clock speed is unchanged, total workflow latency falls when users can trust and approve previews earlier.

## Baselines

- Full final render with no preview.
- Fast preview and independent final render.
- Low-resolution preview upscaled to final without consistency constraints.
- Globally fewer diffusion steps.
- Generic feature/cache acceleration.
- Mask-only latent blending or retake without preview commitment.

## LTX 2.3 integration path

LTX 2.3 already exposes a two-stage production structure, distilled fast variants, IC-LoRA controls, retake/time-window workflows, and custom denoising-loop hooks. The first implementation should not fork LTX. It should wrap the existing pipeline:

- preview: distilled or low-resolution stage
- commitment extraction: masks, trajectories, boundary bands, preview keyframes
- final: two-stage or HQ pipeline conditioned on preview commitments
- evaluation: preview-final report plus latency logs

## Research risk

The main risk is that a final render constrained too strongly to the preview may preserve preview errors. The system needs uncertainty-aware commitments: lock stable structure, but allow high-risk regions to improve.
