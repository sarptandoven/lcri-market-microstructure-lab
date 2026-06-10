# Landscape and Novelty Map

## Current LTX 2.3 and LTX-Video context

Relevant public sources:

- LTX-2 repo: https://github.com/Lightricks/LTX-2
- LTX-2.3 model card: https://huggingface.co/Lightricks/LTX-2.3
- LTX-Video repo: https://github.com/Lightricks/LTX-Video
- LTX-Video paper: https://arxiv.org/abs/2501.00103
- LTX-2 paper: https://arxiv.org/abs/2601.03233

Observed capabilities from public docs and repo inspection:

- LTX-Video reported faster-than-real-time latent video generation in the original paper and repo materials.
- LTX-2.3 exposes full and distilled checkpoints, 8-step distilled paths, two-stage production pipelines, spatial upscalers, retake, IC-LoRA controls, video-to-video controls, motion track controls, and audio-video generation.
- Retake appears primarily temporal-window based, not full object/mask replacement.
- The pipeline structure has natural preview/final affordances, but public docs do not define a preview-to-final consistency benchmark or guarantee.

## Crowded acceleration areas

- Generic feature caching: DeepCache, FasterCache, TeaCache, Adaptive Caching, Pyramid Attention Broadcast.
- Generic sparse attention: Sparse VideoGen, Sparse VideoGen2, Re-ttention, FlashAttention-style backend work.
- Token merging and pruning: ToMe for Stable Diffusion and newer diffusion token merge variants.
- Few-step and consistency distillation: Consistency Models, Latent Consistency Models, VideoLCM, AnimateLCM, DMD, DMD2, LADD, one-step video models.

These are important, but a paper framed only as faster video DiT inference would be competing in a crowded lane.

## Gap this project targets

Preview-to-final consistency is a product-native latency problem. A fast preview that does not predict the final render is not actually useful latency reduction. For masked replacement workflows, the system needs to answer:

- Does the preview preserve the source background?
- Will the replacement identity survive final rendering?
- Are object trajectory and mask occupancy stable?
- Are mask boundaries and soft mattes stable?
- Does final rendering change the approved preview in surprising ways?
- Can accepted preview regions be reused instead of regenerated?

## Why this is likely underexplored

Academic acceleration work usually reports samples per second, FVD, CLIP-like quality, or final generation speed. Product editing workflows care about iteration speed and trust. The key unit is not one final video. It is the number of seconds and attempts until an editor accepts the result.

## Defensible novelty claim

This project should claim **latency-aware preview/final consistency for masked video replacement**, not generic DiT acceleration. It can use caching, adaptive compute, or LTX hooks, but those are mechanisms. The paper contribution is the workflow-level objective and the reusable benchmark/method.
