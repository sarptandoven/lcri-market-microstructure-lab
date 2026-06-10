# Maximal Preview Ladder DiT Research Implementation Plan

> **For Hermes:** Use subagent-driven-development and research-product-engineering. Implement in coherent tranches. Agents must write isolated reports and patches. The controller integrates and verifies.

**Goal:** Turn Preview Ladder DiT into a serious public research artifact: a benchmark, method scaffold, LTX-compatible integration path, reproducible experiments, paper draft, and API product for preview-final consistent masked video replacement.

**Architecture:** Keep the core package dependency-light and deterministic. Put research claims behind reproducible fixtures, schemas, metrics, and CLI commands. Keep real LTX integration optional through adapter contracts until GPU-backed experiments are available.

**Tech Stack:** Python 3.9+, stdlib-only core, pytest, JSON schemas via dataclass validators, CLI module, optional future backends for LTX 2.3 and perceptual metrics.

---

## Research thesis

For production masked video replacement, the relevant latency objective is time-to-acceptable-output. A preview is useful only when it is predictive enough for user decisions and reusable enough to constrain the final render. Preview Ladder DiT should measure, expose, and improve preview trust by converting approved previews into uncertainty-aware commitments for layout, trajectory, occupancy, appearance, boundaries, and background preservation.

The public artifact should make three things falsifiable:

1. Whether a fast preview predicts the final render.
2. Whether accepted preview structure survives final generation.
3. Whether the system reduces iteration cost, not just wall-clock final render time.

## Tranche 1: lock the public benchmark contract

**Objective:** Make the benchmark report contract hard to game and useful for API consumers.

**Files:**
- Modify: `preview_ladder_dit/schema.py`
- Modify: `preview_ladder_dit/benchmark.py`
- Modify: `preview_ladder_dit/harness.py`
- Modify: `tests/test_benchmark.py`
- Modify: `tests/test_fixtures_harness.py`

**Tasks:**
1. Add submission identity fields: method name, method version, backend family, preview mode, final mode, artifact hashes, and report schema version.
2. Validate latency events are monotonic and include `preview_start`, `preview_end`, `final_start`, `final_end` where applicable.
3. Add benchmark-level aggregate fields: mean trust score, pass rate, metric means, metric p95s, and rejected report count.
4. Add tests for valid reports, missing identity fields, invalid latency order, and unknown fixture cases.

**Verification:**
- `python3 -m pytest tests/test_benchmark.py tests/test_fixtures_harness.py -q`
- `python3 -m compileall -q preview_ladder_dit tests`

## Tranche 2: strengthen the metric suite

**Objective:** Add metrics that directly correspond to reviewer-observable video editing failures.

**Files:**
- Modify: `preview_ladder_dit/metrics.py`
- Modify: `preview_ladder_dit/product.py`
- Modify: `tests/test_metrics.py`

**Tasks:**
1. Add boundary signed bias: detects final systematically bleeding inside or outside the mask boundary.
2. Add background temporal leak: detects background flicker introduced outside the mask.
3. Add trajectory acceleration drift: detects motion changes that centroid distance misses.
4. Add preview confidence release rate: records how much of the preview was intentionally not committed.
5. Update scorecard gates and diagnoses to distinguish boundary, background, motion, and appearance failures.

**Verification:**
- `python3 -m pytest tests/test_metrics.py tests/test_commitments_product_schema.py -q`

## Tranche 3: build public synthetic tasks that matter

**Objective:** Make the fixture suite look like a real masked replacement benchmark, not toy arrays.

**Files:**
- Modify: `preview_ladder_dit/fixtures.py`
- Modify: `preview_ladder_dit/benchmark.py`
- Modify: `tests/test_fixtures_harness.py`
- Create: `docs/fixture-taxonomy.md`

**Tasks:**
1. Add cases for thin structures, edge occlusion, parallax mismatch, illumination pulse, shadow leak, camera pan, and fast crossing motion.
2. Add expected metric band calibration for every case.
3. Add fixture metadata fields: failure axis, expected detector metrics, generation seed, and task difficulty.
4. Document why each fixture corresponds to a real production editing failure.

**Verification:**
- `python3 -m pytest tests/test_fixtures_harness.py -q`
- `python3 -m preview_ladder_dit.cli make-benchmark --out /tmp/pld-bench`
- `python3 -m preview_ladder_dit.cli run-benchmark --manifest /tmp/pld-bench/benchmark_manifest.json --out /tmp/pld-run`

## Tranche 4: define the actual method scaffold

**Objective:** Make Preview Ladder DiT more than a metric library by adding a concrete preview commitment policy.

**Files:**
- Modify: `preview_ladder_dit/commitments.py`
- Create: `preview_ladder_dit/policy.py`
- Create: `tests/test_policy.py`
- Modify: `preview_ladder_dit/ltx_adapter.py`

**Tasks:**
1. Add a `CommitmentPolicy` dataclass with lock strengths for background, boundary, trajectory, occupancy, and appearance.
2. Add deterministic policy selection from metrics and uncertainty proxies.
3. Add a latent-token mask projection helper that maps pixel masks to latent grid coordinates.
4. Add an adapter method that produces backend-agnostic conditioning hints for LTX-style denoising loops.
5. Test policy behavior on clean, boundary halo, flicker, and motion mismatch fixtures.

**Verification:**
- `python3 -m pytest tests/test_policy.py tests/test_commitments_product_schema.py -q`

## Tranche 5: make the CLI a publishable artifact

**Objective:** Give users one command path from fixtures to report to figures.

**Files:**
- Modify: `preview_ladder_dit/cli.py`
- Modify: `preview_ladder_dit/cli_tools.py`
- Create: `docs/cli-quickstart.md`
- Modify: `README.md`

**Tasks:**
1. Add `run-paper-demo` command that generates fixtures, runs benchmark, validates reports, writes CSVs, and prints the trust summary.
2. Add `inspect-report` output that explains failures in product language.
3. Add `export-paper-tables` command for aggregate result tables.
4. Add README quickstart with exact commands.

**Verification:**
- `python3 -m preview_ladder_dit.cli run-paper-demo --out /tmp/pld-paper-demo`
- `python3 -m pytest tests/test_cli_prototype.py -q`

## Tranche 6: raise the paper draft to serious conference shape

**Objective:** Convert the draft into a dense technical paper outline with claims tied to measurable artifacts.

**Files:**
- Modify: `docs/paper-draft.md`
- Create: `docs/related-work.md`
- Create: `docs/experiment-plan.md`
- Create: `docs/reviewer-risk-register.md`

**Tasks:**
1. Write full sections: abstract, introduction, problem formulation, benchmark, method, experiments, related work, limitations.
2. Add equations for preview-final consistency, commitment preservation, uncertainty release, and time-to-acceptable-output.
3. Add paper tables and figures that can be generated from current synthetic outputs.
4. Add reviewer attack surface and mitigation plan.
5. Keep novelty claims precise: benchmark and workflow objective are underexplored; mechanisms are related to existing caching, inpainting, retake, sparse attention, and distillation.

**Verification:**
- `python3 -m preview_ladder_dit.cli run-paper-demo --out /tmp/pld-paper-demo`
- Check that every paper claim maps to a report, fixture, metric, or planned LTX experiment.

## Tranche 7: prepare real LTX integration without hard dependency

**Objective:** Make the path to GPU-backed results concrete while preserving CPU-only tests.

**Files:**
- Modify: `preview_ladder_dit/ltx_adapter.py`
- Create: `docs/ltx-integration.md`
- Create: `tests/test_ltx_adapter.py`

**Tasks:**
1. Define request/response contracts for preview-only, final-only, independent preview-final, and committed preview-final modes.
2. Add shape helpers for video arrays, latent grids, masks, and keyframe commitments.
3. Add explicit hook names and expected data flow for LTX 2.3 style pipelines.
4. Add stub backend tests that verify no LTX import is required.

**Verification:**
- `python3 -m pytest tests/test_ltx_adapter.py -q`

## Tranche 8: make it hard to game

**Objective:** Make benchmark submissions credible.

**Files:**
- Modify: `preview_ladder_dit/schema.py`
- Modify: `preview_ladder_dit/benchmark.py`
- Create: `docs/submission-rules.md`
- Create: `tests/test_submission_rules.py`

**Tasks:**
1. Validate artifact hashes and required metadata.
2. Add cheating checks for missing preview latency, identical preview/final without disclosure, malformed masks, and inconsistent task IDs.
3. Add submission rules covering model IDs, seeds, hardware, precision, prompts, masks, and postprocessing.
4. Add benchmark rejection reasons to aggregate output.

**Verification:**
- `python3 -m pytest tests/test_submission_rules.py tests/test_benchmark.py -q`

## Tranche 9: produce first public evidence package

**Objective:** Make the repo demoable even before real LTX results.

**Files:**
- Create: `docs/results-synthetic.md`
- Create or update: `prototypes/demo_run/`
- Modify: `README.md`

**Tasks:**
1. Run the full synthetic benchmark.
2. Export aggregate JSON, metrics CSV, figure CSVs, and report examples.
3. Document what the synthetic evidence proves and what it does not prove.
4. Add next-step checklist for LTX experiments.

**Verification:**
- `python3 -m preview_ladder_dit.cli run-paper-demo --out prototypes/demo_run`
- `python3 -m pytest -q`

## Tranche 10: final integration and quality gates

**Objective:** Keep the repo clean and reproducible.

**Files:**
- All changed files.

**Tasks:**
1. Run all tests.
2. Run compile checks.
3. Remove generated caches that should not be tracked.
4. Inspect `git status --short` and `git diff --stat`.
5. Do not commit until Sarp authorizes a commit message.

**Verification:**
- `python3 -m pytest -q`
- `python3 -m compileall -q preview_ladder_dit tests`
- `git diff --stat`
- `git status --short`
