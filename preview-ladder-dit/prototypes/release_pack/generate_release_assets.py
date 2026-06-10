#!/usr/bin/env python3
"""Generate concrete public-release collateral for Preview Ladder DiT.

This prototype intentionally writes only into the caller-provided output directory.
It does not import target-repo internals, so it can be used as a standalone design
artifact before integrating into the main package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LEADERBOARD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://preview-ladder-dit.org/schemas/leaderboard-entry-v0.1.json",
    "title": "Preview Ladder DiT leaderboard entry",
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "task_suite",
        "method",
        "metrics",
        "latency",
        "artifacts",
        "reproducibility",
    ],
    "properties": {
        "schema_version": {"const": "preview-ladder-leaderboard/v0.1"},
        "run_id": {"type": "string", "minLength": 3},
        "task_suite": {"enum": ["synthetic-v0", "masked-replacement-open-v0", "ltx2.3-dev-v0"]},
        "method": {
            "type": "object",
            "required": ["name", "preview_backend", "final_backend", "commitment_policy"],
            "properties": {
                "name": {"type": "string"},
                "preview_backend": {"type": "string"},
                "final_backend": {"type": "string"},
                "commitment_policy": {"type": "string"},
                "paper_url": {"type": "string"},
                "code_url": {"type": "string"},
            },
        },
        "metrics": {
            "type": "object",
            "required": [
                "preview_final_l1",
                "boundary_consistency_error",
                "temporal_flicker_delta",
                "background_preservation_error",
                "trajectory_center_error",
                "mask_occupancy_delta",
                "temporal_edge_jitter",
                "acceptance_weighted_latency_s",
            ],
            "additionalProperties": {"type": "number"},
        },
        "latency": {
            "type": "object",
            "required": ["time_to_first_preview_s", "final_latency_s", "total_latency_s", "hardware"],
            "properties": {
                "time_to_first_preview_s": {"type": "number", "minimum": 0},
                "final_latency_s": {"type": "number", "minimum": 0},
                "total_latency_s": {"type": "number", "minimum": 0},
                "hardware": {"type": "string"},
            },
        },
        "artifacts": {
            "type": "object",
            "required": ["report_json", "preview_grid", "final_grid", "failure_grid"],
            "additionalProperties": {"type": "string"},
        },
        "reproducibility": {
            "type": "object",
            "required": ["seed", "container", "model_hashes", "command"],
            "properties": {
                "seed": {"type": "integer"},
                "container": {"type": "string"},
                "model_hashes": {"type": "object", "additionalProperties": {"type": "string"}},
                "command": {"type": "string"},
            },
        },
    },
}

README_STORY = """# Preview Ladder DiT

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
"""

ISSUE_TEMPLATE = """name: Benchmark submission
about: Submit a Preview Ladder DiT run for review
labels: benchmark-submission

## Method
- Name:
- Code URL:
- Paper or technical note URL:
- Preview backend:
- Final backend:
- Commitment policy:

## Reproduction
- Command:
- Container or environment:
- Hardware:
- Seeds:
- Model hashes:

## Artifacts
- report.json:
- preview grid:
- final grid:
- failure grid:
- raw media archive:

## Expected review checks
- [ ] Report validates against schema
- [ ] Latency events are monotonic
- [ ] Model hashes and seeds are present
- [ ] No private or non-redistributable source footage
- [ ] Failure cases are included, not only best cases
"""

CONTRIBUTING = """# Contributing

Preview Ladder DiT contributions should improve falsifiability, reproducibility, or product usability.

## Good first contributions

- Add a synthetic fixture with one isolated failure mode.
- Add a metric with a documented unit and lower-is-better direction.
- Add a benchmark adapter that writes the public report schema without requiring private services.
- Add a failure demo with source, mask, preview, final, and report JSON.

## Benchmark submission rules

1. Include the exact command, seed, model identifiers, model hashes when available, and hardware.
2. Submit the full metric vector, not only a scalar rank.
3. Include at least one failure grid per method.
4. Do not use private videos unless you can redistribute them with masks.
5. Label hand-picked demos as demos, not benchmark results.

## Review principle

A method that is slower but makes previews meaningfully more predictive can be a valid contribution. A method that only improves final visual quality while breaking preview-final consistency is out of scope unless it is clearly labeled as a baseline.
"""

DEMO_MANIFEST = {
    "schema_version": "preview-ladder-demo-manifest/v0.1",
    "required_assets_per_demo": [
        "source.mp4",
        "mask.mp4 or mask.rle.json",
        "preview.mp4",
        "final.mp4",
        "side_by_side.mp4",
        "report.json",
        "README.md",
    ],
    "demo_sets": [
        {
            "name": "clean_commitment",
            "purpose": "Shows preview structure preserved through final refinement.",
            "must_show": ["source", "mask", "preview", "final", "metric table"],
        },
        {
            "name": "boundary_failure",
            "purpose": "Shows why boundary consistency and temporal edge jitter matter.",
            "must_show": ["boundary crop", "temporal crop", "failure annotation"],
        },
        {
            "name": "motion_mismatch",
            "purpose": "Shows trajectory drift between accepted preview and final render.",
            "must_show": ["centroid trace overlay", "preview/final split"],
        },
    ],
}


def write_assets(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README_RELEASE_DRAFT.md").write_text(README_STORY, encoding="utf-8")
    (out_dir / "CONTRIBUTING_DRAFT.md").write_text(CONTRIBUTING, encoding="utf-8")
    (out_dir / "ISSUE_TEMPLATE_BENCHMARK_SUBMISSION.md").write_text(ISSUE_TEMPLATE, encoding="utf-8")
    (out_dir / "leaderboard_entry.schema.json").write_text(json.dumps(LEADERBOARD_SCHEMA, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "demo_manifest.json").write_text(json.dumps(DEMO_MANIFEST, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output directory for generated release collateral")
    args = parser.parse_args()
    write_assets(Path(args.out))
    print(json.dumps({"ok": True, "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
