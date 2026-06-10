# Prototype harness runner

Dependency-light runner for Preview Ladder DiT experiments.

What it does:
- Reads a task JSON file containing one task or `tasks[]`.
- Runs named adapters.
- Records latency events with role labels.
- Computes preview-final consistency metrics using the existing metric contract.
- Adds aggregate latency metrics.
- Writes one validated `report.json` per run plus `aggregate.json`.

Supported adapters:
- `fixture`: generate deterministic synthetic fixtures from `preview_ladder_dit.fixtures`.
- `json_file`: load a JSON object with `source`, `preview`, `final`, and `mask` arrays.
- `command_json`: run a command that prints that same JSON object to stdout.

Run:

```bash
python3 prototypes/harness_runner/runner.py prototypes/harness_runner/sample_tasks.json --out prototypes/harness_runner/out
python3 -m preview_ladder_dit.cli validate-report prototypes/harness_runner/out/fixture-clean-run/report.json
python3 -m preview_ladder_dit.cli validate-report prototypes/harness_runner/out/fixture-temporal-flicker-run/report.json
```

The `command_json` adapter is the integration seam for real preview/final engines. A target-repo version should pass task fields and artifact paths through environment variables or stdin once the production adapter contract is finalized.
