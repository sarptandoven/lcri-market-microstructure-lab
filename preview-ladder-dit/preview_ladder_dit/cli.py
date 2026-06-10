from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fixtures import FIXTURE_CASES, write_fixtures
from .harness import run_synthetic
from .schema import load_and_validate_run_report
from .product import build_preview_scorecard
from .benchmark import run_benchmark, validate_benchmark_manifest, write_synthetic_benchmark
from .cli_tools import (
    compare_command,
    evaluate_fixtures_command,
    export_paper_tables_command,
    generate_fixtures_command,
    inspect_report_command,
    paper_figures_command,
    run_paper_demo_command,
    validate_submission_command,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preview-ladder-dit")
    sub = parser.add_subparsers(dest="command", required=True)

    fixtures = sub.add_parser("make-fixtures", help="write deterministic synthetic preview/final fixtures")
    fixtures.add_argument("--out", required=True)
    fixtures.add_argument("--cases", nargs="+", choices=FIXTURE_CASES, default=list(FIXTURE_CASES))
    fixtures.add_argument("--frames", type=int, default=8)
    fixtures.add_argument("--height", type=int, default=32)
    fixtures.add_argument("--width", type=int, default=32)

    generate = sub.add_parser("generate-fixtures", help="write benchmark fixture JSON and manifest")
    generate.add_argument("--out", required=True)
    generate.add_argument("--cases", nargs="+", choices=FIXTURE_CASES, default=list(FIXTURE_CASES))
    generate.add_argument("--frames", type=int, default=8)
    generate.add_argument("--height", type=int, default=32)
    generate.add_argument("--width", type=int, default=32)

    evaluate = sub.add_parser("evaluate", help="score fixture JSON files and write reports, summary, and CSV")
    evaluate.add_argument("--fixtures", required=True, help="fixture JSON file or directory")
    evaluate.add_argument("--out", required=True)

    submission = sub.add_parser("validate-submission", help="validate a directory of benchmark run reports")
    submission.add_argument("path")

    compare = sub.add_parser("compare", help="compare two report directories or metrics CSV files")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--out")

    figures = sub.add_parser("paper-figures", help="emit CSVs and Vega-Lite specs for paper figures")
    figures.add_argument("--reports", required=True)
    figures.add_argument("--out", required=True)

    tables = sub.add_parser("export-paper-tables", help="emit reviewer-facing CSV tables from benchmark reports")
    tables.add_argument("--reports", required=True)
    tables.add_argument("--out", required=True)

    inspect = sub.add_parser("inspect-report", help="print sorted metrics and warnings for one report")
    inspect.add_argument("report")

    scorecard = sub.add_parser("scorecard", help="emit product preview-trust gates for one report")
    scorecard.add_argument("--report", required=True)
    scorecard.add_argument("--out")

    synthetic = sub.add_parser("run-synthetic", help="generate fixtures and report JSON metrics")
    synthetic.add_argument("--out", required=True)
    synthetic.add_argument("--cases", nargs="+", choices=FIXTURE_CASES, default=list(FIXTURE_CASES))
    synthetic.add_argument("--frames", type=int, default=8)
    synthetic.add_argument("--height", type=int, default=32)
    synthetic.add_argument("--width", type=int, default=32)

    bench_manifest = sub.add_parser("make-benchmark", help="write synthetic benchmark fixtures plus benchmark_manifest.json")
    bench_manifest.add_argument("--out", required=True)
    bench_manifest.add_argument("--cases", nargs="+", choices=FIXTURE_CASES, default=list(FIXTURE_CASES))
    bench_manifest.add_argument("--frames", type=int, default=8)
    bench_manifest.add_argument("--height", type=int, default=32)
    bench_manifest.add_argument("--width", type=int, default=32)

    run_bench = sub.add_parser("run-benchmark", help="run a benchmark manifest and write reports plus aggregate CSV/JSON")
    run_bench.add_argument("--manifest", required=True)
    run_bench.add_argument("--out", required=True)

    validate_bench = sub.add_parser("validate-benchmark", help="validate benchmark_manifest.json")
    validate_bench.add_argument("manifest")

    paper_demo = sub.add_parser("run-paper-demo", help="generate fixtures, run benchmark, validate reports, and export paper figures")
    paper_demo.add_argument("--out", required=True)
    paper_demo.add_argument("--cases", nargs="+", choices=FIXTURE_CASES, default=list(FIXTURE_CASES))
    paper_demo.add_argument("--frames", type=int, default=8)
    paper_demo.add_argument("--height", type=int, default=32)
    paper_demo.add_argument("--width", type=int, default=32)

    validate = sub.add_parser("validate-report", help="validate a report JSON contract")
    validate.add_argument("report")

    args = parser.parse_args(argv)
    if args.command == "make-fixtures":
        paths = write_fixtures(args.out, cases=args.cases, frames=args.frames, height=args.height, width=args.width)
        print(json.dumps({"files": [str(p) for p in paths]}, indent=2))
        return 0
    if args.command == "generate-fixtures":
        print(json.dumps(generate_fixtures_command(args.out, cases=args.cases, frames=args.frames, height=args.height, width=args.width), indent=2))
        return 0
    if args.command == "evaluate":
        print(json.dumps(evaluate_fixtures_command(args.fixtures, args.out), indent=2))
        return 0
    if args.command == "validate-submission":
        result = validate_submission_command(args.path)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2
    if args.command == "compare":
        print(json.dumps(compare_command(args.left, args.right, out=args.out), indent=2))
        return 0
    if args.command == "paper-figures":
        print(json.dumps(paper_figures_command(args.reports, args.out), indent=2))
        return 0
    if args.command == "export-paper-tables":
        print(json.dumps(export_paper_tables_command(args.reports, args.out), indent=2))
        return 0
    if args.command == "inspect-report":
        print(json.dumps(inspect_report_command(args.report), indent=2))
        return 0
    if args.command == "scorecard":
        report = load_and_validate_run_report(Path(args.report))
        data = build_preview_scorecard(report).to_dict()
        if args.out:
            Path(args.out).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(data, indent=2))
        return 0
    if args.command == "run-synthetic":
        paths = run_synthetic(args.out, cases=args.cases, frames=args.frames, height=args.height, width=args.width)
        print(json.dumps({"reports": [str(p) for p in paths]}, indent=2))
        return 0
    if args.command == "make-benchmark":
        path = write_synthetic_benchmark(args.out, cases=args.cases, frames=args.frames, height=args.height, width=args.width)
        print(json.dumps({"manifest": str(path)}, indent=2))
        return 0
    if args.command == "run-benchmark":
        print(json.dumps(run_benchmark(args.manifest, args.out), indent=2))
        return 0
    if args.command == "validate-benchmark":
        validate_benchmark_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
        print(json.dumps({"ok": True, "manifest": args.manifest}, indent=2))
        return 0
    if args.command == "run-paper-demo":
        result = run_paper_demo_command(args.out, cases=args.cases, frames=args.frames, height=args.height, width=args.width)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "validate-report":
        load_and_validate_run_report(Path(args.report))
        print(json.dumps({"ok": True, "report": args.report}, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
