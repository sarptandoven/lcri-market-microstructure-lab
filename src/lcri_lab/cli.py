from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lcri_lab.evaluation import (
    evaluate_signals,
    generalization_gap_leaderboard,
    generalization_overview,
    lcri_gap_delta_flags,
    lcri_gap_delta_summary,
    lcri_generalization_gap_delta,
    lcri_generalization_gap_leaderboard,
    regime_generalization_gap,
    regime_metrics,
    signal_generalization_gap,
    transition_conditioned_metrics,
    transition_generalization_gap,
    transition_robustness_summary,
    transition_signal_lift,
)
from lcri_lab.features import add_regime_transition_features
from lcri_lab.ingest import normalize_l2_snapshots
from lcri_lab.model import ARTIFACT_VERSION, LCRIModel, ModelConfig
from lcri_lab.plotting import write_figures
from lcri_lab.reporting import (
    build_artifact_manifest,
    collect_artifact_metadata,
    missing_artifacts,
    verify_artifact_manifest,
    verify_generalization_overview,
    verify_lcri_gap_delta_flags,
    verify_lcri_gap_delta_summary,
    verify_lcri_generalization_gap_delta,
    write_json,
    write_research_summary,
)
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def main() -> None:
    parser = argparse.ArgumentParser(prog="lcri-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("run-demo", help="run the synthetic LCRI research workflow")
    demo.add_argument("--rows", type=int, default=20_000)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--train-frac", type=float, default=0.70)
    demo.add_argument("--output", type=Path, default=Path("reports"))

    normalize = subparsers.add_parser("normalize", help="normalize flat L2 snapshots")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--tick-size", type=float, required=True)
    normalize.add_argument("--levels", type=int, default=5)
    normalize.add_argument("--derive-state", action="store_true")

    fit = subparsers.add_parser("fit", help="fit an LCRI model from order book snapshots")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--model", type=Path, required=True)
    fit.add_argument("--levels", type=int, default=5)
    fit.add_argument("--ridge", type=float, default=1e-3)
    fit.add_argument("--probability-scale", type=float, default=1.0)

    score = subparsers.add_parser("score", help="score order book snapshots with a fitted model")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--columns", help="comma-separated output columns; defaults to all columns")

    describe = subparsers.add_parser("describe-model", help="print fitted model artifact metadata")
    describe.add_argument("--model", type=Path, required=True)

    verify = subparsers.add_parser("verify-report", help="verify generated report artifacts")
    verify.add_argument("--report-dir", type=Path, default=Path("reports"))

    args = parser.parse_args()
    if args.command == "run-demo":
        run_demo(rows=args.rows, seed=args.seed, train_frac=args.train_frac, output=args.output)
    elif args.command == "normalize":
        normalize_snapshots(
            input_path=args.input,
            output_path=args.output,
            tick_size=args.tick_size,
            levels=args.levels,
            derive_state=args.derive_state,
        )
    elif args.command == "fit":
        fit_model(
            input_path=args.input,
            model_path=args.model,
            levels=args.levels,
            ridge=args.ridge,
            probability_scale=args.probability_scale,
        )
    elif args.command == "score":
        columns = args.columns.split(",") if args.columns else None
        score_model(input_path=args.input, model_path=args.model, output_path=args.output, columns=columns)
    elif args.command == "describe-model":
        describe_model(model_path=args.model)
    elif args.command == "verify-report":
        verify_report(report_dir=args.report_dir)


def run_demo(rows: int, seed: int, output: Path, train_frac: float = 0.70) -> None:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")

    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    books = simulate_order_books(SimulationConfig(rows=rows, seed=seed))
    train = books.sample(frac=train_frac, random_state=seed)
    heldout = books.drop(index=train.index)
    model = LCRIModel().fit(train)
    scored = add_regime_transition_features(model.score_frame(books))
    heldout_scored = add_regime_transition_features(model.score_frame(heldout))

    metrics = evaluate_signals(scored)
    heldout_metrics = evaluate_signals(heldout_scored)
    generalization_gap = signal_generalization_gap(metrics, heldout_metrics)
    by_regime = regime_metrics(scored)
    heldout_by_regime = regime_metrics(heldout_scored)
    regime_gap = regime_generalization_gap(by_regime, heldout_by_regime)
    by_transition = transition_conditioned_metrics(scored)
    heldout_by_transition = transition_conditioned_metrics(heldout_scored)
    transition_gap = transition_generalization_gap(by_transition, heldout_by_transition)
    overview = generalization_overview(generalization_gap, regime_gap, transition_gap)
    gap_leaderboard = generalization_gap_leaderboard(generalization_gap, regime_gap, transition_gap)
    lcri_gap_leaderboard = lcri_generalization_gap_leaderboard(
        generalization_gap,
        regime_gap,
        transition_gap,
    )
    lcri_gap_delta = lcri_generalization_gap_delta(generalization_gap, regime_gap, transition_gap)
    lcri_gap_flags = lcri_gap_delta_flags(lcri_gap_delta)
    lcri_gap_summary = lcri_gap_delta_summary(lcri_gap_delta)
    transition_lift = transition_signal_lift(scored)
    heldout_transition_lift = transition_signal_lift(heldout_scored)
    transition_robustness = transition_robustness_summary(scored)
    heldout_transition_robustness = transition_robustness_summary(heldout_scored)

    artifact_paths = [
        "lcri-model.json",
        "sample_snapshots.csv",
        "metrics.csv",
        "heldout_metrics.csv",
        "generalization_gap.csv",
        "regime_metrics.csv",
        "heldout_regime_metrics.csv",
        "regime_generalization_gap.csv",
        "transition_metrics.csv",
        "heldout_transition_metrics.csv",
        "transition_generalization_gap.csv",
        "generalization_overview.json",
        "generalization_gap_leaderboard.csv",
        "lcri_generalization_gap_leaderboard.csv",
        "lcri_generalization_gap_delta.csv",
        "lcri_gap_delta_flags.csv",
        "lcri_gap_delta_summary.json",
        "transition_lift.csv",
        "heldout_transition_lift.csv",
        "transition_robustness.json",
        "heldout_transition_robustness.json",
        "research_summary.md",
        "figures/raw_vs_lcri_scatter.png",
        "figures/regime_signal_quality.png",
        "figures/transition_signal_quality.png",
        "figures/heldout_transition_signal_quality.png",
        "figures/calibration_curve.png",
        "figures/heldout_calibration_curve.png",
        "figures/generalization_gap.png",
        "figures/regime_generalization_gap.png",
        "figures/transition_generalization_gap.png",
        "figures/lcri_generalization_gap_delta.png",
    ]

    model.save(output / "lcri-model.json")
    scored.head(500).to_csv(output / "sample_snapshots.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    heldout_metrics.to_csv(output / "heldout_metrics.csv", index=False)
    generalization_gap.to_csv(output / "generalization_gap.csv", index=False)
    by_regime.to_csv(output / "regime_metrics.csv", index=False)
    heldout_by_regime.to_csv(output / "heldout_regime_metrics.csv", index=False)
    regime_gap.to_csv(output / "regime_generalization_gap.csv", index=False)
    by_transition.to_csv(output / "transition_metrics.csv", index=False)
    heldout_by_transition.to_csv(output / "heldout_transition_metrics.csv", index=False)
    transition_gap.to_csv(output / "transition_generalization_gap.csv", index=False)
    write_json(output / "generalization_overview.json", overview)
    gap_leaderboard.to_csv(output / "generalization_gap_leaderboard.csv", index=False)
    lcri_gap_leaderboard.to_csv(output / "lcri_generalization_gap_leaderboard.csv", index=False)
    lcri_gap_delta.to_csv(output / "lcri_generalization_gap_delta.csv", index=False)
    lcri_gap_flags.to_csv(output / "lcri_gap_delta_flags.csv", index=False)
    write_json(output / "lcri_gap_delta_summary.json", lcri_gap_summary)
    transition_lift.to_csv(output / "transition_lift.csv", index=False)
    heldout_transition_lift.to_csv(output / "heldout_transition_lift.csv", index=False)
    write_json(output / "transition_robustness.json", transition_robustness)
    write_json(output / "heldout_transition_robustness.json", heldout_transition_robustness)
    write_figures(
        scored,
        by_regime,
        output / "figures",
        transition_table=by_transition,
        heldout_transition_table=heldout_by_transition,
        heldout_frame=heldout_scored,
        generalization_gap=generalization_gap,
        regime_generalization_gap=regime_gap,
        transition_generalization_gap=transition_gap,
        lcri_generalization_gap_delta=lcri_gap_delta,
    )

    heldout_rows = len(books) - len(train)
    write_research_summary(
        output / "research_summary.md",
        rows=len(books),
        train_rows=len(train),
        heldout_rows=heldout_rows,
        seed=seed,
        train_frac=train_frac,
        metrics=metrics,
        heldout_metrics=heldout_metrics,
        generalization_gap=generalization_gap,
        regime_generalization_gap=regime_gap,
        transition_generalization_gap=transition_gap,
        generalization_overview=overview,
        generalization_gap_leaderboard=gap_leaderboard,
        lcri_generalization_gap_leaderboard=lcri_gap_leaderboard,
        lcri_generalization_gap_delta=lcri_gap_delta,
        lcri_gap_delta_flags=lcri_gap_flags,
        lcri_gap_delta_summary=lcri_gap_summary,
        transition_lift=transition_lift,
        transition_robustness=transition_robustness,
        heldout_transition_lift=heldout_transition_lift,
        heldout_transition_robustness=heldout_transition_robustness,
    )
    artifact_metadata = collect_artifact_metadata(output, artifact_paths)
    manifest = build_artifact_manifest(
        rows=len(books),
        train_rows=len(train),
        heldout_rows=heldout_rows,
        seed=seed,
        train_frac=train_frac,
        model_artifact_version=ARTIFACT_VERSION,
        artifacts=artifact_paths,
        artifact_metadata=artifact_metadata,
    )
    write_json(output / "artifact_manifest.json", manifest)
    missing = missing_artifacts(output, [*artifact_paths, "artifact_manifest.json"])
    if missing:
        raise RuntimeError(f"missing generated artifacts: {missing}")

    print("Wrote research artifacts")
    print(f"rows: {len(books)} total, {len(train)} train, {heldout_rows} held out")
    print(f"train fraction: {train_frac:.2f}")
    print(f"model: {output / 'lcri-model.json'}")
    print(f"metrics: {output / 'metrics.csv'}")
    print(f"heldout metrics: {output / 'heldout_metrics.csv'}")
    print(f"generalization gap: {output / 'generalization_gap.csv'}")
    print(f"regime metrics: {output / 'regime_metrics.csv'}")
    print(f"heldout regime metrics: {output / 'heldout_regime_metrics.csv'}")
    print(f"regime generalization gap: {output / 'regime_generalization_gap.csv'}")
    print(f"transition metrics: {output / 'transition_metrics.csv'}")
    print(f"heldout transition metrics: {output / 'heldout_transition_metrics.csv'}")
    print(f"transition generalization gap: {output / 'transition_generalization_gap.csv'}")
    print(f"generalization overview: {output / 'generalization_overview.json'}")
    print(f"generalization gap leaderboard: {output / 'generalization_gap_leaderboard.csv'}")
    print(f"lcri generalization gap leaderboard: {output / 'lcri_generalization_gap_leaderboard.csv'}")
    print(f"lcri generalization gap delta: {output / 'lcri_generalization_gap_delta.csv'}")
    print(f"lcri gap delta flags: {output / 'lcri_gap_delta_flags.csv'}")
    print(f"lcri gap delta summary: {output / 'lcri_gap_delta_summary.json'}")
    print(f"transition lift: {output / 'transition_lift.csv'}")
    print(f"heldout transition lift: {output / 'heldout_transition_lift.csv'}")
    print(f"transition robustness: {output / 'transition_robustness.json'}")
    print(f"heldout transition robustness: {output / 'heldout_transition_robustness.json'}")
    print(f"summary: {output / 'research_summary.md'}")
    print(f"manifest: {output / 'artifact_manifest.json'}")
    print(f"figures: {output / 'figures'}")
    print()
    print(metrics.to_string(index=False))


def verify_report(report_dir: Path) -> None:
    manifest_path = report_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    errors = [
        *verify_artifact_manifest(report_dir, manifest),
        *verify_generalization_overview(report_dir),
        *verify_lcri_generalization_gap_delta(report_dir),
        *verify_lcri_gap_delta_flags(report_dir),
        *verify_lcri_gap_delta_summary(report_dir),
    ]
    if errors:
        raise ValueError(f"report verification failed: {errors}")
    print(f"Verified report artifacts: {report_dir}")


def normalize_snapshots(
    input_path: Path,
    output_path: Path,
    tick_size: float,
    levels: int = 5,
    derive_state: bool = False,
) -> None:
    frame = pd.read_csv(input_path)
    normalized = normalize_l2_snapshots(
        frame, tick_size=tick_size, levels=levels, derive_state=derive_state
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    print(f"Wrote normalized snapshots: {output_path}")


def fit_model(
    input_path: Path,
    model_path: Path,
    levels: int,
    ridge: float = 1e-3,
    probability_scale: float = 1.0,
) -> None:
    frame = pd.read_csv(input_path)
    model = LCRIModel(ModelConfig(levels=levels, ridge=ridge, probability_scale=probability_scale)).fit(
        frame
    )
    model.save(model_path)
    print(f"Wrote model: {model_path}")


def describe_model(model_path: Path) -> None:
    model = LCRIModel.load(model_path)
    features = model.baseline.coefficients.size - 1
    regimes = sorted(model.baseline.residual_scale_by_regime or {})
    print(f"schema_version: {model.artifact_version()}")
    print(f"levels: {model.config.levels}")
    print(f"ridge: {model.config.ridge}")
    print(f"probability_scale: {model.config.probability_scale}")
    print(f"features: {features}")
    print(f"regimes: {', '.join(regimes)}")


def score_model(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    columns: list[str] | None = None,
) -> None:
    frame = pd.read_csv(input_path)
    model = LCRIModel.load(model_path)
    scored = model.score_frame(frame)
    if columns is not None:
        columns = [column.strip() for column in columns if column.strip()]
        missing = sorted(set(columns) - set(scored.columns))
        if missing:
            raise ValueError(f"requested score output columns are unavailable: {missing}")
        scored = scored.loc[:, columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Wrote scores: {output_path}")


if __name__ == "__main__":
    main()
