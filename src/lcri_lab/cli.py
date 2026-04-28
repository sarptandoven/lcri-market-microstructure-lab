from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lcri_lab.evaluation import evaluate_signals, regime_metrics
from lcri_lab.model import LCRIModel, ModelConfig
from lcri_lab.plotting import write_figures
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def main() -> None:
    parser = argparse.ArgumentParser(prog="lcri-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("run-demo", help="run the synthetic LCRI research workflow")
    demo.add_argument("--rows", type=int, default=20_000)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", type=Path, default=Path("reports"))

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

    args = parser.parse_args()
    if args.command == "run-demo":
        run_demo(rows=args.rows, seed=args.seed, output=args.output)
    elif args.command == "fit":
        fit_model(
            input_path=args.input,
            model_path=args.model,
            levels=args.levels,
            ridge=args.ridge,
            probability_scale=args.probability_scale,
        )
    elif args.command == "score":
        score_model(input_path=args.input, model_path=args.model, output_path=args.output)


def run_demo(rows: int, seed: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    books = simulate_order_books(SimulationConfig(rows=rows, seed=seed))
    train = books.sample(frac=0.70, random_state=seed)
    model = LCRIModel().fit(train)
    scored = model.score_frame(books)

    metrics = evaluate_signals(scored)
    by_regime = regime_metrics(scored)

    model.save(output / "lcri-model.json")
    scored.head(500).to_csv(output / "sample_snapshots.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    by_regime.to_csv(output / "regime_metrics.csv", index=False)
    write_figures(scored, by_regime, output / "figures")

    print("Wrote research artifacts")
    print(f"model: {output / 'lcri-model.json'}")
    print(f"metrics: {output / 'metrics.csv'}")
    print(f"regime metrics: {output / 'regime_metrics.csv'}")
    print(f"figures: {output / 'figures'}")
    print()
    print(metrics.to_string(index=False))


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


def score_model(input_path: Path, model_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(input_path)
    model = LCRIModel.load(model_path)
    scored = model.score_frame(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Wrote scores: {output_path}")


if __name__ == "__main__":
    main()
