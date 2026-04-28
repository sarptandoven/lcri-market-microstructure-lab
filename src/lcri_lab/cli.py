from __future__ import annotations

import argparse
from pathlib import Path

from lcri_lab.baseline import LiquidityBaseline, compute_lcri
from lcri_lab.evaluation import evaluate_signals, regime_metrics
from lcri_lab.features import compute_features
from lcri_lab.plotting import write_figures
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def main() -> None:
    parser = argparse.ArgumentParser(prog="lcri-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("run-demo", help="run the synthetic LCRI research demo")
    demo.add_argument("--rows", type=int, default=20_000)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", type=Path, default=Path("reports"))

    args = parser.parse_args()
    if args.command == "run-demo":
        run_demo(rows=args.rows, seed=args.seed, output=args.output)


def run_demo(rows: int, seed: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    books = simulate_order_books(SimulationConfig(rows=rows, seed=seed))
    features = compute_features(books)

    train = features.sample(frac=0.70, random_state=seed)
    baseline = LiquidityBaseline().fit(train)
    scored = compute_lcri(features, baseline)

    metrics = evaluate_signals(scored)
    by_regime = regime_metrics(scored)

    scored.head(500).to_csv(output / "sample_snapshots.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    by_regime.to_csv(output / "regime_metrics.csv", index=False)
    write_figures(scored, by_regime, output / "figures")

    print("Wrote demo artifacts")
    print(f"metrics: {output / 'metrics.csv'}")
    print(f"regime metrics: {output / 'regime_metrics.csv'}")
    print(f"figures: {output / 'figures'}")
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
