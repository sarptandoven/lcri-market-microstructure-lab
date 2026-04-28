from pathlib import Path

from lcri_lab.model import LCRIModel
from lcri_lab.simulator import SimulationConfig, simulate_order_books


output_dir = Path("reports/example")
output_dir.mkdir(parents=True, exist_ok=True)

snapshots = simulate_order_books(SimulationConfig(rows=5_000, seed=42))
train = snapshots.iloc[:3_500]
test = snapshots.iloc[3_500:]

model = LCRIModel().fit(train)
scored = model.score_frame(test)

model.save(output_dir / "lcri-model.json")
scored[["timestamp", "regime", "raw_imbalance", "lcri", "lcri_probability"]].to_csv(
    output_dir / "scores.csv",
    index=False,
)

print(scored[["timestamp", "regime", "raw_imbalance", "lcri", "lcri_probability"]].head())
