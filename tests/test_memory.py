import pandas as pd
import pytest

from lcri_lab.memory import (
    add_liquidity_memory_half_life,
    add_pressure_memory,
    pressure_memory_decay_summary,
)


def test_pressure_memory_adds_persistence_features() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [0.5, 1.0, 1.5, -0.5],
            "imbalance_fracture": [0.1, -0.2, 0.3, -0.4],
        }
    )

    output = add_pressure_memory(frame, window=3)

    expected = {
        "pressure_memory",
        "fracture_memory",
        "pressure_memory_z",
        "memory_fracture_alignment",
        "pressure_decay_risk",
        "latent_liquidity_fracture",
    }
    assert expected.issubset(output.columns)
    assert output["pressure_decay_risk"].ge(0).all()
    assert output["pressure_memory"].iloc[0] == pytest.approx(0.5)
    assert output["latent_liquidity_fracture"].iloc[0] == pytest.approx(0.05)


def test_pressure_memory_rejects_missing_inputs() -> None:
    with pytest.raises(ValueError, match="missing pressure memory columns"):
        add_pressure_memory(pd.DataFrame({"lcri": [1.0]}))


def test_pressure_memory_rejects_non_finite_inputs() -> None:
    frame = pd.DataFrame({"lcri": [float("nan")], "imbalance_fracture": [0.1]})

    with pytest.raises(ValueError, match="finite"):
        add_pressure_memory(frame)


def test_pressure_memory_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window"):
        add_pressure_memory(
            pd.DataFrame({"lcri": [1.0], "imbalance_fracture": [0.1]}),
            window=1,
        )
    with pytest.raises(ValueError, match="integer"):
        add_pressure_memory(
            pd.DataFrame({"lcri": [1.0], "imbalance_fracture": [0.1]}),
            window=2.5,
        )


def test_liquidity_memory_half_life_marks_local_decay_events() -> None:
    frame = pd.DataFrame({"pressure_memory": [0.0, 4.0, 3.0, 1.9, -5.0, -2.0]})

    output = add_liquidity_memory_half_life(frame, window=4)

    assert output["pressure_memory_decay_event"].tolist() == [False, False, False, True, False, True]
    assert output["pressure_memory_half_life"].tolist() == [0.0, 0.0, 0.0, 2.0, 0.0, 1.0]
    assert output["pressure_memory_decay_state"].tolist() == [
        "inactive",
        "persistent",
        "persistent",
        "slow_decay",
        "persistent",
        "fast_decay",
    ]
    assert output["pressure_memory_decay_ratio"].iloc[3] == pytest.approx(1.9 / 4.0)
    assert output["pressure_memory_release_velocity"].iloc[3] == pytest.approx((1.0 - 1.9 / 4.0) / 2.0)
    assert output["pressure_memory_release_velocity"].iloc[5] == pytest.approx(0.6)


def test_pressure_memory_decay_summary_counts_state_release_speed() -> None:
    frame = pd.DataFrame({"pressure_memory": [0.0, 4.0, 3.0, 1.9, -5.0, -2.0]})
    output = add_liquidity_memory_half_life(frame, window=4)

    summary = pressure_memory_decay_summary(output).set_index("pressure_memory_decay_state")

    assert summary.loc["fast_decay", "observations"] == 1
    assert summary.loc["fast_decay", "event_rate"] == pytest.approx(1.0)
    assert summary.loc["fast_decay", "mean_release_velocity"] == pytest.approx(0.6)
    assert summary.loc["persistent", "decay_events"] == 0


def test_liquidity_memory_half_life_respects_groups() -> None:
    frame = pd.DataFrame(
        {"venue": ["a", "b", "a", "b"], "pressure_memory": [4.0, 4.0, 1.0, 3.0]}
    )

    output = add_liquidity_memory_half_life(frame, window=3, group_col="venue")

    assert output["pressure_memory_decay_event"].tolist() == [False, False, True, False]


def test_liquidity_memory_half_life_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="missing half-life columns"):
        add_liquidity_memory_half_life(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError, match="decay_fraction"):
        add_liquidity_memory_half_life(
            pd.DataFrame({"pressure_memory": [1.0]}), decay_fraction=1.0
        )
