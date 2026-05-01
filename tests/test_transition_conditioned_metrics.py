import pandas as pd
import pytest

from lcri_lab.evaluation import transition_conditioned_metrics


def test_transition_conditioned_metrics_splits_stable_and_transition_rows() -> None:
    frame = pd.DataFrame(
        {
            "raw_imbalance": [0.2, -0.1, 0.7, -0.8, 0.3],
            "lcri": [0.4, -0.2, 1.1, -1.4, 0.5],
            "future_direction": [1, 0, 1, 0, 1],
            "regime_changed": [0, 0, 1, 1, 0],
        }
    )

    output = transition_conditioned_metrics(frame)

    assert set(output["segment"]) == {"stable", "transition"}
    assert set(output["signal"]) == {"raw_imbalance", "lcri"}
    assert output.set_index(["segment", "signal"]).loc[("stable", "lcri"), "rows"] == 3
    assert output.set_index(["segment", "signal"]).loc[("transition", "lcri"), "rows"] == 2


def test_transition_conditioned_metrics_accepts_custom_transition_column() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [0.1, -0.4, 0.8],
            "future_direction": [1, 0, 1],
            "transition_flag": [0, 1, 0],
        }
    )

    output = transition_conditioned_metrics(
        frame,
        signals=["lcri"],
        transition_col="transition_flag",
    )

    assert output["rows"].sum() == 3
    assert output["signal"].tolist() == ["lcri", "lcri"]


def test_transition_conditioned_metrics_rejects_missing_transition_column() -> None:
    frame = pd.DataFrame({"lcri": [0.1], "future_direction": [1]})

    with pytest.raises(ValueError, match="regime_changed"):
        transition_conditioned_metrics(frame, signals=["lcri"])
