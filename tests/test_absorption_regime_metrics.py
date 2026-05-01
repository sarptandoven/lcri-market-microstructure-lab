import pandas as pd
import pytest

from lcri_lab.evaluation import absorption_regime_metrics


def test_absorption_regime_metrics_compares_pressure_series() -> None:
    frame = pd.DataFrame(
        {
            "absorption_regime": ["absorbed", "absorbed", "transmitted", "transmitted"],
            "lcri": [1.0, -1.0, 1.0, -1.0],
            "transmission_pressure": [-0.2, -0.8, 0.9, -0.7],
            "future_direction": [0, 0, 1, 0],
        }
    )

    result = absorption_regime_metrics(frame)

    assert set(result["absorption_regime"]) == {"absorbed", "transmitted"}
    assert set(result["signal"]) == {"lcri", "transmission_pressure"}
    transmitted = result[
        (result["absorption_regime"] == "transmitted")
        & (result["signal"] == "transmission_pressure")
    ]
    assert transmitted["directional_accuracy"].iloc[0] == pytest.approx(1.0)


def test_absorption_regime_metrics_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="empty"):
        absorption_regime_metrics(pd.DataFrame())


def test_absorption_regime_metrics_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="absorption_regime"):
        absorption_regime_metrics(pd.DataFrame({"lcri": [1.0]}))
