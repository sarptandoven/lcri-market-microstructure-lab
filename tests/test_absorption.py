import pandas as pd
import pytest

from lcri_lab.absorption import add_shadow_absorption


def test_shadow_absorption_splits_transmitted_and_absorbed_pressure() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [1.0, 2.0, -2.0],
            "pressure_memory": [0.95, 0.10, -0.20],
            "pressure_decay_risk": [0.05, 0.80, 0.70],
            "fracture_memory": [0.10, -0.60, 0.50],
        }
    )

    output = add_shadow_absorption(frame, threshold=0.35)

    assert output["absorption_regime"].tolist() == ["transmitted", "absorbed", "absorbed"]
    assert output["shadow_absorption"].iloc[0] < output["shadow_absorption"].iloc[1]
    assert output["transmission_pressure"].abs().le(output["lcri"].abs()).all()


def test_shadow_absorption_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing absorption columns"):
        add_shadow_absorption(pd.DataFrame({"lcri": [1.0]}))


def test_shadow_absorption_rejects_negative_threshold() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [1.0],
            "pressure_memory": [1.0],
            "pressure_decay_risk": [0.0],
            "fracture_memory": [0.0],
        }
    )

    with pytest.raises(ValueError, match="threshold"):
        add_shadow_absorption(frame, threshold=-0.1)
