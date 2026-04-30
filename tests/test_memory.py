import pandas as pd
import pytest

from lcri_lab.memory import add_pressure_memory


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
    }
    assert expected.issubset(output.columns)
    assert output["pressure_decay_risk"].ge(0).all()
    assert output["pressure_memory"].iloc[0] == pytest.approx(0.5)


def test_pressure_memory_rejects_missing_inputs() -> None:
    with pytest.raises(ValueError, match="missing pressure memory columns"):
        add_pressure_memory(pd.DataFrame({"lcri": [1.0]}))


def test_pressure_memory_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window"):
        add_pressure_memory(
            pd.DataFrame({"lcri": [1.0], "imbalance_fracture": [0.1]}),
            window=1,
        )
