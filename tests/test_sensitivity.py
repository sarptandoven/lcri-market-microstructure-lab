import pandas as pd
import pytest

from lcri_lab.sensitivity import publishability_latency_sweep


def test_publishability_latency_sweep_is_monotonic() -> None:
    frame = pd.DataFrame(
        {
            "lcri_probability": [0.75, 0.70, 0.30],
            "long_net_return_ticks": [2.0, 0.8, -2.0],
            "short_net_return_ticks": [-2.0, -0.8, 2.0],
        }
    )

    result = publishability_latency_sweep(frame, latency_grid=(0.0, 0.5, 1.0))

    assert result["latency_penalty_ticks"].tolist() == [0.0, 0.5, 1.0]
    assert result["publishable_count"].tolist() == [3, 3, 2]
    assert result["publishable_count"].is_monotonic_decreasing


def test_publishability_latency_sweep_rejects_invalid_grid() -> None:
    frame = pd.DataFrame(
        {
            "lcri_probability": [0.75],
            "long_net_return_ticks": [1.0],
            "short_net_return_ticks": [-1.0],
        }
    )

    with pytest.raises(ValueError, match="latency_grid"):
        publishability_latency_sweep(frame, latency_grid=(-0.1,))
