import pandas as pd
import pytest

from lcri_lab.evaluation import lcri_tail_diagnostics


def test_lcri_tail_diagnostics_reports_both_sides() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [2.5, 1.2, -1.4, -2.2, 0.2],
            "future_direction": [1, 0, 0, 1, 1],
            "future_return_ticks": [3.0, -1.0, -2.0, 2.0, 0.0],
        }
    )

    result = lcri_tail_diagnostics(frame, thresholds=(1.0, 2.0))

    assert set(result["side"]) == {"positive", "negative"}
    assert result["rows"].sum() == 6
    positive_two = result[(result["threshold"] == 2.0) & (result["side"] == "positive")]
    assert positive_two["hit_rate"].iloc[0] == pytest.approx(1.0)
    negative_two = result[(result["threshold"] == 2.0) & (result["side"] == "negative")]
    assert negative_two["mean_future_return_ticks"].iloc[0] == pytest.approx(2.0)


def test_lcri_tail_diagnostics_handles_empty_tail_buckets() -> None:
    frame = pd.DataFrame({"lcri": [0.1], "future_direction": [1]})

    result = lcri_tail_diagnostics(frame, thresholds=(3.0,))

    assert result["rows"].tolist() == [0, 0]
    assert result["hit_rate"].tolist() == [0.0, 0.0]


def test_lcri_tail_diagnostics_rejects_invalid_thresholds() -> None:
    frame = pd.DataFrame({"lcri": [1.0], "future_direction": [1]})

    with pytest.raises(ValueError, match="positive"):
        lcri_tail_diagnostics(frame, thresholds=(0.0,))
