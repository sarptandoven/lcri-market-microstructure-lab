import pandas as pd
import pytest

from lcri_lab.evaluation import evaluate_cost_aware_signals


def test_evaluate_cost_aware_signals_excludes_abstentions() -> None:
    frame = pd.DataFrame(
        {
            "raw_imbalance": [0.2, -0.3, 0.1, -0.1],
            "lcri": [1.0, -1.0, 0.5, 0.2],
            "cost_aware_direction": [1, 0, -1, -1],
        }
    )

    result = evaluate_cost_aware_signals(frame)

    assert result["rows"].tolist() == [2, 2]
    assert result["abstained_rows"].tolist() == [2, 2]
    assert result["directional_accuracy"].tolist() == [1.0, 1.0]


def test_evaluate_cost_aware_signals_rejects_all_abstain_frame() -> None:
    frame = pd.DataFrame(
        {
            "raw_imbalance": [0.1],
            "lcri": [0.2],
            "cost_aware_direction": [-1],
        }
    )

    with pytest.raises(ValueError, match="no tradable rows"):
        evaluate_cost_aware_signals(frame)
