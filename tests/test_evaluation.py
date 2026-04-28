import pandas as pd
import pytest

from lcri_lab.evaluation import calibration_curve, evaluate_signals, regime_metrics
from lcri_lab.model import LCRIModel
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def test_calibration_curve_rejects_non_positive_bins() -> None:
    books = simulate_order_books(SimulationConfig(rows=120, seed=21))
    scored = LCRIModel().fit(books.iloc[:80]).score_frame(books.iloc[80:])

    with pytest.raises(ValueError, match="bins"):
        calibration_curve(scored, signal="lcri", bins=0)


def test_evaluation_rejects_empty_frames() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_signals(pd.DataFrame())
    with pytest.raises(ValueError, match="empty"):
        regime_metrics(pd.DataFrame())
