import numpy as np
import pandas as pd
import pytest

from lcri_lab.evaluation import feature_stability_report


def test_feature_stability_report_splits_by_regime() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["thin", "thin", "thick", "thick"],
            "lcri": [1.0, np.nan, 0.5, 0.7],
            "liquidity_void_ratio": [0.2, 0.3, 0.1, 0.2],
        }
    )

    report = feature_stability_report(frame, ["lcri", "liquidity_void_ratio"])

    assert set(report["regime"]) == {"thin", "thick"}
    assert set(report["feature"]) == {"lcri", "liquidity_void_ratio"}
    thin_lcri = report[(report["regime"] == "thin") & (report["feature"] == "lcri")]
    assert thin_lcri["finite_rate"].iloc[0] == pytest.approx(0.5)


def test_feature_stability_report_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"regime": ["thin"], "lcri": [1.0]})

    with pytest.raises(ValueError, match="missing"):
        feature_stability_report(frame, ["missing_feature"])
