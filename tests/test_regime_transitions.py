import pandas as pd
import pytest

from lcri_lab.features import add_regime_transition_features


def test_regime_transition_features_mark_entries() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["thick", "thick", "thin", "stressed", "stressed", "thick"],
        }
    )

    output = add_regime_transition_features(frame, window=3)

    assert output["regime_changed"].tolist() == [0, 0, 1, 1, 0, 1]
    assert output["thin_entry"].tolist() == [0, 0, 1, 0, 0, 0]
    assert output["stressed_entry"].tolist() == [0, 0, 0, 1, 0, 0]
    assert output["regime_transition_count"].iloc[-1] == pytest.approx(2.0)
    assert output["regime_transition_intensity"].iloc[-1] == pytest.approx(2.0 / 3.0)


def test_regime_transition_features_reject_invalid_window() -> None:
    with pytest.raises(ValueError, match="window"):
        add_regime_transition_features(pd.DataFrame({"regime": ["thick"]}), window=0)


def test_regime_transition_features_reject_missing_regime() -> None:
    with pytest.raises(ValueError, match="missing regime"):
        add_regime_transition_features(pd.DataFrame({"x": [1]}))
