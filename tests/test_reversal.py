import pandas as pd
import pytest

from lcri_lab.reversal import (
    add_queue_reversal_risk,
    add_reversal_lead_lag_coupling,
    reversal_coupling_regime_stress,
    reversal_stress_concentration_summary,
)


def test_queue_reversal_risk_flags_fragile_pressure() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [2.0, 1.0, -1.5],
            "pressure_memory": [-0.5, 1.2, 0.4],
            "transmission_pressure": [0.2, 0.9, -0.2],
            "liquidity_void_ratio": [0.30, 0.05, 0.20],
        }
    )

    output = add_queue_reversal_risk(frame, threshold=0.75)

    assert output["queue_reversal_flag"].tolist() == [True, False, True]
    assert output["queue_reversal_pressure"].iloc[0] < 0.0
    assert output["queue_reversal_pressure"].iloc[2] > 0.0
    assert output["transmission_gap"].ge(0.0).all()


def test_queue_reversal_risk_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing reversal columns"):
        add_queue_reversal_risk(pd.DataFrame({"lcri": [1.0]}))


def test_queue_reversal_risk_rejects_invalid_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [1.0],
            "pressure_memory": [1.0],
            "transmission_pressure": [1.0],
            "liquidity_void_ratio": [0.0],
        }
    )

    for threshold in [-0.1, float("nan")]:
        with pytest.raises(ValueError, match="threshold"):
            add_queue_reversal_risk(frame, threshold=threshold)


def test_reversal_lead_lag_coupling_scores_next_queue_snapback() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["thin", "thin", "thick", "thick"],
            "transmission_pressure": [2.0, -1.0, -2.0, 1.0],
            "queue_reversal_pressure": [0.0, 0.9, 0.0, -0.8],
            "queue_reversal_risk": [0.1, 0.6, 0.2, 0.5],
        }
    )

    output = add_reversal_lead_lag_coupling(frame, group_col="regime")

    assert output["next_queue_reversal_pressure"].tolist() == [0.9, 0.0, -0.8, 0.0]
    assert output["reversal_lead_lag_flag"].tolist() == [True, False, True, False]
    assert output["reversal_lead_lag_coupling"].iloc[0] == pytest.approx(0.18)
    assert output["reversal_lead_lag_coupling"].iloc[2] == pytest.approx(0.8 * 0.5 / 3.0)


def test_reversal_lead_lag_coupling_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing reversal coupling columns"):
        add_reversal_lead_lag_coupling(pd.DataFrame({"transmission_pressure": [1.0]}))


def test_reversal_coupling_regime_stress_ranks_concentrated_regimes() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["thin", "thin", "thick", "thick"],
            "transmission_pressure": [1.0, 1.0, 4.0, 4.0],
            "reversal_lead_lag_coupling": [0.6, 0.4, 0.0, 0.0],
        }
    )

    stress = reversal_coupling_regime_stress(frame)

    assert stress["regime"].tolist() == ["thin", "thick"]
    assert stress.loc[0, "coupled_rows"] == 2
    assert stress.loc[0, "coupling_share"] == pytest.approx(1.0)
    assert stress.loc[0, "transmission_exposure_share"] == pytest.approx(0.2)
    assert stress.loc[0, "stress_concentration_ratio"] == pytest.approx(5.0)


def test_reversal_coupling_regime_stress_rejects_negative_coupling() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["thin"],
            "transmission_pressure": [1.0],
            "reversal_lead_lag_coupling": [-0.1],
        }
    )

    with pytest.raises(ValueError, match="non-negative"):
        reversal_coupling_regime_stress(frame)


def test_reversal_stress_concentration_summary_flags_clustered_stress() -> None:
    stress = pd.DataFrame(
        {
            "regime": ["thin", "thick"],
            "rows": [20, 80],
            "coupled_rows": [7, 1],
            "coupling_share": [0.75, 0.25],
            "transmission_exposure_share": [0.30, 0.70],
            "stress_concentration_ratio": [2.5, 0.36],
        }
    )

    summary = reversal_stress_concentration_summary(stress, concentration_threshold=2.0)

    assert summary["top_regime"] == "thin"
    assert summary["coupled_rows"] == 8
    assert summary["max_stress_concentration_ratio"] == pytest.approx(2.5)
    assert summary["is_concentrated"] is True
    assert summary["gate_decision"] == "review"


def test_reversal_stress_concentration_summary_handles_empty_table() -> None:
    summary = reversal_stress_concentration_summary(pd.DataFrame())

    assert summary["top_regime"] == "none"
    assert summary["gate_decision"] == "pass"
