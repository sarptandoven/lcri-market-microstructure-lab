import pandas as pd
import pytest

from lcri_lab.reversal import (
    add_queue_reversal_risk,
    add_reversal_lead_lag_coupling,
    fracture_reversal_release_gate,
    reversal_coupling_regime_stress,
    reversal_stress_concentration_summary,
    reversal_transition_gate_diagnostics,
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


def test_fracture_reversal_release_gate_blocks_confirmed_failure_mode() -> None:
    reversal_summary = {
        "gate_decision": "review",
        "max_stress_concentration_ratio": 2.6,
        "top_regime": "thin",
    }
    heldout_reversal_summary = {
        "gate_decision": "pass",
        "max_stress_concentration_ratio": 1.1,
        "top_regime": "thick",
    }
    fracture_gate = {
        "decision": "block",
        "passes": False,
        "max_fracture_pressure": 0.18,
        "worst_pressure_quantile": "q8",
    }

    gate = fracture_reversal_release_gate(
        reversal_summary,
        fracture_gate,
        heldout_reversal_summary=heldout_reversal_summary,
    )

    assert gate["decision"] == "block"
    assert gate["passes"] is False
    assert gate["top_reversal_regime"] == "thin"
    assert gate["max_reversal_stress_concentration_ratio"] == pytest.approx(2.6)
    assert "reinforced" in str(gate["reason"])


def test_fracture_reversal_release_gate_reviews_unconfirmed_fracture() -> None:
    reversal_summary = {
        "gate_decision": "pass",
        "max_stress_concentration_ratio": 1.0,
        "top_regime": "none",
    }
    fracture_gate = {
        "decision": "block",
        "max_fracture_pressure": 0.12,
        "worst_pressure_quantile": "q5",
    }

    gate = fracture_reversal_release_gate(reversal_summary, fracture_gate)

    assert gate["decision"] == "review"
    assert "without confirming" in str(gate["reason"])


def test_fracture_reversal_release_gate_rejects_incomplete_inputs() -> None:
    with pytest.raises(ValueError, match="missing reversal stress summary columns"):
        fracture_reversal_release_gate(
            {"gate_decision": "pass"},
            {"decision": "pass", "max_fracture_pressure": 0.0, "worst_pressure_quantile": "none"},
        )


def test_reversal_transition_gate_diagnostics_localizes_combined_gate_stress() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["thick", "thin", "thin", "shock", "shock"],
            "regime_changed": [0, 1, 0, 1, 0],
            "reversal_lead_lag_coupling": [0.0, 0.8, 0.1, 0.2, 0.0],
        }
    )
    release_gate = {"decision": "block", "passes": False}

    diagnostics = reversal_transition_gate_diagnostics(
        frame,
        release_gate,
        stress_share_threshold=0.70,
    )

    assert diagnostics["transition"].tolist() == ["thick->thin", "thin->shock"]
    assert diagnostics.loc[0, "rows"] == 1
    assert diagnostics.loc[0, "coupled_rows"] == 1
    assert diagnostics.loc[0, "transition_stress_share"] == pytest.approx(0.8)
    assert diagnostics.loc[0, "release_gate_decision"] == "block"
    assert diagnostics.loc[0, "transition_gate_decision"] == "review"
    assert diagnostics.loc[1, "transition_gate_decision"] == "pass"


def test_reversal_transition_gate_diagnostics_requires_active_release_gate() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["thick", "thin"],
            "regime_changed": [0, 1],
            "reversal_lead_lag_coupling": [0.0, 1.0],
        }
    )

    diagnostics = reversal_transition_gate_diagnostics(
        frame,
        {"decision": "pass", "passes": True},
        stress_share_threshold=0.50,
    )

    assert diagnostics.loc[0, "transition_stress_share"] == pytest.approx(1.0)
    assert diagnostics.loc[0, "transition_gate_decision"] == "pass"


def test_reversal_transition_gate_diagnostics_rejects_bad_inputs() -> None:
    frame = pd.DataFrame(
        {"regime": ["thin"], "regime_changed": [1], "reversal_lead_lag_coupling": [-0.1]}
    )

    with pytest.raises(ValueError, match="non-negative"):
        reversal_transition_gate_diagnostics(frame, {"decision": "review", "passes": False})

    with pytest.raises(ValueError, match="missing release gate columns"):
        reversal_transition_gate_diagnostics(frame, {"decision": "review"})
