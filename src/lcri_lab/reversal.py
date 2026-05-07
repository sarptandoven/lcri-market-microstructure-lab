from __future__ import annotations

import numpy as np
import pandas as pd


def add_queue_reversal_risk(
    frame: pd.DataFrame,
    *,
    pressure_col: str = "lcri",
    memory_col: str = "pressure_memory",
    transmission_col: str = "transmission_pressure",
    void_col: str = "liquidity_void_ratio",
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Estimate when visible pressure is vulnerable to a queue reversal.

    The research idea is that pressure becomes fragile when current residual
    imbalance disagrees with memory, transmission is weaker than displayed
    pressure, and the book has voids behind the touch. That combination can mark
    a crowded signal that is likely to snap back rather than continue.
    """
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    required = [pressure_col, memory_col, transmission_col, void_col]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing reversal columns: {missing}")

    output = frame.copy()
    pressure = output[pressure_col].astype(float)
    memory = output[memory_col].astype(float)
    transmission = output[transmission_col].astype(float)
    void = output[void_col].astype(float)
    values = np.column_stack([pressure, memory, transmission, void])
    if not np.isfinite(values).all():
        raise ValueError("reversal inputs must be finite")

    memory_disagreement = np.maximum(-(np.sign(pressure) * memory), 0.0)
    transmission_gap = np.maximum(np.abs(pressure) - np.abs(transmission), 0.0)
    normalized_gap = transmission_gap / (1.0 + np.abs(pressure))
    reversal_risk = memory_disagreement + normalized_gap + void.clip(lower=0.0)
    reversal_pressure = -np.sign(pressure) * reversal_risk

    output["memory_disagreement"] = memory_disagreement
    output["transmission_gap"] = transmission_gap
    output["queue_reversal_risk"] = reversal_risk
    output["queue_reversal_pressure"] = reversal_pressure
    output["queue_reversal_flag"] = reversal_risk >= threshold
    return output


def add_reversal_lead_lag_coupling(
    frame: pd.DataFrame,
    *,
    transmission_col: str = "transmission_pressure",
    reversal_col: str = "queue_reversal_pressure",
    risk_col: str = "queue_reversal_risk",
    group_col: str | None = None,
) -> pd.DataFrame:
    """Measure whether transmitted pressure leads queue-reversal pressure.

    Latent-liquidity fracture is most actionable when pressure that survives
    shadow absorption is followed by a same-direction queue snapback. This
    feature shifts the reversal field one row ahead, optionally inside a regime
    or instrument group, and scores same-direction transmission-to-reversal
    coupling. Positive values mark pressure that is being echoed by future queue
    reversal stress; negative values mark pressure that is likely being absorbed.
    """
    required = [transmission_col, reversal_col, risk_col]
    if group_col is not None:
        required.append(group_col)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing reversal coupling columns: {missing}")

    output = frame.copy()
    transmission = output[transmission_col].astype(float)
    reversal = output[reversal_col].astype(float)
    risk = output[risk_col].astype(float)
    if not np.isfinite(np.column_stack([transmission, reversal, risk])).all():
        raise ValueError("reversal coupling inputs must be finite")

    if group_col is None:
        next_reversal = reversal.shift(-1)
        next_risk = risk.shift(-1)
    else:
        groups = output[group_col]
        next_reversal = reversal.groupby(groups, sort=False).shift(-1)
        next_risk = risk.groupby(groups, sort=False).shift(-1)

    aligned = np.sign(transmission) * next_reversal.fillna(0.0)
    coupling = np.maximum(aligned, 0.0) * next_risk.fillna(0.0) / (1.0 + transmission.abs())
    output["next_queue_reversal_pressure"] = next_reversal.fillna(0.0)
    output["next_queue_reversal_risk"] = next_risk.fillna(0.0)
    output["reversal_lead_lag_coupling"] = coupling
    output["reversal_lead_lag_flag"] = coupling > 0.0
    return output


def reversal_coupling_regime_stress(
    frame: pd.DataFrame,
    *,
    regime_col: str = "regime",
    coupling_col: str = "reversal_lead_lag_coupling",
    transmission_col: str = "transmission_pressure",
) -> pd.DataFrame:
    """Summarize where lead-lag reversal stress concentrates by regime.

    A regime whose share of coupling stress is larger than its share of
    transmitted pressure is a microstructure failure mode.
    """
    required = [regime_col, coupling_col, transmission_col]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing reversal stress columns: {missing}")
    if frame.empty:
        return pd.DataFrame(columns=_REGIME_STRESS_COLUMNS)

    regime = frame[regime_col].astype(str)
    coupling = frame[coupling_col].astype(float)
    exposure = frame[transmission_col].astype(float).abs()
    if not np.isfinite(np.column_stack([coupling, exposure])).all():
        raise ValueError("reversal stress inputs must be finite")
    if (coupling < 0.0).any():
        raise ValueError("reversal coupling must be non-negative")

    total_coupling = coupling.sum()
    total_exposure = exposure.sum()
    rows = []
    for value, index in regime.groupby(regime, sort=True).groups.items():
        regime_coupling = coupling.loc[index]
        regime_exposure = exposure.loc[index]
        coupling_share = regime_coupling.sum() / total_coupling if total_coupling else 0.0
        exposure_share = regime_exposure.sum() / total_exposure if total_exposure else 0.0
        rows.append(
            {
                "regime": value,
                "rows": len(index),
                "coupled_rows": int((regime_coupling > 0.0).sum()),
                "coupling_share": float(coupling_share),
                "transmission_exposure_share": float(exposure_share),
                "stress_concentration_ratio": float(coupling_share / exposure_share) if exposure_share else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=_REGIME_STRESS_COLUMNS).sort_values(
        "stress_concentration_ratio", ascending=False
    ).reset_index(drop=True)


def reversal_stress_concentration_summary(
    stress: pd.DataFrame,
    *,
    concentration_threshold: float = 1.5,
) -> dict[str, float | int | str | bool]:
    """Summarize whether reversal stress is dangerously regime-concentrated.

    This turns the regime stress table into a compact report gate. The key
    research signal is not just that queue reversals exist, but that their
    lead-lag coupling clusters in a narrow regime more than transmitted pressure
    exposure alone explains.
    """
    if not np.isfinite(concentration_threshold) or concentration_threshold < 0.0:
        raise ValueError("concentration_threshold must be finite and non-negative")
    if stress.empty:
        return {
            "regimes": 0,
            "coupled_rows": 0,
            "top_regime": "none",
            "max_stress_concentration_ratio": 0.0,
            "top_coupling_share": 0.0,
            "top_transmission_exposure_share": 0.0,
            "concentration_threshold": float(concentration_threshold),
            "is_concentrated": False,
            "gate_decision": "pass",
        }
    missing = sorted(set(_REGIME_STRESS_COLUMNS) - set(stress.columns))
    if missing:
        raise ValueError(f"missing reversal stress summary columns: {missing}")

    numeric = stress[
        ["coupled_rows", "coupling_share", "transmission_exposure_share", "stress_concentration_ratio"]
    ].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("reversal stress summary inputs must be finite")
    top = stress.sort_values("stress_concentration_ratio", ascending=False).iloc[0]
    max_ratio = float(top["stress_concentration_ratio"])
    is_concentrated = max_ratio >= concentration_threshold and float(top["coupling_share"]) > 0.0
    return {
        "regimes": int(len(stress)),
        "coupled_rows": int(stress["coupled_rows"].astype(int).sum()),
        "top_regime": str(top["regime"]),
        "max_stress_concentration_ratio": max_ratio,
        "top_coupling_share": float(top["coupling_share"]),
        "top_transmission_exposure_share": float(top["transmission_exposure_share"]),
        "concentration_threshold": float(concentration_threshold),
        "is_concentrated": bool(is_concentrated),
        "gate_decision": "review" if is_concentrated else "pass",
    }


def fracture_reversal_release_gate(
    reversal_summary: dict[str, float | int | str | bool],
    fracture_gate: dict[str, float | int | str | bool],
    *,
    heldout_reversal_summary: dict[str, float | int | str | bool] | None = None,
) -> dict[str, float | int | str | bool]:
    """Combine calibration fracture pressure with queue-reversal stress.

    Calibration fracture pressure says the model's probability shape is broken.
    Reversal stress says that broken shape is concentrated in a microstructure
    regime where transmitted pressure is followed by queue snapback. Either one
    deserves review; both together are stronger evidence that the LCRI artifact
    is describing a latent-liquidity failure mode rather than harmless noise.
    """
    _require_reversal_summary(reversal_summary, label="reversal stress summary")
    if heldout_reversal_summary:
        _require_reversal_summary(heldout_reversal_summary, label="heldout reversal stress summary")
    missing_fracture = sorted({"decision", "max_fracture_pressure", "worst_pressure_quantile"} - set(fracture_gate))
    if missing_fracture:
        raise ValueError(f"missing fracture gate columns: {missing_fracture}")

    heldout_reversal_summary = heldout_reversal_summary or {}
    fracture_blocks = str(fracture_gate["decision"]) == "block" or fracture_gate.get("passes") is False
    reversal_reviews = str(reversal_summary["gate_decision"]) == "review"
    heldout_reversal_reviews = str(heldout_reversal_summary.get("gate_decision", "pass")) == "review"
    any_reversal_review = reversal_reviews or heldout_reversal_reviews
    decision = "pass"
    if fracture_blocks and any_reversal_review:
        decision = "block"
    elif fracture_blocks or any_reversal_review:
        decision = "review"

    max_reversal_ratio = max(
        float(reversal_summary["max_stress_concentration_ratio"]),
        float(heldout_reversal_summary.get("max_stress_concentration_ratio", 0.0)),
    )
    return {
        "decision": decision,
        "passes": decision == "pass",
        "fracture_decision": str(fracture_gate["decision"]),
        "reversal_gate_decision": str(reversal_summary["gate_decision"]),
        "heldout_reversal_gate_decision": str(heldout_reversal_summary.get("gate_decision", "pass")),
        "max_fracture_pressure": float(fracture_gate["max_fracture_pressure"]),
        "worst_pressure_quantile": str(fracture_gate["worst_pressure_quantile"]),
        "max_reversal_stress_concentration_ratio": max_reversal_ratio,
        "top_reversal_regime": str(reversal_summary["top_regime"]),
        "reason": _fracture_reversal_gate_reason(fracture_blocks, any_reversal_review),
    }


def _require_reversal_summary(summary: dict[str, float | int | str | bool], *, label: str) -> None:
    missing = sorted({"gate_decision", "max_stress_concentration_ratio", "top_regime"} - set(summary))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _fracture_reversal_gate_reason(fracture_blocks: bool, reversal_reviews: bool) -> str:
    if fracture_blocks and reversal_reviews:
        return "calibration fracture is reinforced by concentrated queue-reversal stress"
    if fracture_blocks:
        return "calibration fracture requires review without confirming reversal concentration"
    if reversal_reviews:
        return "queue-reversal concentration requires review without calibration fracture"
    return "no calibration fracture or concentrated queue-reversal stress detected"


_REGIME_STRESS_COLUMNS = [
    "regime",
    "rows",
    "coupled_rows",
    "coupling_share",
    "transmission_exposure_share",
    "stress_concentration_ratio",
]
