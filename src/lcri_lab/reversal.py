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
