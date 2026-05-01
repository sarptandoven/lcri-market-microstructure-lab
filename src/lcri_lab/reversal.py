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
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
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
