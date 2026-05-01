from __future__ import annotations

import numpy as np
import pandas as pd


def add_shadow_absorption(
    frame: pd.DataFrame,
    *,
    pressure_col: str = "lcri",
    memory_col: str = "pressure_memory",
    decay_col: str = "pressure_decay_risk",
    fracture_col: str = "fracture_memory",
    threshold: float = 0.35,
) -> pd.DataFrame:
    """Estimate whether residual pressure is absorbed or transmitted.

    Strong residual pressure is not always tradable pressure. If the residual is
    large but pressure memory is weak, decay risk is high, or fracture leans the
    other way, the book may be absorbing the visible imbalance instead of letting
    it transmit into price.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    required = [pressure_col, memory_col, decay_col, fracture_col]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing absorption columns: {missing}")

    output = frame.copy()
    pressure = output[pressure_col].astype(float)
    memory = output[memory_col].astype(float)
    decay = output[decay_col].astype(float)
    fracture = output[fracture_col].astype(float)
    values = np.column_stack([pressure, memory, decay, fracture])
    if not np.isfinite(values).all():
        raise ValueError("absorption inputs must be finite")

    memory_alignment = np.sign(pressure) * memory
    fracture_drag = np.maximum(-(np.sign(pressure) * fracture), 0.0)
    shadow_absorption = decay + fracture_drag + np.maximum(0.0, np.abs(pressure) - memory_alignment)
    transmission_pressure = pressure / (1.0 + shadow_absorption)

    output["memory_alignment"] = memory_alignment
    output["shadow_absorption"] = shadow_absorption
    output["transmission_pressure"] = transmission_pressure
    output["absorption_regime"] = np.select(
        [shadow_absorption <= threshold, transmission_pressure.abs() >= pressure.abs() * 0.5],
        ["transmitted", "contested"],
        default="absorbed",
    )
    return output
