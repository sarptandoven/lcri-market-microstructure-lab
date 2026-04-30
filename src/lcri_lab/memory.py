from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_pressure_memory(
    frame: pd.DataFrame,
    *,
    window: int = 20,
    signal_col: str = "lcri",
    fracture_col: str = "imbalance_fracture",
) -> pd.DataFrame:
    """Add rolling pressure memory features for LCRI research.

    A one-shot imbalance residual is less informative when it immediately mean
    reverts. Persistent residual pressure with persistent book fracture is a
    different state: the book keeps showing abnormal pressure while the ladder
    remains internally inconsistent.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    required = [signal_col, fracture_col]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing pressure memory columns: {missing}")

    output = frame.copy()
    signal = output[signal_col].astype(float)
    fracture = output[fracture_col].astype(float)
    if not np.isfinite(signal).all() or not np.isfinite(fracture).all():
        raise ValueError("pressure memory inputs must be finite")

    signal_memory = signal.ewm(span=window, adjust=False, min_periods=1).mean()
    fracture_memory = fracture.ewm(span=window, adjust=False, min_periods=1).mean()
    signal_std = signal.rolling(window=window, min_periods=2).std().fillna(0.0)

    output["pressure_memory"] = signal_memory
    output["fracture_memory"] = fracture_memory
    output["pressure_memory_z"] = _safe_zscore(signal_memory, signal_std)
    output["memory_fracture_alignment"] = np.sign(signal_memory) * fracture_memory.abs()
    output["pressure_decay_risk"] = (signal - signal_memory).abs() / (1.0 + signal.abs())
    return output


def _safe_zscore(value: pd.Series, scale: pd.Series) -> pd.Series:
    safe_scale = scale.mask(scale <= 0.0)
    zscore = value / safe_scale
    return zscore.replace([math.inf, -math.inf], np.nan).fillna(0.0)
