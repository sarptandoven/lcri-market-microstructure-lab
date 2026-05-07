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
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValueError("window must be an integer")
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


def add_liquidity_memory_half_life(
    frame: pd.DataFrame,
    *,
    window: int = 20,
    memory_col: str = "pressure_memory",
    decay_fraction: float = 0.50,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Estimate how quickly residual pressure memory loses half its force.

    Local half-life is observed only after absolute pressure memory has fallen
    below ``decay_fraction`` of the strongest memory impulse in the lookback
    window. Short half-lives indicate fragile pressure; long or unobserved
    half-lives indicate persistent liquidity-memory stress.
    """
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValueError("window must be an integer")
    if window < 2:
        raise ValueError("window must be at least 2")
    if not np.isfinite(decay_fraction) or not 0.0 < decay_fraction < 1.0:
        raise ValueError("decay_fraction must be finite and between 0 and 1")
    required = [memory_col] + ([group_col] if group_col else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing half-life columns: {missing}")

    output = frame.copy()
    memory = output[memory_col].astype(float)
    if not np.isfinite(memory).all():
        raise ValueError("half-life memory inputs must be finite")

    if group_col is None:
        half_life, decay_ratio = _memory_half_life_block(memory.to_numpy(), window, decay_fraction)
    else:
        half_life = pd.Series(0.0, index=output.index)
        decay_ratio = pd.Series(0.0, index=output.index)
        for _, index in output.groupby(group_col, sort=False).groups.items():
            group_half_life, group_decay_ratio = _memory_half_life_block(
                memory.loc[index].to_numpy(), window, decay_fraction
            )
            half_life.loc[index] = group_half_life.to_numpy()
            decay_ratio.loc[index] = group_decay_ratio.to_numpy()

    decay_event = half_life.gt(0.0)
    slow_decay = half_life.ge(max(2.0, float(window) / 2.0))
    inactive_memory = half_life.eq(0.0) & decay_ratio.eq(0.0)
    output["pressure_memory_half_life"] = half_life.astype(float)
    output["pressure_memory_decay_ratio"] = decay_ratio.astype(float)
    output["pressure_memory_decay_event"] = decay_event
    output["pressure_memory_decay_state"] = np.select(
        [inactive_memory, decay_event & slow_decay, decay_event],
        ["inactive", "slow_decay", "fast_decay"],
        default="persistent",
    )
    return output


def _safe_zscore(value: pd.Series, scale: pd.Series) -> pd.Series:
    safe_scale = scale.mask(scale <= 0.0)
    zscore = value / safe_scale
    return zscore.replace([math.inf, -math.inf], np.nan).fillna(0.0)


def _memory_half_life_block(
    memory: np.ndarray, window: int, decay_fraction: float
) -> tuple[pd.Series, pd.Series]:
    magnitude = np.abs(memory)
    half_life = np.zeros(len(memory), dtype=float)
    decay_ratio = np.zeros(len(memory), dtype=float)
    for row in range(len(memory)):
        start = max(0, row - window + 1)
        local = magnitude[start : row + 1]
        peak_offset = int(np.argmax(local))
        peak = float(local[peak_offset])
        age = row - (start + peak_offset)
        if peak > 0.0:
            decay_ratio[row] = float(magnitude[row] / peak)
        if peak > 0.0 and age > 0 and magnitude[row] <= decay_fraction * peak:
            half_life[row] = float(age)
    return pd.Series(half_life), pd.Series(decay_ratio)
