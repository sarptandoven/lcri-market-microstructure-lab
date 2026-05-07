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
    output["latent_liquidity_fracture"] = signal_memory.abs() * fracture_memory.abs()
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
    release_velocity = (1.0 - decay_ratio).where(decay_event, 0.0) / half_life.mask(~decay_event, 1.0)
    output["pressure_memory_half_life"] = half_life.astype(float)
    output["pressure_memory_decay_ratio"] = decay_ratio.astype(float)
    output["pressure_memory_release_velocity"] = release_velocity.astype(float)
    output["pressure_memory_decay_event"] = decay_event
    output["pressure_memory_decay_state"] = np.select(
        [inactive_memory, decay_event & slow_decay, decay_event],
        ["inactive", "slow_decay", "fast_decay"],
        default="persistent",
    )
    return output


def pressure_memory_decay_summary(
    frame: pd.DataFrame,
    *,
    state_col: str = "pressure_memory_decay_state",
    half_life_col: str = "pressure_memory_half_life",
    velocity_col: str = "pressure_memory_release_velocity",
    fracture_col: str = "latent_liquidity_fracture",
) -> pd.DataFrame:
    """Summarize decay-state frequency and release speed diagnostics."""
    required = [state_col, half_life_col, velocity_col]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing pressure memory summary columns: {missing}")

    optional = [fracture_col] if fracture_col in frame.columns else []
    data = frame[required + optional].copy()
    data[half_life_col] = data[half_life_col].astype(float)
    data[velocity_col] = data[velocity_col].astype(float)
    if optional:
        data[fracture_col] = data[fracture_col].astype(float)
    finite_cols = [half_life_col, velocity_col, *optional]
    if not np.isfinite(data[finite_cols].to_numpy()).all():
        raise ValueError("pressure memory summary inputs must be finite")

    rows = []
    total = max(1, len(data))
    for state, group in data.groupby(state_col, sort=True):
        decay_events = group[half_life_col].gt(0.0)
        event_half_life = group.loc[decay_events, half_life_col]
        event_velocity = group.loc[decay_events, velocity_col]
        row = {
            "pressure_memory_decay_state": state,
            "observations": int(len(group)),
            "share": float(len(group) / total),
            "decay_events": int(decay_events.sum()),
            "event_rate": float(decay_events.mean()),
            "mean_half_life": _finite_mean_or_zero(event_half_life),
            "mean_release_velocity": _finite_mean_or_zero(event_velocity),
        }
        if optional:
            row["mean_latent_liquidity_fracture"] = _finite_mean_or_zero(group[fracture_col])
        rows.append(row)
    return pd.DataFrame(rows)


def classify_pressure_memory_artifacts(
    frame: pd.DataFrame,
    *,
    state_col: str = "pressure_memory_decay_state",
    velocity_col: str = "pressure_memory_release_velocity",
    fracture_col: str = "latent_liquidity_fracture",
    high_fracture_quantile: float = 0.75,
) -> pd.DataFrame:
    """Classify decay states into pressure-memory artifact families."""
    required = [state_col, velocity_col, fracture_col]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing pressure memory artifact columns: {missing}")
    if not np.isfinite(high_fracture_quantile) or not 0.0 < high_fracture_quantile < 1.0:
        raise ValueError("high_fracture_quantile must be finite and between 0 and 1")

    data = frame[required].copy()
    data[velocity_col] = data[velocity_col].astype(float)
    data[fracture_col] = data[fracture_col].astype(float)
    if not np.isfinite(data[[velocity_col, fracture_col]].to_numpy()).all():
        raise ValueError("pressure memory artifact inputs must be finite")

    fracture_bar = float(data[fracture_col].quantile(high_fracture_quantile))
    velocity_bar = float(data.loc[data[velocity_col].gt(0.0), velocity_col].median())
    velocity_bar = velocity_bar if math.isfinite(velocity_bar) else 0.0
    rows = []
    for state, group in data.groupby(state_col, sort=True):
        mean_velocity = _finite_mean_or_zero(group[velocity_col])
        mean_fracture = _finite_mean_or_zero(group[fracture_col])
        elevated_fracture = mean_fracture >= fracture_bar and mean_fracture > 0.0
        active_release = mean_velocity >= velocity_bar and mean_velocity > 0.0
        if state == "persistent" and elevated_fracture:
            artifact = "latent_fracture_persistence"
        elif state == "fast_decay" and elevated_fracture and active_release:
            artifact = "fractured_fast_release"
        elif state == "slow_decay" and elevated_fracture:
            artifact = "sticky_fracture_decay"
        else:
            artifact = "benign_decay"
        rows.append(
            {
                "pressure_memory_decay_state": state,
                "observations": int(len(group)),
                "mean_release_velocity": mean_velocity,
                "mean_latent_liquidity_fracture": mean_fracture,
                "pressure_memory_artifact": artifact,
                "artifact_severity": float(mean_fracture * (1.0 + mean_velocity)),
            }
        )
    return pd.DataFrame(rows)


def _finite_mean_or_zero(value: pd.Series) -> float:
    mean = float(value.mean()) if len(value) else 0.0
    return mean if math.isfinite(mean) else 0.0


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
