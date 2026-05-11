from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_microstructure_alpha_stack(
    frame: pd.DataFrame,
    *,
    window: int = 20,
    signal_col: str = "pressure_memory",
    fracture_col: str = "latent_liquidity_fracture",
    return_col: str = "gross_return_ticks",
    spread_col: str = "spread_ticks",
    depth_col: str | None = None,
) -> pd.DataFrame:
    """Add path-dependent microstructure alpha diagnostics.

    The stack treats LCRI as a residual pressure process instead of a single
    cross-sectional score. It combines four effects that matter for short-horizon
    order-book alpha research:

    - toxic pressure resonance: same-signed pressure that survives while latent
      liquidity fracture is elevated;
    - resiliency-adjusted alpha: pressure that remains predictive after charging
      spread and depth-fracture penalties;
    - phase-shift alpha: pressure that flips into adverse-selection returns;
    - crowding exhaustion: high pressure whose incremental slope stalls.
    """
    _validate_window(window)
    required = [signal_col, fracture_col, return_col, spread_col]
    if depth_col is not None:
        required.append(depth_col)
    _require_columns(frame, required, label="microstructure alpha")

    output = frame.copy()
    signal = output[signal_col].astype(float)
    fracture = output[fracture_col].astype(float)
    returns = output[return_col].astype(float)
    spread = output[spread_col].astype(float)
    arrays = [signal, fracture, returns, spread]
    if depth_col is None:
        depth_penalty = pd.Series(1.0, index=output.index, dtype=float)
    else:
        depth = output[depth_col].astype(float)
        arrays.append(depth)
        depth_penalty = 1.0 / (1.0 + _safe_zscore(depth.rolling(window, min_periods=2).std().fillna(0.0)))
    _require_finite(arrays, label="microstructure alpha inputs")

    signed_alignment = np.sign(signal).replace(0.0, np.nan) * np.sign(returns).replace(0.0, np.nan)
    adverse_phase = signed_alignment.lt(0.0).fillna(False).astype(float)
    pressure_slope = signal.diff().fillna(0.0)
    pressure_accel = pressure_slope.diff().fillna(0.0)
    resonance = signal.ewm(span=window, adjust=False, min_periods=1).mean() * fracture.abs()
    resonance_persistence = resonance.abs().rolling(window, min_periods=2).mean().fillna(resonance.abs())
    spread_cost = spread.clip(lower=1.0) / spread.clip(lower=1.0).rolling(window, min_periods=1).median()

    output["toxic_pressure_resonance"] = resonance.astype(float)
    output["resonance_persistence"] = resonance_persistence.astype(float)
    output["phase_shift_alpha"] = (adverse_phase * signal.abs() * fracture.abs()).astype(float)
    output["resiliency_adjusted_alpha"] = (
        signal * (1.0 + fracture.abs()) * depth_penalty / (1.0 + spread_cost)
    ).astype(float)
    output["crowding_exhaustion"] = (
        signal.abs().rolling(window, min_periods=2).mean().fillna(0.0)
        * pressure_slope.abs().rolling(window, min_periods=2).mean().fillna(0.0)
        / (1.0 + pressure_accel.abs().rolling(window, min_periods=2).mean().fillna(0.0))
    ).astype(float)
    output["microstructure_alpha_score"] = (
        output["resiliency_adjusted_alpha"].abs()
        + output["toxic_pressure_resonance"].abs()
        + output["phase_shift_alpha"]
        - output["crowding_exhaustion"].clip(lower=0.0)
    ).astype(float)
    return output


def microstructure_alpha_regime_summary(
    frame: pd.DataFrame,
    *,
    regime_col: str = "pressure_memory_decay_state",
    score_col: str = "microstructure_alpha_score",
    resonance_col: str = "toxic_pressure_resonance",
    phase_col: str = "phase_shift_alpha",
    adjusted_col: str = "resiliency_adjusted_alpha",
) -> pd.DataFrame:
    """Summarize alpha concentration by pressure-memory regime."""
    required = [regime_col, score_col, resonance_col, phase_col, adjusted_col]
    _require_columns(frame, required, label="microstructure alpha regime summary")
    data = frame[required].copy()
    for column in [score_col, resonance_col, phase_col, adjusted_col]:
        data[column] = data[column].astype(float)
    _require_finite([data[score_col], data[resonance_col], data[phase_col], data[adjusted_col]], label="alpha summary inputs")

    total_score = float(data[score_col].clip(lower=0.0).sum())
    rows = []
    for regime, group in data.groupby(regime_col, sort=True):
        positive_score = float(group[score_col].clip(lower=0.0).sum())
        rows.append(
            {
                "pressure_memory_decay_state": regime,
                "observations": int(len(group)),
                "alpha_share": _safe_divide(positive_score, total_score),
                "mean_microstructure_alpha_score": _finite_mean(group[score_col]),
                "mean_toxic_pressure_resonance": _finite_mean(group[resonance_col]),
                "mean_phase_shift_alpha": _finite_mean(group[phase_col]),
                "mean_resiliency_adjusted_alpha": _finite_mean(group[adjusted_col]),
            }
        )
    return pd.DataFrame(rows)


def alpha_research_gate(
    summary: pd.DataFrame,
    *,
    min_alpha_share: float = 0.20,
    max_phase_shift_alpha: float = 1.00,
) -> dict[str, float | int | str | bool]:
    """Gate whether the strongest alpha pocket is investable or toxic."""
    if not np.isfinite(min_alpha_share) or not 0.0 <= min_alpha_share <= 1.0:
        raise ValueError("min_alpha_share must be finite and between 0 and 1")
    if not np.isfinite(max_phase_shift_alpha) or max_phase_shift_alpha < 0.0:
        raise ValueError("max_phase_shift_alpha must be finite and non-negative")
    required = [
        "pressure_memory_decay_state",
        "alpha_share",
        "mean_microstructure_alpha_score",
        "mean_phase_shift_alpha",
    ]
    _require_columns(summary, required, label="alpha research gate")
    if summary.empty:
        return {
            "alpha_gate": "block",
            "selected_regime": "none",
            "selected_alpha_share": 0.0,
            "selected_phase_shift_alpha": 0.0,
            "review_rows": 0,
            "investable": False,
        }

    data = summary.copy()
    for column in ["alpha_share", "mean_microstructure_alpha_score", "mean_phase_shift_alpha"]:
        data[column] = data[column].astype(float)
    _require_finite(
        [data["alpha_share"], data["mean_microstructure_alpha_score"], data["mean_phase_shift_alpha"]],
        label="alpha gate inputs",
    )
    if not data["alpha_share"].between(0.0, 1.0).all():
        raise ValueError("alpha_share must be between 0 and 1")

    ranked = data.sort_values(
        ["alpha_share", "mean_microstructure_alpha_score"], ascending=[False, False]
    ).reset_index(drop=True)
    selected = ranked.iloc[0]
    toxic = selected["mean_phase_shift_alpha"] > max_phase_shift_alpha
    concentrated = selected["alpha_share"] >= min_alpha_share
    gate = "pass" if concentrated and not toxic else "review" if concentrated else "block"
    return {
        "alpha_gate": gate,
        "selected_regime": str(selected["pressure_memory_decay_state"]),
        "selected_alpha_share": float(selected["alpha_share"]),
        "selected_phase_shift_alpha": float(selected["mean_phase_shift_alpha"]),
        "review_rows": int((data["mean_phase_shift_alpha"] > max_phase_shift_alpha).sum()),
        "investable": bool(gate == "pass"),
    }


def alpha_toxicity_review_table(
    summary: pd.DataFrame,
    *,
    min_alpha_share: float = 0.20,
    max_phase_shift_alpha: float = 1.00,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """Rank alpha regimes that need toxicity review before release.

    The table separates investable concentration from toxic concentration and
    points reviewers to pressure-memory regimes where alpha looks like adverse
    selection rather than usable signal.
    """
    _validate_alpha_review_thresholds(min_alpha_share, max_phase_shift_alpha, min_score)
    required = [
        "pressure_memory_decay_state",
        "observations",
        "alpha_share",
        "mean_microstructure_alpha_score",
        "mean_phase_shift_alpha",
        "mean_toxic_pressure_resonance",
    ]
    _require_columns(summary, required, label="alpha toxicity review")
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "pressure_memory_decay_state",
                "observations",
                "alpha_share",
                "mean_microstructure_alpha_score",
                "mean_phase_shift_alpha",
                "mean_toxic_pressure_resonance",
                "toxicity_score",
                "review_label",
            ]
        )

    data = summary[required].copy()
    numeric_columns = [
        "observations",
        "alpha_share",
        "mean_microstructure_alpha_score",
        "mean_phase_shift_alpha",
        "mean_toxic_pressure_resonance",
    ]
    for column in numeric_columns:
        data[column] = data[column].astype(float)
    _require_finite([data[column] for column in numeric_columns], label="alpha toxicity inputs")
    if not data["alpha_share"].between(0.0, 1.0).all():
        raise ValueError("alpha_share must be between 0 and 1")
    if (data["observations"] < 0.0).any():
        raise ValueError("observations must be non-negative")

    concentration_excess = (data["alpha_share"] - min_alpha_share).clip(lower=0.0)
    phase_excess = (data["mean_phase_shift_alpha"] - max_phase_shift_alpha).clip(lower=0.0)
    score_excess = (data["mean_microstructure_alpha_score"] - min_score).clip(lower=0.0)
    data["toxicity_score"] = (
        concentration_excess
        * (1.0 + phase_excess)
        * (1.0 + data["mean_toxic_pressure_resonance"].abs())
        * (1.0 + score_excess)
    ).astype(float)
    data["review_label"] = np.select(
        [
            (concentration_excess > 0.0) & (phase_excess > 0.0),
            concentration_excess > 0.0,
            phase_excess > 0.0,
        ],
        ["toxic_concentration", "concentrated_alpha", "phase_shift_watch"],
        default="clear",
    )
    data["observations"] = data["observations"].astype(int)
    return data.sort_values(
        ["toxicity_score", "alpha_share", "mean_phase_shift_alpha"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def alpha_toxicity_review_summary(review_table: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize the highest-priority alpha toxicity review row."""
    if review_table.empty:
        return {
            "rows": 0,
            "review_rows": 0,
            "top_regime": "none",
            "top_review_label": "clear",
            "top_toxicity_score": 0.0,
        }
    _require_columns(
        review_table,
        ["pressure_memory_decay_state", "review_label", "toxicity_score"],
        label="alpha toxicity review summary",
    )
    scores = review_table["toxicity_score"].astype(float)
    _require_finite([scores], label="alpha toxicity review summary inputs")
    top = review_table.loc[scores.idxmax()]
    labels = review_table["review_label"].astype(str)
    return {
        "rows": int(len(review_table)),
        "review_rows": int((labels != "clear").sum()),
        "top_regime": str(top["pressure_memory_decay_state"]),
        "top_review_label": str(top["review_label"]),
        "top_toxicity_score": float(top["toxicity_score"]),
    }


def _validate_window(window: int) -> None:
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValueError("window must be an integer")
    if window < 2:
        raise ValueError("window must be at least 2")


def _validate_alpha_review_thresholds(
    min_alpha_share: float,
    max_phase_shift_alpha: float,
    min_score: float,
) -> None:
    if not np.isfinite(min_alpha_share) or not 0.0 <= min_alpha_share <= 1.0:
        raise ValueError("min_alpha_share must be finite and between 0 and 1")
    if not np.isfinite(max_phase_shift_alpha) or max_phase_shift_alpha < 0.0:
        raise ValueError("max_phase_shift_alpha must be finite and non-negative")
    if not np.isfinite(min_score) or min_score < 0.0:
        raise ValueError("min_score must be finite and non-negative")


def _require_columns(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _require_finite(series: list[pd.Series], *, label: str) -> None:
    if not np.isfinite(np.column_stack([item.to_numpy(dtype=float) for item in series])).all():
        raise ValueError(f"{label} must be finite")


def _safe_zscore(value: pd.Series) -> pd.Series:
    scale = value.rolling(max(2, min(20, len(value))), min_periods=2).std().replace(0.0, np.nan)
    zscore = (value - value.rolling(max(2, min(20, len(value))), min_periods=1).mean()) / scale
    return zscore.replace([math.inf, -math.inf], np.nan).fillna(0.0).abs()


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _finite_mean(value: pd.Series) -> float:
    mean = float(value.mean()) if len(value) else 0.0
    return mean if math.isfinite(mean) else 0.0
