from __future__ import annotations

import numpy as np
import pandas as pd

from lcri_lab.schema import snapshot_required_columns


def compute_features(order_books: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    if levels < 1:
        raise ValueError("levels must be at least 1")

    bid_cols = [f"bid_sz_{level}" for level in range(1, levels + 1)]
    ask_cols = [f"ask_sz_{level}" for level in range(1, levels + 1)]
    required = snapshot_required_columns(levels)
    missing = sorted(set(required) - set(order_books.columns))
    if missing:
        raise ValueError(f"missing order book columns: {missing}")

    frame = order_books.copy()
    _validate_numeric_inputs(frame, required, bid_cols + ask_cols)

    bid_depth = frame[bid_cols].sum(axis=1)
    ask_depth = frame[ask_cols].sum(axis=1)
    total_depth = bid_depth + ask_depth

    frame["bid_depth"] = bid_depth
    frame["ask_depth"] = ask_depth
    frame["total_depth"] = total_depth
    frame["top_bid_depth"] = frame["bid_sz_1"]
    frame["top_ask_depth"] = frame["ask_sz_1"]
    frame["raw_imbalance"] = _safe_divide(bid_depth - ask_depth, total_depth)
    frame["top_imbalance"] = _safe_divide(
        frame["top_bid_depth"] - frame["top_ask_depth"],
        frame["top_bid_depth"] + frame["top_ask_depth"],
    )
    frame["log_total_depth"] = np.log1p(total_depth)
    frame["depth_slope"] = _depth_slope(frame, bid_cols, ask_cols)
    frame["queue_pressure"] = _safe_divide(
        frame["top_bid_depth"] - frame["top_ask_depth"],
        frame["total_depth"],
    )
    frame["spread_depth_ratio"] = _safe_divide(frame["spread"], np.log1p(frame["total_depth"]))
    frame["liquidity_score"] = (
        frame["log_total_depth"] * frame["replenishment_rate"] / (1.0 + frame["spread_ticks"])
    )
    frame["future_signed_move"] = np.sign(frame["next_mid"] - frame["mid"]).clip(lower=0)
    return frame


def tag_liquidity_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign coarse liquidity regimes from spread, depth, volatility, and replenishment."""
    required = ["spread_ticks", "volatility", "replenishment_rate", "liquidity_score"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing regime tagging columns: {missing}")

    output = frame.copy()
    score = output["liquidity_score"].astype(float)
    low_score = score.quantile(0.30)
    high_score = score.quantile(0.70)
    high_vol = output["volatility"].astype(float).quantile(0.75)

    regimes = np.select(
        [
            (output["spread_ticks"] >= 4) | (output["volatility"] >= high_vol),
            (score <= low_score) & (output["replenishment_rate"] < 0.50),
            (score >= high_score) & (output["replenishment_rate"] >= 0.75),
        ],
        ["stressed", "thin", "replenishing"],
        default="thick",
    )
    output["regime"] = regimes.astype(str)
    return output


def feature_columns() -> list[str]:
    return [
        "spread_ticks",
        "volatility",
        "replenishment_rate",
        "log_total_depth",
        "depth_slope",
        "spread_depth_ratio",
        "liquidity_score",
    ]


def _validate_numeric_inputs(
    frame: pd.DataFrame, required: list[str], size_columns: list[str]
) -> None:
    values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("order book numeric columns must be finite")
    if (frame[size_columns] < 0.0).to_numpy().any():
        raise ValueError("order book sizes must be non-negative")
    if (frame["spread"] <= 0.0).to_numpy().any():
        raise ValueError("spread must be positive")
    if (frame["spread_ticks"] < 1.0).to_numpy().any():
        raise ValueError("spread_ticks must be at least 1")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _depth_slope(frame: pd.DataFrame, bid_cols: list[str], ask_cols: list[str]) -> pd.Series:
    near = frame[[bid_cols[0], ask_cols[0]]].sum(axis=1)
    far = frame[[bid_cols[-1], ask_cols[-1]]].sum(axis=1)
    return _safe_divide(near - far, near + far).fillna(0.0)
