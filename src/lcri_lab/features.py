from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(order_books: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    if levels < 1:
        raise ValueError("levels must be at least 1")

    bid_cols = [f"bid_sz_{level}" for level in range(1, levels + 1)]
    ask_cols = [f"ask_sz_{level}" for level in range(1, levels + 1)]
    missing = sorted(set(_required_columns(levels)) - set(order_books.columns))
    if missing:
        raise ValueError(f"missing order book columns: {missing}")

    frame = order_books.copy()

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


def _required_columns(levels: int) -> list[str]:
    size_columns = [
        f"{side}_sz_{level}"
        for level in range(1, levels + 1)
        for side in ("bid", "ask")
    ]
    return [
        "mid",
        "next_mid",
        "spread",
        "spread_ticks",
        "volatility",
        "replenishment_rate",
        *size_columns,
    ]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _depth_slope(frame: pd.DataFrame, bid_cols: list[str], ask_cols: list[str]) -> pd.Series:
    near = frame[[bid_cols[0], ask_cols[0]]].sum(axis=1)
    far = frame[[bid_cols[-1], ask_cols[-1]]].sum(axis=1)
    return _safe_divide(near - far, near + far).fillna(0.0)
