from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lcri_lab.schema import l2_price_columns, l2_size_columns


def normalize_l2_snapshots(
    frame: pd.DataFrame,
    *,
    tick_size: float,
    levels: int = 5,
    derive_state: bool = False,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Normalize flat L2 snapshots into the package scoring schema.

    The input is expected to contain bid/ask price and size columns named
    ``bid_px_1``, ``bid_sz_1``, ``ask_px_1``, ``ask_sz_1`` through ``levels``.
    Existing state columns are preserved. Missing top-of-book derived columns
    are filled from prices so vendor-specific feeds can be adapted before fit
    or score steps.
    """
    if not isinstance(levels, int) or isinstance(levels, bool):
        raise ValueError("levels must be an integer")
    if levels < 1:
        raise ValueError("levels must be at least 1")
    if not isinstance(tick_size, (int, float)) or isinstance(tick_size, bool):
        raise ValueError("tick_size must be numeric")
    if not math.isfinite(tick_size) or tick_size <= 0.0:
        raise ValueError("tick_size must be a finite positive value")

    output = frame.copy()
    price_columns = l2_price_columns(levels)
    size_columns = l2_size_columns(levels)
    missing = sorted(set(price_columns + size_columns) - set(output.columns))
    if missing:
        raise ValueError(f"missing L2 snapshot columns: {missing}")

    values = output[price_columns + size_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("L2 price and size columns must be finite")
    if (output[size_columns] < 0.0).to_numpy().any():
        raise ValueError("L2 sizes must be non-negative")

    best_bid = output["bid_px_1"].astype(float)
    best_ask = output["ask_px_1"].astype(float)
    spread = best_ask - best_bid
    if (spread <= 0.0).to_numpy().any():
        raise ValueError("best ask must be greater than best bid")
    _validate_price_ladder(output, levels=levels)

    if "mid" not in output:
        output["mid"] = (best_bid + best_ask) / 2.0
    if "spread" not in output:
        output["spread"] = spread
    if "spread_ticks" not in output:
        output["spread_ticks"] = np.maximum(1.0, np.rint(spread / tick_size))
    if "next_mid" not in output:
        output["next_mid"] = output["mid"].shift(-1).fillna(output["mid"])
    if "regime" not in output:
        output["regime"] = "unclassified"
    if derive_state:
        output = add_l2_state_features(output, levels=levels, volatility_window=volatility_window)
    return output


def _validate_price_ladder(frame: pd.DataFrame, *, levels: int) -> None:
    """Reject L2 snapshots with inverted same-side price levels."""
    for level in range(2, levels + 1):
        previous = level - 1
        if (frame[f"bid_px_{level}"] > frame[f"bid_px_{previous}"]).to_numpy().any():
            raise ValueError("bid prices must be non-increasing across levels")
        if (frame[f"ask_px_{level}"] < frame[f"ask_px_{previous}"]).to_numpy().any():
            raise ValueError("ask prices must be non-decreasing across levels")


def add_l2_state_features(
    frame: pd.DataFrame,
    *,
    levels: int = 5,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Fill state features required by the model from normalized L2 snapshots."""
    if not isinstance(levels, int) or isinstance(levels, bool):
        raise ValueError("levels must be an integer")
    if levels < 1:
        raise ValueError("levels must be at least 1")
    if not isinstance(volatility_window, int) or isinstance(volatility_window, bool):
        raise ValueError("volatility_window must be an integer")
    if volatility_window < 2:
        raise ValueError("volatility_window must be at least 2")

    output = frame.copy()
    size_columns = l2_size_columns(levels)
    required = ["mid", *size_columns]
    missing = sorted(set(required) - set(output.columns))
    if missing:
        raise ValueError(f"missing normalized L2 columns: {missing}")

    values = output[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("normalized L2 columns must be finite")
    if (output[size_columns] < 0.0).to_numpy().any():
        raise ValueError("normalized L2 sizes must be non-negative")

    depth = output[size_columns].sum(axis=1)
    depth_change = depth.diff().abs().fillna(0.0)
    returns = output["mid"].astype(float).pct_change().fillna(0.0)

    if "volatility" not in output:
        output["volatility"] = returns.rolling(volatility_window, min_periods=2).std().fillna(0.0)
    if "replenishment_rate" not in output:
        output["replenishment_rate"] = (1.0 - depth_change / depth.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(1.0)
    return output
