from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_transaction_cost_labels(
    frame: pd.DataFrame,
    *,
    tick_size: float,
    cost_ticks: float = 1.0,
) -> pd.DataFrame:
    """Add next-mid labels after a symmetric spread/slippage cost in ticks."""
    if not math.isfinite(tick_size) or tick_size <= 0.0:
        raise ValueError("tick_size must be a finite positive value")
    if not math.isfinite(cost_ticks) or cost_ticks < 0.0:
        raise ValueError("cost_ticks must be a finite non-negative value")

    required = ["mid", "next_mid"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing transaction label columns: {missing}")

    output = frame.copy()
    gross_ticks = (output["next_mid"].astype(float) - output["mid"].astype(float)) / tick_size
    long_net = gross_ticks - cost_ticks
    short_net = -gross_ticks - cost_ticks

    output["gross_return_ticks"] = gross_ticks
    output["long_net_return_ticks"] = long_net
    output["short_net_return_ticks"] = short_net
    output["cost_aware_direction"] = np.select(
        [long_net > 0.0, short_net > 0.0],
        [1, 0],
        default=-1,
    )
    return output
