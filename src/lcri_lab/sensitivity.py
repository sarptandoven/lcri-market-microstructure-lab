from __future__ import annotations

import numpy as np
import pandas as pd

from lcri_lab.publishability import PublishabilityConfig, add_publishability_gate


def publishability_latency_sweep(
    frame: pd.DataFrame,
    latency_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
    *,
    base_config: PublishabilityConfig | None = None,
) -> pd.DataFrame:
    """Measure publishability decay as latency penalty increases."""
    if not latency_grid:
        raise ValueError("latency_grid must be non-empty")
    if any((not np.isfinite(value)) or value < 0.0 for value in latency_grid):
        raise ValueError("latency_grid values must be finite and non-negative")

    base_config = base_config or PublishabilityConfig()
    rows = []
    for latency in latency_grid:
        config = PublishabilityConfig(
            min_edge_ticks=base_config.min_edge_ticks,
            probability_threshold=base_config.probability_threshold,
            crowding_penalty_ticks=base_config.crowding_penalty_ticks,
            latency_penalty_ticks=float(latency),
        )
        gated = add_publishability_gate(frame, config=config)
        publishable = gated["is_publishable"].astype(bool)
        rows.append(
            {
                "latency_penalty_ticks": float(latency),
                "publishable_count": int(publishable.sum()),
                "publishable_rate": float(publishable.mean()),
                "mean_publishable_edge_ticks": float(
                    gated.loc[publishable, "publishable_edge_ticks"].mean()
                )
                if publishable.any()
                else 0.0,
            }
        )
    return pd.DataFrame(rows)
