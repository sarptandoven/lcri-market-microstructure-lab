from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PublishabilityConfig:
    """Cost and confidence controls for deciding whether an LCRI signal is publishable."""

    min_edge_ticks: float = 0.25
    probability_threshold: float = 0.55
    crowding_penalty_ticks: float = 0.0
    latency_penalty_ticks: float = 0.0

    def __post_init__(self) -> None:
        for name, value in {
            "min_edge_ticks": self.min_edge_ticks,
            "probability_threshold": self.probability_threshold,
            "crowding_penalty_ticks": self.crowding_penalty_ticks,
            "latency_penalty_ticks": self.latency_penalty_ticks,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.min_edge_ticks < 0.0:
            raise ValueError("min_edge_ticks must be non-negative")
        if not 0.5 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must be in [0.5, 1.0]")
        if self.crowding_penalty_ticks < 0.0:
            raise ValueError("crowding_penalty_ticks must be non-negative")
        if self.latency_penalty_ticks < 0.0:
            raise ValueError("latency_penalty_ticks must be non-negative")


def add_publishability_gate(
    frame: pd.DataFrame,
    *,
    config: PublishabilityConfig | None = None,
) -> pd.DataFrame:
    """Add a publishability decision using cost-aware labels and LCRI probabilities.

    The gate is deliberately conservative. It only publishes long or short signals
    when the model confidence clears a probability threshold and the estimated net
    edge remains positive after crowding and latency penalties.
    """
    config = config or PublishabilityConfig()
    required = {
        "lcri_probability",
        "long_net_return_ticks",
        "short_net_return_ticks",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing publishability columns: {missing}")

    output = frame.copy()
    probability = output["lcri_probability"].astype(float)
    long_edge = output["long_net_return_ticks"].astype(float) - config.crowding_penalty_ticks - config.latency_penalty_ticks
    short_edge = output["short_net_return_ticks"].astype(float) - config.crowding_penalty_ticks - config.latency_penalty_ticks

    long_candidate = (probability >= config.probability_threshold) & (long_edge >= config.min_edge_ticks)
    short_candidate = ((1.0 - probability) >= config.probability_threshold) & (short_edge >= config.min_edge_ticks)

    output["publishable_edge_ticks"] = np.select(
        [long_candidate, short_candidate],
        [long_edge, short_edge],
        default=np.maximum(long_edge, short_edge),
    )
    output["publishable_side"] = np.select(
        [long_candidate, short_candidate],
        ["long", "short"],
        default="abstain",
    )
    output["is_publishable"] = output["publishable_side"] != "abstain"
    return output
