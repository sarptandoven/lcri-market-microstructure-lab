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


def publishability_margin_diagnostics(
    frame: pd.DataFrame,
    *,
    config: PublishabilityConfig | None = None,
) -> pd.DataFrame:
    """Measure each row's distance from the conservative publishability frontier.

    Rows near zero are threshold-fragile: a small model-confidence move, latency
    estimate, or crowding update can flip the publish/abstain decision. The
    signed margin is positive only when both confidence and net edge clear their
    side-specific gates.
    """
    config = config or PublishabilityConfig()
    required = {"lcri_probability", "long_net_return_ticks", "short_net_return_ticks"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing publishability columns: {missing}")

    probability = frame["lcri_probability"].astype(float)
    penalty = config.crowding_penalty_ticks + config.latency_penalty_ticks
    long_edge = frame["long_net_return_ticks"].astype(float) - penalty
    short_edge = frame["short_net_return_ticks"].astype(float) - penalty
    values = np.column_stack([probability, long_edge, short_edge])
    if not np.isfinite(values).all():
        raise ValueError("publishability inputs must be finite")

    long_confidence_margin = probability - config.probability_threshold
    short_confidence_margin = (1.0 - probability) - config.probability_threshold
    long_edge_margin = long_edge - config.min_edge_ticks
    short_edge_margin = short_edge - config.min_edge_ticks
    long_margin = np.minimum(long_confidence_margin, long_edge_margin)
    short_margin = np.minimum(short_confidence_margin, short_edge_margin)
    signed_margin = np.maximum(long_margin, short_margin)
    preferred_side = np.select(
        [long_margin >= short_margin, short_margin > long_margin],
        ["long", "short"],
        default="long",
    )
    output = pd.DataFrame(
        {
            "preferred_side": preferred_side,
            "long_confidence_margin": long_confidence_margin,
            "short_confidence_margin": short_confidence_margin,
            "long_edge_margin_ticks": long_edge_margin,
            "short_edge_margin_ticks": short_edge_margin,
            "publishability_margin": signed_margin,
            "frontier_distance": np.abs(signed_margin),
            "is_threshold_fragile": np.abs(signed_margin) <= 0.05,
        },
        index=frame.index,
    )
    return output.reset_index(drop=True)


def publishability_margin_summary(
    diagnostics: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize threshold-fragile publishability frontier diagnostics."""
    if diagnostics.empty:
        return {
            "rows": 0,
            "publishable_margin_rows": 0,
            "abstain_margin_rows": 0,
            "threshold_fragile_rows": 0,
            "threshold_fragile_share": 0.0,
            "minimum_frontier_distance": 0.0,
            "closest_frontier_side": "none",
        }
    required = {"preferred_side", "publishability_margin", "frontier_distance", "is_threshold_fragile"}
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"missing publishability diagnostic columns: {missing}")

    margin = diagnostics["publishability_margin"].astype(float)
    distance = diagnostics["frontier_distance"].astype(float)
    fragile = diagnostics["is_threshold_fragile"].astype(bool)
    if not np.isfinite(np.column_stack([margin, distance])).all():
        raise ValueError("publishability diagnostics must be finite")
    closest = diagnostics.loc[distance.idxmin()]
    return {
        "rows": len(diagnostics),
        "publishable_margin_rows": int((margin >= 0.0).sum()),
        "abstain_margin_rows": int((margin < 0.0).sum()),
        "threshold_fragile_rows": int(fragile.sum()),
        "threshold_fragile_share": float(fragile.mean()),
        "minimum_frontier_distance": float(distance.min()),
        "closest_frontier_side": str(closest["preferred_side"]),
    }



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
    long_return = output["long_net_return_ticks"].astype(float)
    short_return = output["short_net_return_ticks"].astype(float)
    if not np.isfinite(np.column_stack([probability, long_return, short_return])).all():
        raise ValueError("publishability inputs must be finite")
    penalty = config.crowding_penalty_ticks + config.latency_penalty_ticks
    long_edge = long_return - penalty
    short_edge = short_return - penalty
    values = np.column_stack([probability, long_edge, short_edge])
    if not np.isfinite(values).all():
        raise ValueError("publishability inputs must be finite")

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
