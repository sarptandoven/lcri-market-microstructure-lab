from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FillProbabilityConfig:
    """Controls for snapshot-based passive fill and adverse-selection proxies."""

    queue_position_fraction: float = 0.50
    min_probability: float = 0.01
    max_probability: float = 0.99
    pressure_scale: float = 1.0
    queue_penalty_scale: float = 1.0
    adverse_selection_scale: float = 1.0

    def __post_init__(self) -> None:
        for name, value in {
            "queue_position_fraction": self.queue_position_fraction,
            "min_probability": self.min_probability,
            "max_probability": self.max_probability,
            "pressure_scale": self.pressure_scale,
            "queue_penalty_scale": self.queue_penalty_scale,
            "adverse_selection_scale": self.adverse_selection_scale,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.queue_position_fraction <= 1.0:
            raise ValueError("queue_position_fraction must be in [0.0, 1.0]")
        if not 0.0 <= self.min_probability < 1.0:
            raise ValueError("min_probability must be in [0.0, 1.0)")
        if not 0.0 < self.max_probability <= 1.0:
            raise ValueError("max_probability must be in (0.0, 1.0]")
        if self.min_probability >= self.max_probability:
            raise ValueError("min_probability must be less than max_probability")
        if self.pressure_scale <= 0.0:
            raise ValueError("pressure_scale must be positive")
        if self.queue_penalty_scale < 0.0:
            raise ValueError("queue_penalty_scale must be non-negative")
        if self.adverse_selection_scale < 0.0:
            raise ValueError("adverse_selection_scale must be non-negative")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _finite_values(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    values = frame[columns].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{label} inputs must be finite")
    return values


def _normalize_group_columns(
    frame: pd.DataFrame,
    group_cols: str | list[str] | tuple[str, ...] | None,
    label: str,
) -> list[str]:
    if group_cols is None:
        return []
    if isinstance(group_cols, str):
        grouping_columns = [group_cols]
    else:
        grouping_columns = list(group_cols)
    if not grouping_columns:
        raise ValueError("group_cols must be a non-empty sequence when provided")
    _require_columns(frame, set(grouping_columns), label)
    return grouping_columns


def _logistic(value: pd.Series) -> pd.Series:
    clipped = value.clip(-40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _clip_probability(value: pd.Series, config: FillProbabilityConfig) -> pd.Series:
    return value.clip(config.min_probability, config.max_probability)


def add_queue_position_features(
    frame: pd.DataFrame,
    *,
    levels: int = 5,
    queue_position_fraction: float = 0.50,
) -> pd.DataFrame:
    """Approximate queue-ahead state from visible top-of-book depth.

    The repo currently works with L2 snapshots rather than event-level order-add,
    cancel, and trade messages. This function creates a transparent passive-order
    proxy: a child order is assumed to join at a fixed fraction through the best
    bid/ask queue, then normalizes that visible queue-ahead by same-side depth.
    """
    if levels < 1:
        raise ValueError("levels must be positive")
    if not math.isfinite(queue_position_fraction):
        raise ValueError("queue_position_fraction must be finite")
    if not 0.0 <= queue_position_fraction <= 1.0:
        raise ValueError("queue_position_fraction must be in [0.0, 1.0]")

    required = {"bid_sz_1", "ask_sz_1"}
    required.update({f"bid_sz_{level}" for level in range(1, levels + 1)})
    required.update({f"ask_sz_{level}" for level in range(1, levels + 1)})
    _require_columns(frame, required, "queue position")
    values = _finite_values(frame, sorted(required), "queue position")
    if (values < 0.0).any().any():
        raise ValueError("queue position sizes must be non-negative")

    output = frame.copy()
    bid_total = pd.Series(0.0, index=frame.index)
    ask_total = pd.Series(0.0, index=frame.index)
    for level in range(1, levels + 1):
        bid_total = bid_total + values[f"bid_sz_{level}"]
        ask_total = ask_total + values[f"ask_sz_{level}"]
    bid_ahead = values["bid_sz_1"] * queue_position_fraction
    ask_ahead = values["ask_sz_1"] * queue_position_fraction

    output["bid_queue_ahead"] = bid_ahead
    output["ask_queue_ahead"] = ask_ahead
    output["bid_queue_share"] = np.divide(
        bid_ahead,
        bid_total.replace(0.0, np.nan),
    ).fillna(0.0)
    output["ask_queue_share"] = np.divide(
        ask_ahead,
        ask_total.replace(0.0, np.nan),
    ).fillna(0.0)
    output["queue_position_imbalance"] = bid_ahead - ask_ahead
    return output


def add_queue_position_realized_fill_proxy(
    frame: pd.DataFrame,
    *,
    horizon: int = 1,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
) -> pd.DataFrame:
    """Infer passive fill labels from visible queue depletion over a snapshot horizon.

    Event-level add/cancel/trade data is the gold standard for queue priority, but
    the lab's core artifacts often operate on L2 snapshots. This proxy marks a
    passive bid as filled when any of the next ``horizon`` snapshots either loses
    the current best bid price level or depletes at least the estimated bid queue
    ahead; passive asks are handled symmetrically when the best ask level is lost
    upward or same-price visible size depletion clears the estimated ask queue
    ahead. ``group_cols`` prevents fill labels from leaking across independent
    symbols, venues, or sessions when a batch contains interleaved books.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise ValueError("horizon must be an integer")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    required = {
        "bid_px_1",
        "ask_px_1",
        "bid_sz_1",
        "ask_sz_1",
        "bid_queue_ahead",
        "ask_queue_ahead",
    }
    _require_columns(frame, required, "queue position realized fill proxy")
    if group_cols is None:
        grouping_columns: list[str] = []
    elif isinstance(group_cols, str):
        grouping_columns = [group_cols]
    else:
        grouping_columns = list(group_cols)
    if group_cols is not None and not grouping_columns:
        raise ValueError("group_cols must be a non-empty sequence when provided")
    if grouping_columns:
        _require_columns(frame, set(grouping_columns), "queue position realized fill proxy group")
    if frame.empty:
        output = frame.copy()
        output["bid_visible_depletion"] = pd.Series(dtype=float)
        output["ask_visible_depletion"] = pd.Series(dtype=float)
        output["bid_queue_depletion_ratio"] = pd.Series(dtype=float)
        output["ask_queue_depletion_ratio"] = pd.Series(dtype=float)
        output[bid_realized_col] = pd.Series(dtype=float)
        output[ask_realized_col] = pd.Series(dtype=float)
        return output

    values = _finite_values(frame, sorted(required), "queue position realized fill proxy")
    size_columns = ["bid_sz_1", "ask_sz_1", "bid_queue_ahead", "ask_queue_ahead"]
    if (values[size_columns] < 0.0).any().any():
        raise ValueError("queue position realized fill proxy sizes must be non-negative")

    def future_value(column: str, offset: int) -> pd.Series:
        if not grouping_columns:
            return values[column].shift(-offset)
        keys = [frame[group_col] for group_col in grouping_columns]
        return values[column].groupby(keys, sort=False, dropna=False).shift(-offset)

    bid_depletion = pd.Series(0.0, index=frame.index)
    ask_depletion = pd.Series(0.0, index=frame.index)
    has_future = pd.Series(False, index=frame.index)
    for offset in range(1, horizon + 1):
        future_bid_px = future_value("bid_px_1", offset)
        future_ask_px = future_value("ask_px_1", offset)
        future_bid_sz = future_value("bid_sz_1", offset)
        future_ask_sz = future_value("ask_sz_1", offset)
        valid_future = (
            future_bid_px.notna()
            & future_ask_px.notna()
            & future_bid_sz.notna()
            & future_ask_sz.notna()
        )
        has_future = has_future | valid_future

        bid_level_lost = valid_future & (future_bid_px < values["bid_px_1"])
        ask_level_lost = valid_future & (future_ask_px > values["ask_px_1"])
        bid_same_level = valid_future & (future_bid_px == values["bid_px_1"])
        ask_same_level = valid_future & (future_ask_px == values["ask_px_1"])

        bid_same_depletion = (values["bid_sz_1"] - future_bid_sz).clip(lower=0.0).fillna(0.0)
        ask_same_depletion = (values["ask_sz_1"] - future_ask_sz).clip(lower=0.0).fillna(0.0)
        bid_step_depletion = pd.Series(
            np.where(
                bid_level_lost,
                values["bid_sz_1"],
                np.where(bid_same_level, bid_same_depletion, 0.0),
            ),
            index=frame.index,
        )
        ask_step_depletion = pd.Series(
            np.where(
                ask_level_lost,
                values["ask_sz_1"],
                np.where(ask_same_level, ask_same_depletion, 0.0),
            ),
            index=frame.index,
        )
        bid_depletion = bid_depletion.combine(bid_step_depletion, max)
        ask_depletion = ask_depletion.combine(ask_step_depletion, max)

    bid_queue = values["bid_queue_ahead"]
    ask_queue = values["ask_queue_ahead"]
    bid_ratio = pd.Series(
        np.where(
            bid_queue > 0.0,
            bid_depletion / bid_queue.replace(0.0, np.nan),
            np.where(bid_depletion > 0.0, 1.0, 0.0),
        ),
        index=frame.index,
    )
    ask_ratio = pd.Series(
        np.where(
            ask_queue > 0.0,
            ask_depletion / ask_queue.replace(0.0, np.nan),
            np.where(ask_depletion > 0.0, 1.0, 0.0),
        ),
        index=frame.index,
    )

    output = frame.copy()
    output["bid_visible_depletion"] = bid_depletion
    output["ask_visible_depletion"] = ask_depletion
    output["bid_queue_depletion_ratio"] = bid_ratio
    output["ask_queue_depletion_ratio"] = ask_ratio
    bid_realized = has_future & np.where(bid_queue > 0.0, bid_depletion >= bid_queue, bid_depletion > 0.0)
    ask_realized = has_future & np.where(ask_queue > 0.0, ask_depletion >= ask_queue, ask_depletion > 0.0)
    output[bid_realized_col] = bid_realized.astype(float)
    output[ask_realized_col] = ask_realized.astype(float)
    return output


def add_event_level_realized_fill_proxy(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon: float,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
    event_side_col: str = "side",
    event_price_col: str = "price",
    event_size_col: str = "size",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    trade_event_types: tuple[str, ...] = ("trade",),
    cancel_event_types: tuple[str, ...] = ("cancel", "delete"),
    bid_trade_sides: tuple[str, ...] = ("sell", "bid"),
    ask_trade_sides: tuple[str, ...] = ("buy", "ask"),
    bid_cancel_sides: tuple[str, ...] = ("bid",),
    ask_cancel_sides: tuple[str, ...] = ("ask",),
) -> pd.DataFrame:
    """Label passive fills from event-level queue-depleting trades and cancels.

    This is the live-data counterpart to the snapshot depletion proxy. For each
    snapshot, the function measures queue depletion at the displayed best bid/ask
    over ``(snapshot_time, snapshot_time + horizon]``. By default, sell trades and
    bid-side cancels consume bid queue-ahead while buy trades and ask-side cancels
    consume ask queue-ahead. Venue-specific event-type and side aliases can be
    supplied for feeds that encode aggressor side or book side differently.
    Optional grouping columns keep symbols, venues, or sessions from sharing event
    flow in batched data.
    """
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be a finite positive value")
    if group_cols is None:
        grouping_columns: list[str] = []
    elif isinstance(group_cols, str):
        grouping_columns = [group_cols]
    else:
        grouping_columns = list(group_cols)
    if group_cols is not None and not grouping_columns:
        raise ValueError("group_cols must be a non-empty sequence when provided")

    snapshot_required = {
        timestamp_col,
        "bid_px_1",
        "ask_px_1",
        "bid_queue_ahead",
        "ask_queue_ahead",
        *grouping_columns,
    }
    event_required = {
        timestamp_col,
        event_type_col,
        event_side_col,
        event_price_col,
        event_size_col,
        *grouping_columns,
    }
    _require_columns(snapshots, snapshot_required, "event-level realized fill snapshot")
    _require_columns(events, event_required, "event-level realized fill event")

    output = snapshots.copy()
    output["bid_event_depletion"] = pd.Series(0.0, index=snapshots.index)
    output["ask_event_depletion"] = pd.Series(0.0, index=snapshots.index)
    output["bid_event_depletion_ratio"] = pd.Series(0.0, index=snapshots.index)
    output["ask_event_depletion_ratio"] = pd.Series(0.0, index=snapshots.index)
    output[bid_realized_col] = pd.Series(0.0, index=snapshots.index)
    output[ask_realized_col] = pd.Series(0.0, index=snapshots.index)
    if snapshots.empty:
        return output

    snapshot_numeric = _finite_values(
        snapshots,
        [timestamp_col, "bid_px_1", "ask_px_1", "bid_queue_ahead", "ask_queue_ahead"],
        "event-level realized fill snapshot",
    )
    event_numeric = _finite_values(
        events,
        [timestamp_col, event_price_col, event_size_col],
        "event-level realized fill event",
    )
    if (snapshot_numeric[["bid_queue_ahead", "ask_queue_ahead"]] < 0.0).any().any():
        raise ValueError("event-level realized fill snapshot queues must be non-negative")
    if (event_numeric[event_size_col] < 0.0).any():
        raise ValueError("event-level realized fill event sizes must be non-negative")

    def normalized_aliases(values: tuple[str, ...], label: str) -> set[str]:
        if not values:
            raise ValueError(f"{label} must contain at least one alias")
        aliases = {str(value).lower() for value in values}
        if any(alias == "" for alias in aliases):
            raise ValueError(f"{label} aliases must be non-empty")
        return aliases

    trade_type_aliases = normalized_aliases(trade_event_types, "trade_event_types")
    cancel_type_aliases = normalized_aliases(cancel_event_types, "cancel_event_types")
    bid_trade_side_aliases = normalized_aliases(bid_trade_sides, "bid_trade_sides")
    ask_trade_side_aliases = normalized_aliases(ask_trade_sides, "ask_trade_sides")
    bid_cancel_side_aliases = normalized_aliases(bid_cancel_sides, "bid_cancel_sides")
    ask_cancel_side_aliases = normalized_aliases(ask_cancel_sides, "ask_cancel_sides")

    event_types = events[event_type_col].astype(str).str.lower()
    event_sides = events[event_side_col].astype(str).str.lower()
    bid_depleting_events = (event_types.isin(trade_type_aliases) & event_sides.isin(bid_trade_side_aliases)) | (
        event_types.isin(cancel_type_aliases) & event_sides.isin(bid_cancel_side_aliases)
    )
    ask_depleting_events = (event_types.isin(trade_type_aliases) & event_sides.isin(ask_trade_side_aliases)) | (
        event_types.isin(cancel_type_aliases) & event_sides.isin(ask_cancel_side_aliases)
    )

    bid_depletions: list[float] = []
    ask_depletions: list[float] = []
    for _, row in snapshots.iterrows():
        start = float(row[timestamp_col])
        event_window = (event_numeric[timestamp_col] > start) & (
            event_numeric[timestamp_col] <= start + horizon
        )
        if grouping_columns:
            for group_col in grouping_columns:
                event_window = event_window & (events[group_col] == row[group_col])

        bid_at_price = event_numeric[event_price_col] == float(row["bid_px_1"])
        ask_at_price = event_numeric[event_price_col] == float(row["ask_px_1"])
        bid_size = event_numeric.loc[event_window & bid_at_price & bid_depleting_events, event_size_col].sum()
        ask_size = event_numeric.loc[event_window & ask_at_price & ask_depleting_events, event_size_col].sum()
        bid_depletions.append(float(bid_size))
        ask_depletions.append(float(ask_size))

    bid_depletion = pd.Series(bid_depletions, index=snapshots.index)
    ask_depletion = pd.Series(ask_depletions, index=snapshots.index)
    bid_queue = snapshot_numeric["bid_queue_ahead"]
    ask_queue = snapshot_numeric["ask_queue_ahead"]
    bid_ratio = pd.Series(
        np.where(
            bid_queue > 0.0,
            bid_depletion / bid_queue.replace(0.0, np.nan),
            np.where(bid_depletion > 0.0, 1.0, 0.0),
        ),
        index=snapshots.index,
    ).fillna(0.0)
    ask_ratio = pd.Series(
        np.where(
            ask_queue > 0.0,
            ask_depletion / ask_queue.replace(0.0, np.nan),
            np.where(ask_depletion > 0.0, 1.0, 0.0),
        ),
        index=snapshots.index,
    ).fillna(0.0)

    output["bid_event_depletion"] = bid_depletion
    output["ask_event_depletion"] = ask_depletion
    output["bid_event_depletion_ratio"] = bid_ratio
    output["ask_event_depletion_ratio"] = ask_ratio
    output[bid_realized_col] = np.where(bid_queue > 0.0, bid_depletion >= bid_queue, bid_depletion > 0.0).astype(float)
    output[ask_realized_col] = np.where(ask_queue > 0.0, ask_depletion >= ask_queue, ask_depletion > 0.0).astype(float)
    return output


def add_passive_fill_probabilities(
    frame: pd.DataFrame,
    *,
    config: FillProbabilityConfig | None = None,
    pressure_col: str = "lcri",
) -> pd.DataFrame:
    """Estimate passive bid/ask fill probability and adverse-fill risk.

    Negative residual pressure implies sell pressure that can deplete the bid and
    fill passive buys. Positive residual pressure analogously depletes the ask and
    fills passive sells. The queue-share penalty makes the proxy execution-aware:
    deeper position in the visible queue lowers fill odds even when pressure is high.
    """
    config = config or FillProbabilityConfig()
    required = {
        pressure_col,
        "bid_queue_share",
        "ask_queue_share",
        "spread_ticks",
        "volatility",
        "replenishment_rate",
    }
    _require_columns(frame, required, "fill probability")
    values = _finite_values(frame, sorted(required), "fill probability")

    pressure = values[pressure_col]
    bid_queue_share = values["bid_queue_share"].clip(lower=0.0)
    ask_queue_share = values["ask_queue_share"].clip(lower=0.0)
    spread_ticks = values["spread_ticks"].clip(lower=0.0)
    volatility = values["volatility"].clip(lower=0.0)
    replenishment = values["replenishment_rate"].clip(lower=0.0)

    spread_stress = np.log1p(spread_ticks)
    volatility_stress = np.log1p(volatility)
    thin_book_stress = 1.0 / (1.0 + replenishment)

    bid_depletion_pressure = (-pressure / config.pressure_scale) + 0.25 * spread_stress
    ask_depletion_pressure = (pressure / config.pressure_scale) + 0.25 * spread_stress

    bid_fill_logit = (
        bid_depletion_pressure
        + 0.35 * volatility_stress
        + 0.35 * thin_book_stress
        - config.queue_penalty_scale * bid_queue_share
    )
    ask_fill_logit = (
        ask_depletion_pressure
        + 0.35 * volatility_stress
        + 0.35 * thin_book_stress
        - config.queue_penalty_scale * ask_queue_share
    )

    bid_fill = _clip_probability(_logistic(bid_fill_logit), config)
    ask_fill = _clip_probability(_logistic(ask_fill_logit), config)

    bid_adverse_logit = (
        bid_depletion_pressure * config.adverse_selection_scale
        + 0.50 * volatility_stress
        + 0.20 * spread_stress
        - 0.40 * replenishment
    )
    ask_adverse_logit = (
        ask_depletion_pressure * config.adverse_selection_scale
        + 0.50 * volatility_stress
        + 0.20 * spread_stress
        - 0.40 * replenishment
    )

    output = frame.copy()
    output["bid_depletion_pressure"] = bid_depletion_pressure
    output["ask_depletion_pressure"] = ask_depletion_pressure
    output["bid_fill_probability"] = bid_fill
    output["ask_fill_probability"] = ask_fill
    output["bid_adverse_fill_probability"] = _clip_probability(_logistic(bid_adverse_logit), config)
    output["ask_adverse_fill_probability"] = _clip_probability(_logistic(ask_adverse_logit), config)
    output["fill_probability_imbalance"] = ask_fill - bid_fill
    output["passive_fill_regime"] = np.select(
        [bid_fill > ask_fill, ask_fill > bid_fill],
        ["bid_depletion", "ask_depletion"],
        default="balanced",
    )
    return output


def add_execution_adjusted_edge(
    frame: pd.DataFrame,
    *,
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
    long_net_col: str = "long_net_return_ticks",
    short_net_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Convert directional LCRI edge into passive-fill-adjusted tradable edge."""
    required = {
        signal_col,
        probability_col,
        long_net_col,
        short_net_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    _require_columns(frame, required, "execution edge")
    values = _finite_values(frame, sorted(required), "execution edge")

    signal = values[signal_col]
    long_return = values[long_net_col]
    short_return = values[short_net_col]
    bid_fill = values["bid_fill_probability"].clip(0.0, 1.0)
    ask_fill = values["ask_fill_probability"].clip(0.0, 1.0)
    bid_adverse = values["bid_adverse_fill_probability"].clip(0.0, 1.0)
    ask_adverse = values["ask_adverse_fill_probability"].clip(0.0, 1.0)

    long_edge = bid_fill * long_return - bid_adverse * long_return.abs()
    short_edge = ask_fill * short_return - ask_adverse * short_return.abs()
    best_edge = np.maximum(long_edge, short_edge)
    best_side = np.select(
        [(long_edge > 0.0) & (long_edge >= short_edge), (short_edge > 0.0) & (short_edge > long_edge)],
        ["long", "short"],
        default="abstain",
    )

    output = frame.copy()
    output["long_fill_adjusted_edge_ticks"] = long_edge
    output["short_fill_adjusted_edge_ticks"] = short_edge
    output["best_execution_side"] = best_side
    output["execution_adjusted_edge_ticks"] = best_edge
    output["execution_adjusted_lcri_score"] = np.select(
        [best_side == "long", best_side == "short"],
        [signal, signal],
        default=0.0,
    )
    return output


def queue_position_toxicity_surface(
    frame: pd.DataFrame,
    *,
    queue_bins: int = 5,
    regime_col: str | None = "regime",
    side_col: str = "best_execution_side",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    long_return_col: str = "long_net_return_ticks",
    short_return_col: str = "short_net_return_ticks",
    toxic_adverse_to_fill_ratio: float = 0.75,
    toxic_loss_rate: float = 0.50,
    toxic_edge_ticks: float = 0.0,
) -> pd.DataFrame:
    """Surface passive queue fills by depth and adverse-selection toxicity.

    Fill probability alone can reward the wrong passive quotes: deeper queue cells
    may fill primarily when the market is moving against them. This diagnostic is
    side-aware, selects the bid/ask queue and fill columns for the chosen execution
    side, and reports adverse-to-fill ratios, realized loss rates, and execution
    edge per regime/queue bin so queue capacity claims can be screened for toxic
    fills rather than raw fill volume.
    """
    if not isinstance(queue_bins, int) or isinstance(queue_bins, bool):
        raise ValueError("queue_bins must be an integer")
    if queue_bins < 1:
        raise ValueError("queue_bins must be at least 1")
    for name, value in {
        "toxic_adverse_to_fill_ratio": toxic_adverse_to_fill_ratio,
        "toxic_loss_rate": toxic_loss_rate,
        "toxic_edge_ticks": toxic_edge_ticks,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if toxic_adverse_to_fill_ratio < 0.0:
        raise ValueError("toxic_adverse_to_fill_ratio must be non-negative")
    if not 0.0 <= toxic_loss_rate <= 1.0:
        raise ValueError("toxic_loss_rate must be in [0.0, 1.0]")

    required = {
        side_col,
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        bid_realized_col,
        ask_realized_col,
        long_return_col,
        short_return_col,
        "execution_adjusted_edge_ticks",
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position toxicity surface")
    if frame.empty:
        return _empty_queue_position_toxicity_surface()

    numeric_columns = [
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        bid_realized_col,
        ask_realized_col,
        long_return_col,
        short_return_col,
        "execution_adjusted_edge_ticks",
    ]
    values = _finite_values(frame, numeric_columns, "queue position toxicity surface")
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_queue_position_toxicity_surface()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["regime"] = (
        frame.loc[tradable, regime_col].astype(str) if regime_col is not None else "all"
    )
    selected["best_execution_side"] = selected_side
    selected["queue_share"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_queue_share"],
        values.loc[tradable, "ask_queue_share"],
    )
    selected["predicted_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["adverse_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_adverse_fill_probability"],
        values.loc[tradable, "ask_adverse_fill_probability"],
    )
    selected["realized_fill"] = np.where(
        selected_side == "long",
        values.loc[tradable, bid_realized_col],
        values.loc[tradable, ask_realized_col],
    )
    selected["realized_edge_ticks"] = np.where(
        selected_side == "long",
        values.loc[tradable, long_return_col],
        values.loc[tradable, short_return_col],
    )
    selected["execution_adjusted_edge_ticks"] = values.loc[
        tradable, "execution_adjusted_edge_ticks"
    ]

    if not selected["queue_share"].ge(0.0).all():
        raise ValueError("queue position toxicity surface queue shares must be non-negative")
    for column in ["predicted_fill_probability", "adverse_fill_probability", "realized_fill"]:
        if not selected[column].between(0.0, 1.0).all():
            raise ValueError("queue position toxicity surface probabilities must be in [0, 1]")

    rows: list[dict[str, float | int | str]] = []
    for (regime, execution_side), side_group in selected.groupby(
        ["regime", "best_execution_side"], sort=True
    ):
        side_group = side_group.copy()
        side_group["queue_bin"] = _rank_probability_bins(side_group["queue_share"], queue_bins)
        for queue_bin, group in side_group.groupby("queue_bin", sort=True):
            fill_probability = float(group["predicted_fill_probability"].mean())
            adverse_probability = float(group["adverse_fill_probability"].mean())
            adverse_to_fill = (
                adverse_probability / fill_probability if fill_probability > 0.0 else 0.0
            )
            realized_loss_rate = float((group["realized_edge_ticks"] < 0.0).mean())
            mean_edge = float(group["execution_adjusted_edge_ticks"].mean())
            toxic = (
                adverse_to_fill >= toxic_adverse_to_fill_ratio
                or realized_loss_rate >= toxic_loss_rate
                or mean_edge < toxic_edge_ticks
            )
            rows.append(
                {
                    "regime": str(regime),
                    "best_execution_side": str(execution_side),
                    "queue_bin": int(queue_bin),
                    "rows": int(len(group)),
                    "mean_queue_share": float(group["queue_share"].mean()),
                    "mean_predicted_fill_probability": fill_probability,
                    "mean_adverse_fill_probability": adverse_probability,
                    "adverse_to_fill_ratio": float(adverse_to_fill),
                    "realized_fill_rate": float(group["realized_fill"].mean()),
                    "realized_loss_rate": realized_loss_rate,
                    "mean_realized_edge_ticks": float(group["realized_edge_ticks"].mean()),
                    "mean_execution_adjusted_edge_ticks": mean_edge,
                    "queue_toxicity_label": (
                        "toxic_queue_fill" if toxic else "benign_queue_fill"
                    ),
                }
            )
    if not rows:
        return _empty_queue_position_toxicity_surface()
    output = pd.DataFrame(rows)[list(_empty_queue_position_toxicity_surface().columns)]
    if regime_col is not None:
        output = output.rename(columns={"regime": regime_col})
    return output


def queue_position_fraction_sweep(
    frame: pd.DataFrame,
    *,
    fractions: list[float] | tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
    levels: int = 5,
    fill_config: FillProbabilityConfig | None = None,
    pressure_col: str = "lcri",
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
    long_net_col: str = "long_net_return_ticks",
    short_net_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Stress-test passive execution quality across assumed queue placement.

    Snapshot data cannot observe true order priority, so passive fill quality should
    not be reported for a single hard-coded queue assumption. This sweep recomputes
    queue-ahead features, passive fill odds, and execution-adjusted edge at several
    join-depth fractions, producing a placement-sensitivity artifact for research
    review and demos.
    """
    if isinstance(fractions, (str, bytes)):
        raise ValueError("fractions must be a non-empty sequence of finite values")
    fractions = list(fractions)
    if not fractions:
        raise ValueError("fractions must be a non-empty sequence")
    for fraction in fractions:
        if not math.isfinite(float(fraction)):
            raise ValueError("queue_position_fraction values must be finite")
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("queue_position_fraction values must be in [0.0, 1.0]")
    if frame.empty:
        return _empty_queue_position_fraction_sweep()

    rows: list[dict[str, float | int | str]] = []
    for fraction in fractions:
        queued = add_queue_position_features(
            frame,
            levels=levels,
            queue_position_fraction=float(fraction),
        )
        filled = add_passive_fill_probabilities(
            queued,
            config=fill_config,
            pressure_col=pressure_col,
        )
        executed = add_execution_adjusted_edge(
            filled,
            signal_col=signal_col,
            probability_col=probability_col,
            long_net_col=long_net_col,
            short_net_col=short_net_col,
        )
        summary = execution_adjusted_edge_summary(executed)
        rows_count = int(summary["rows"])
        abstain_rows = int(summary["abstain_rows"])
        rows.append(
            {
                "queue_position_fraction": float(fraction),
                "rows": rows_count,
                "mean_bid_queue_share": float(executed["bid_queue_share"].mean()),
                "mean_ask_queue_share": float(executed["ask_queue_share"].mean()),
                "mean_bid_fill_probability": float(summary["mean_bid_fill_probability"]),
                "mean_ask_fill_probability": float(summary["mean_ask_fill_probability"]),
                "mean_fill_probability_imbalance": float(
                    executed["fill_probability_imbalance"].mean()
                ),
                "mean_bid_adverse_fill_probability": float(
                    executed["bid_adverse_fill_probability"].mean()
                ),
                "mean_ask_adverse_fill_probability": float(
                    executed["ask_adverse_fill_probability"].mean()
                ),
                "mean_execution_adjusted_edge_ticks": float(
                    summary["mean_execution_adjusted_edge_ticks"]
                ),
                "tradable_share": float(summary["tradable_share"]),
                "abstain_share": float(abstain_rows / rows_count) if rows_count else 0.0,
                "dominant_execution_side": str(summary["dominant_execution_side"]),
            }
        )
    return pd.DataFrame(rows)[list(_empty_queue_position_fraction_sweep().columns)]


def queue_position_regime_fraction_sweep(
    frame: pd.DataFrame,
    *,
    regime_col: str = "regime",
    fractions: list[float] | tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
    levels: int = 5,
    fill_config: FillProbabilityConfig | None = None,
    pressure_col: str = "lcri",
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
    long_net_col: str = "long_net_return_ticks",
    short_net_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Run quote-placement passive capacity sweeps independently by regime.

    This is the state-aware companion to ``queue_position_fraction_sweep``. It
    keeps each liquidity/event regime separate before capacity-frontier reduction,
    preventing benign high-liquidity states from masking that passive queue edge is
    front-of-queue-only or absent in thin/stress regimes.
    """
    if not isinstance(regime_col, str) or not regime_col:
        raise ValueError("regime_col must be a non-empty string")
    columns = [regime_col, *list(_empty_queue_position_fraction_sweep().columns)]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    required = {
        regime_col,
        pressure_col,
        signal_col,
        probability_col,
        long_net_col,
        short_net_col,
        "spread_ticks",
        "volatility",
        "replenishment_rate",
    }
    required.update({f"bid_sz_{level}" for level in range(1, levels + 1)})
    required.update({f"ask_sz_{level}" for level in range(1, levels + 1)})
    _require_columns(frame, required, "queue position regime fraction sweep")

    rows: list[pd.DataFrame] = []
    for regime, regime_frame in frame.groupby(regime_col, sort=True, dropna=False):
        sweep = queue_position_fraction_sweep(
            regime_frame,
            fractions=fractions,
            levels=levels,
            fill_config=fill_config,
            pressure_col=pressure_col,
            signal_col=signal_col,
            probability_col=probability_col,
            long_net_col=long_net_col,
            short_net_col=short_net_col,
        )
        sweep.insert(0, regime_col, str(regime))
        rows.append(sweep)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def queue_position_capacity_frontier(
    sweep: pd.DataFrame,
    *,
    min_edge_ticks: float = 0.0,
    min_tradable_share: float = 0.50,
) -> dict[str, float | int | str]:
    """Find the deepest queue placement that preserves executable LCRI edge.

    ``queue_position_fraction_sweep`` shows how passive edge decays as assumed
    queue priority worsens. This reducer turns that curve into an actionable
    capacity frontier: the deepest quote-placement fraction that still clears a
    minimum execution-adjusted edge and tradable-share gate. It is intentionally
    threshold-based so publishability/demo artifacts can state whether alpha only
    works at unrealistic front-of-queue placement or survives meaningful queue
    depth.
    """
    if not math.isfinite(min_edge_ticks):
        raise ValueError("min_edge_ticks must be finite")
    if not math.isfinite(min_tradable_share):
        raise ValueError("min_tradable_share must be finite")
    if not 0.0 <= min_tradable_share <= 1.0:
        raise ValueError("min_tradable_share must be in [0.0, 1.0]")

    empty = _empty_queue_position_capacity_frontier()
    if sweep.empty:
        return empty

    required = {
        "queue_position_fraction",
        "rows",
        "mean_execution_adjusted_edge_ticks",
        "tradable_share",
        "dominant_execution_side",
    }
    _require_columns(sweep, required, "queue position capacity frontier")
    values = _finite_values(
        sweep,
        [
            "queue_position_fraction",
            "rows",
            "mean_execution_adjusted_edge_ticks",
            "tradable_share",
        ],
        "queue position capacity frontier",
    )
    if not values["queue_position_fraction"].between(0.0, 1.0).all():
        raise ValueError("queue position capacity frontier fractions must be in [0, 1]")
    if (values["rows"] < 0.0).any():
        raise ValueError("queue position capacity frontier rows must be non-negative")
    if not values["tradable_share"].between(0.0, 1.0).all():
        raise ValueError("queue position capacity frontier tradable shares must be in [0, 1]")

    data = values.copy()
    data["dominant_execution_side"] = sweep["dominant_execution_side"].astype(str)
    data = data.sort_values("queue_position_fraction", ignore_index=True)
    front = data.iloc[0]
    viable = data[
        (data["mean_execution_adjusted_edge_ticks"] >= min_edge_ticks)
        & (data["tradable_share"] >= min_tradable_share)
    ]

    if viable.empty:
        result = empty.copy()
        result.update(
            {
                "rows": int(len(data)),
                "front_queue_position_fraction": float(front["queue_position_fraction"]),
                "front_mean_execution_adjusted_edge_ticks": float(
                    front["mean_execution_adjusted_edge_ticks"]
                ),
                "front_tradable_share": float(front["tradable_share"]),
                "capacity_label": "no_viable_passive_capacity",
            }
        )
        return result

    capacity = viable.iloc[-1]
    edge_decay = float(
        front["mean_execution_adjusted_edge_ticks"]
        - capacity["mean_execution_adjusted_edge_ticks"]
    )
    tradable_decay = float(front["tradable_share"] - capacity["tradable_share"])
    max_fraction = float(capacity["queue_position_fraction"])
    result = {
        "rows": int(len(data)),
        "viable_rows": int(len(viable)),
        "front_queue_position_fraction": float(front["queue_position_fraction"]),
        "max_viable_queue_position_fraction": max_fraction,
        "front_mean_execution_adjusted_edge_ticks": float(
            front["mean_execution_adjusted_edge_ticks"]
        ),
        "max_viable_mean_execution_adjusted_edge_ticks": float(
            capacity["mean_execution_adjusted_edge_ticks"]
        ),
        "edge_decay_to_capacity_ticks": edge_decay,
        "front_tradable_share": float(front["tradable_share"]),
        "max_viable_tradable_share": float(capacity["tradable_share"]),
        "tradable_share_decay_to_capacity": tradable_decay,
        "dominant_execution_side_at_capacity": str(capacity["dominant_execution_side"]),
        "capacity_label": _queue_capacity_label(
            max_fraction=max_fraction,
            edge_decay=edge_decay,
            tradable_decay=tradable_decay,
        ),
    }
    return result


def queue_position_regime_capacity_frontier(
    sweep: pd.DataFrame,
    *,
    regime_col: str = "regime",
    min_edge_ticks: float = 0.0,
    min_tradable_share: float = 0.50,
) -> pd.DataFrame:
    """Compute queue-capacity frontiers separately by liquidity/event regime.

    Global queue capacity can hide that passive edge only survives in benign
    regimes. This reducer applies ``queue_position_capacity_frontier`` within each
    regime and adds a brittleness label plus the capacity shortfall to a full-depth
    queue placement, making execution capacity publishability auditable by state.
    """
    if not isinstance(regime_col, str) or not regime_col:
        raise ValueError("regime_col must be a non-empty string")
    columns = [
        regime_col,
        "rows",
        "viable_rows",
        "front_queue_position_fraction",
        "max_viable_queue_position_fraction",
        "front_mean_execution_adjusted_edge_ticks",
        "max_viable_mean_execution_adjusted_edge_ticks",
        "edge_decay_to_capacity_ticks",
        "front_tradable_share",
        "max_viable_tradable_share",
        "tradable_share_decay_to_capacity",
        "dominant_execution_side_at_capacity",
        "capacity_label",
        "capacity_shortfall_fraction",
        "capacity_brittleness_label",
    ]
    if sweep.empty:
        return pd.DataFrame(columns=columns)
    required = {
        regime_col,
        "queue_position_fraction",
        "rows",
        "mean_execution_adjusted_edge_ticks",
        "tradable_share",
        "dominant_execution_side",
    }
    _require_columns(sweep, required, "queue position regime capacity frontier")

    rows: list[dict[str, float | int | str]] = []
    for regime, regime_sweep in sweep.groupby(regime_col, sort=True, dropna=False):
        frontier = queue_position_capacity_frontier(
            regime_sweep,
            min_edge_ticks=min_edge_ticks,
            min_tradable_share=min_tradable_share,
        )
        max_fraction = float(frontier["max_viable_queue_position_fraction"])
        shortfall = max(0.0, 1.0 - max_fraction)
        frontier.update(
            {
                regime_col: str(regime),
                "capacity_shortfall_fraction": shortfall,
                "capacity_brittleness_label": _queue_regime_capacity_brittleness_label(
                    viable_rows=int(frontier["viable_rows"]),
                    max_fraction=max_fraction,
                    capacity_label=str(frontier["capacity_label"]),
                ),
            }
        )
        rows.append(frontier)
    return pd.DataFrame(rows, columns=columns)


def queue_position_regime_capacity_concentration(
    frontier: pd.DataFrame,
    *,
    regime_col: str = "regime",
    max_front_only_share: float = 0.25,
    min_viable_regime_share: float = 0.75,
) -> dict[str, float | int | str]:
    """Summarize whether passive queue capacity is concentrated in few regimes.

    Regime-specific frontiers are useful but easy to overstate: a global passive
    edge is not publishable when most liquidity states only work at the very
    front of queue or have no viable capacity. This reducer converts the regime
    frontier into a compact release-review statistic that highlights state
    dependency before capacity is promoted into demo or paper claims.
    """
    if not isinstance(regime_col, str) or not regime_col:
        raise ValueError("regime_col must be a non-empty string")
    for name, value in {
        "max_front_only_share": max_front_only_share,
        "min_viable_regime_share": min_viable_regime_share,
    }.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0.0, 1.0]")

    empty: dict[str, float | int | str] = {
        "regimes": 0,
        "viable_regimes": 0,
        "viable_regime_share": 0.0,
        "front_only_or_no_capacity_regimes": 0,
        "front_only_or_no_capacity_share": 0.0,
        "full_capacity_regimes": 0,
        "mean_max_viable_queue_position_fraction": 0.0,
        "median_max_viable_queue_position_fraction": 0.0,
        "mean_capacity_shortfall_fraction": 0.0,
        "worst_capacity_regime": "none",
        "worst_capacity_brittleness_label": "none",
        "capacity_concentration_label": "no_regime_capacity_data",
    }
    if frontier.empty:
        return empty

    required = {
        regime_col,
        "viable_rows",
        "max_viable_queue_position_fraction",
        "max_viable_mean_execution_adjusted_edge_ticks",
        "capacity_shortfall_fraction",
        "capacity_brittleness_label",
    }
    _require_columns(frontier, required, "queue position regime capacity concentration")
    values = _finite_values(
        frontier,
        [
            "viable_rows",
            "max_viable_queue_position_fraction",
            "max_viable_mean_execution_adjusted_edge_ticks",
            "capacity_shortfall_fraction",
        ],
        "queue position regime capacity concentration",
    )
    if (values["viable_rows"] < 0.0).any():
        raise ValueError("queue position regime capacity concentration viable_rows must be non-negative")
    if not values["max_viable_queue_position_fraction"].between(0.0, 1.0).all():
        raise ValueError(
            "queue position regime capacity concentration fractions must be in [0.0, 1.0]"
        )
    if not values["capacity_shortfall_fraction"].between(0.0, 1.0).all():
        raise ValueError(
            "queue position regime capacity concentration shortfalls must be in [0.0, 1.0]"
        )

    data = values.copy()
    data[regime_col] = frontier[regime_col].astype(str)
    data["capacity_brittleness_label"] = frontier["capacity_brittleness_label"].astype(str)
    regimes = int(len(data))
    viable = data["viable_rows"] > 0.0
    front_only_or_none = (data["max_viable_queue_position_fraction"] <= 0.0) | (~viable)
    full_capacity = data["max_viable_queue_position_fraction"] >= 1.0
    viable_regimes = int(viable.sum())
    front_only_or_no_capacity_regimes = int(front_only_or_none.sum())
    viable_share = float(viable_regimes / regimes) if regimes else 0.0
    front_only_share = (
        float(front_only_or_no_capacity_regimes / regimes) if regimes else 0.0
    )
    worst = data.sort_values(
        [
            "max_viable_queue_position_fraction",
            "max_viable_mean_execution_adjusted_edge_ticks",
            regime_col,
        ],
        ascending=[True, True, True],
    ).iloc[0]
    if viable_regimes == 0:
        label = "no_viable_regime_capacity"
    elif front_only_share > max_front_only_share or viable_share < min_viable_regime_share:
        label = "capacity_regime_concentrated"
    else:
        label = "capacity_regime_diversified"

    return {
        "regimes": regimes,
        "viable_regimes": viable_regimes,
        "viable_regime_share": viable_share,
        "front_only_or_no_capacity_regimes": front_only_or_no_capacity_regimes,
        "front_only_or_no_capacity_share": front_only_share,
        "full_capacity_regimes": int(full_capacity.sum()),
        "mean_max_viable_queue_position_fraction": float(
            data["max_viable_queue_position_fraction"].mean()
        ),
        "median_max_viable_queue_position_fraction": float(
            data["max_viable_queue_position_fraction"].median()
        ),
        "mean_capacity_shortfall_fraction": float(data["capacity_shortfall_fraction"].mean()),
        "worst_capacity_regime": str(worst[regime_col]),
        "worst_capacity_brittleness_label": str(worst["capacity_brittleness_label"]),
        "capacity_concentration_label": label,
    }



def queue_position_capacity_stability(
    research_frontier: dict[str, float | int | str],
    heldout_frontier: dict[str, float | int | str],
    *,
    max_fraction_gap: float = 0.10,
    max_edge_gap_ticks: float = 0.10,
    max_tradable_share_gap: float = 0.05,
) -> dict[str, float | int | str | bool]:
    """Compare in-sample and heldout passive queue-capacity frontiers.

    Capacity can look publishable if only the research segment clears the passive
    edge/tradability gates. This reducer turns the train/heldout frontier pair
    into an explicit stability gate: how much queue depth, edge, tradable share,
    and side selection survive out of sample.
    """
    for name, value in {
        "max_fraction_gap": max_fraction_gap,
        "max_edge_gap_ticks": max_edge_gap_ticks,
        "max_tradable_share_gap": max_tradable_share_gap,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    required = {
        "viable_rows",
        "max_viable_queue_position_fraction",
        "max_viable_mean_execution_adjusted_edge_ticks",
        "max_viable_tradable_share",
        "dominant_execution_side_at_capacity",
        "capacity_label",
    }
    for label, frontier in {
        "research": research_frontier,
        "heldout": heldout_frontier,
    }.items():
        missing = sorted(required - set(frontier))
        if missing:
            raise ValueError(f"missing {label} capacity frontier keys: {missing}")

    research_fraction = float(research_frontier["max_viable_queue_position_fraction"])
    heldout_fraction = float(heldout_frontier["max_viable_queue_position_fraction"])
    research_edge = float(research_frontier["max_viable_mean_execution_adjusted_edge_ticks"])
    heldout_edge = float(heldout_frontier["max_viable_mean_execution_adjusted_edge_ticks"])
    research_tradable = float(research_frontier["max_viable_tradable_share"])
    heldout_tradable = float(heldout_frontier["max_viable_tradable_share"])
    research_viable_rows = int(research_frontier["viable_rows"])
    heldout_viable_rows = int(heldout_frontier["viable_rows"])

    numeric_values = [
        research_fraction,
        heldout_fraction,
        research_edge,
        heldout_edge,
        research_tradable,
        heldout_tradable,
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("capacity frontier numeric values must be finite")
    if not 0.0 <= research_fraction <= 1.0 or not 0.0 <= heldout_fraction <= 1.0:
        raise ValueError("capacity frontier fractions must be in [0.0, 1.0]")
    if not 0.0 <= research_tradable <= 1.0 or not 0.0 <= heldout_tradable <= 1.0:
        raise ValueError("capacity frontier tradable shares must be in [0.0, 1.0]")
    if research_viable_rows < 0 or heldout_viable_rows < 0:
        raise ValueError("capacity frontier viable rows must be non-negative")

    fraction_gap = heldout_fraction - research_fraction
    edge_gap = heldout_edge - research_edge
    tradable_gap = heldout_tradable - research_tradable
    viable_row_gap = heldout_viable_rows - research_viable_rows
    dominant_side_changed = (
        str(research_frontier["dominant_execution_side_at_capacity"])
        != str(heldout_frontier["dominant_execution_side_at_capacity"])
    )
    label = _queue_capacity_stability_label(
        fraction_gap=fraction_gap,
        edge_gap=edge_gap,
        tradable_gap=tradable_gap,
        viable_row_gap=viable_row_gap,
        dominant_side_changed=dominant_side_changed,
        heldout_label=str(heldout_frontier["capacity_label"]),
        max_fraction_gap=max_fraction_gap,
        max_edge_gap_ticks=max_edge_gap_ticks,
        max_tradable_share_gap=max_tradable_share_gap,
    )

    return {
        "research_capacity_label": str(research_frontier["capacity_label"]),
        "heldout_capacity_label": str(heldout_frontier["capacity_label"]),
        "capacity_fraction_gap": float(fraction_gap),
        "capacity_edge_gap_ticks": float(edge_gap),
        "capacity_tradable_share_gap": float(tradable_gap),
        "capacity_viable_row_gap": int(viable_row_gap),
        "dominant_side_changed": dominant_side_changed,
        "capacity_stability_label": label,
    }


def queue_position_regime_capacity_stability(
    research_frontier: pd.DataFrame,
    heldout_frontier: pd.DataFrame,
    *,
    regime_col: str = "regime",
    max_fraction_gap: float = 0.10,
    max_edge_gap_ticks: float = 0.10,
    max_tradable_share_gap: float = 0.05,
) -> pd.DataFrame:
    """Compare passive queue-capacity frontiers by regime across samples.

    A global heldout capacity check can pass while the state that matters most
    loses executable depth. This joins research and heldout regime frontiers,
    applies the same capacity stability logic per liquidity state, and flags
    missing, gained, lost, stable, and fragile regime-specific passive capacity.
    """
    if not isinstance(regime_col, str) or not regime_col:
        raise ValueError("regime_col must be a non-empty string")
    frontier_columns = {
        regime_col,
        "viable_rows",
        "max_viable_queue_position_fraction",
        "max_viable_mean_execution_adjusted_edge_ticks",
        "max_viable_tradable_share",
        "dominant_execution_side_at_capacity",
        "capacity_label",
    }
    if research_frontier.empty and heldout_frontier.empty:
        return _empty_queue_position_regime_capacity_stability(regime_col=regime_col)
    _require_columns(
        research_frontier,
        frontier_columns,
        "research regime capacity frontier",
    )
    _require_columns(
        heldout_frontier,
        frontier_columns,
        "heldout regime capacity frontier",
    )

    rows: list[dict[str, float | int | str | bool]] = []
    research_by_regime = {
        str(row[regime_col]): row for _, row in research_frontier.iterrows()
    }
    heldout_by_regime = {str(row[regime_col]): row for _, row in heldout_frontier.iterrows()}
    regimes = sorted(set(research_by_regime) | set(heldout_by_regime))
    empty_frontier = _empty_queue_position_capacity_frontier()
    for regime in regimes:
        research_missing = regime not in research_by_regime
        heldout_missing = regime not in heldout_by_regime
        research = (
            empty_frontier.copy()
            if research_missing
            else research_by_regime[regime].to_dict()
        )
        heldout = (
            empty_frontier.copy()
            if heldout_missing
            else heldout_by_regime[regime].to_dict()
        )
        stability = queue_position_capacity_stability(
            research,
            heldout,
            max_fraction_gap=max_fraction_gap,
            max_edge_gap_ticks=max_edge_gap_ticks,
            max_tradable_share_gap=max_tradable_share_gap,
        )
        research_viable_rows = int(research["viable_rows"])
        heldout_viable_rows = int(heldout["viable_rows"])
        lost_capacity = research_viable_rows > 0 and heldout_viable_rows <= 0
        gained_capacity = research_viable_rows <= 0 and heldout_viable_rows > 0
        if lost_capacity or heldout_missing:
            label = "regime_capacity_lost"
        elif gained_capacity or research_missing:
            label = "regime_capacity_gained"
        elif stability["capacity_stability_label"] == "capacity_stable":
            label = "regime_capacity_stable"
        else:
            label = "regime_capacity_fragile"
        rows.append(
            {
                regime_col: regime,
                "research_missing": bool(research_missing),
                "heldout_missing": bool(heldout_missing),
                "research_viable_rows": research_viable_rows,
                "heldout_viable_rows": heldout_viable_rows,
                "research_capacity_label": str(stability["research_capacity_label"]),
                "heldout_capacity_label": str(stability["heldout_capacity_label"]),
                "research_max_viable_queue_position_fraction": float(
                    research["max_viable_queue_position_fraction"]
                ),
                "heldout_max_viable_queue_position_fraction": float(
                    heldout["max_viable_queue_position_fraction"]
                ),
                "capacity_fraction_gap": float(stability["capacity_fraction_gap"]),
                "capacity_edge_gap_ticks": float(stability["capacity_edge_gap_ticks"]),
                "capacity_tradable_share_gap": float(
                    stability["capacity_tradable_share_gap"]
                ),
                "capacity_viable_row_gap": int(stability["capacity_viable_row_gap"]),
                "dominant_side_changed": bool(stability["dominant_side_changed"]),
                "lost_capacity": bool(lost_capacity),
                "gained_capacity": bool(gained_capacity),
                "regime_capacity_stability_label": label,
            }
        )
    output = pd.DataFrame(rows, columns=_empty_queue_position_regime_capacity_stability(regime_col=regime_col).columns)
    for column in ["research_missing", "heldout_missing", "dominant_side_changed", "lost_capacity", "gained_capacity"]:
        output[column] = output[column].astype(object)
    return output


def queue_position_regime_capacity_stability_summary(
    stability: pd.DataFrame,
    *,
    regime_col: str = "regime",
) -> dict[str, float | int | str]:
    """Reduce regime capacity stability into a publishability gate summary."""
    if not isinstance(regime_col, str) or not regime_col:
        raise ValueError("regime_col must be a non-empty string")
    empty = {
        "regimes": 0,
        "common_regimes": 0,
        "missing_research_regimes": 0,
        "missing_heldout_regimes": 0,
        "stable_regimes": 0,
        "fragile_regimes": 0,
        "lost_capacity_regimes": 0,
        "gained_capacity_regimes": 0,
        "stable_regime_share": 0.0,
        "lost_capacity_share": 0.0,
        "mean_capacity_fraction_gap": 0.0,
        "worst_regime": "none",
        "worst_regime_capacity_stability_label": "none",
        "regime_capacity_stability_label": "no_regime_capacity_stability_data",
    }
    if stability.empty:
        return empty
    required = {
        regime_col,
        "research_missing",
        "heldout_missing",
        "capacity_fraction_gap",
        "capacity_edge_gap_ticks",
        "capacity_tradable_share_gap",
        "lost_capacity",
        "gained_capacity",
        "regime_capacity_stability_label",
    }
    _require_columns(stability, required, "regime capacity stability")
    values = _finite_values(
        stability,
        [
            "capacity_fraction_gap",
            "capacity_edge_gap_ticks",
            "capacity_tradable_share_gap",
        ],
        "regime capacity stability",
    )
    data = stability.copy()
    regimes = int(len(data))
    research_missing = data["research_missing"].astype(bool)
    heldout_missing = data["heldout_missing"].astype(bool)
    common = ~(research_missing | heldout_missing)
    labels = data["regime_capacity_stability_label"].astype(str)
    lost = data["lost_capacity"].astype(bool)
    gained = data["gained_capacity"].astype(bool) | (labels == "regime_capacity_gained")
    stable = labels == "regime_capacity_stable"
    fragile = labels == "regime_capacity_fragile"
    if bool(lost.any()) or bool(heldout_missing.any()):
        label = "regime_capacity_not_replicated"
    elif bool(fragile.any()):
        label = "regime_capacity_fragile"
    elif bool(stable.any()) and int(stable.sum()) == regimes:
        label = "regime_capacity_stable"
    else:
        label = "regime_capacity_mixed"
    sort_data = pd.DataFrame(
        {
            regime_col: data[regime_col].astype(str),
            "label": labels,
            "lost": lost.astype(int),
            "heldout_missing": heldout_missing.astype(int),
            "fraction_gap": values["capacity_fraction_gap"],
            "edge_gap": values["capacity_edge_gap_ticks"],
        }
    )
    worst = sort_data.sort_values(
        ["lost", "heldout_missing", "fraction_gap", "edge_gap", regime_col],
        ascending=[False, False, True, True, True],
    ).iloc[0]
    return {
        "regimes": regimes,
        "common_regimes": int(common.sum()),
        "missing_research_regimes": int(research_missing.sum()),
        "missing_heldout_regimes": int(heldout_missing.sum()),
        "stable_regimes": int(stable.sum()),
        "fragile_regimes": int(fragile.sum()),
        "lost_capacity_regimes": int(lost.sum()),
        "gained_capacity_regimes": int(gained.sum()),
        "stable_regime_share": float(stable.sum() / regimes) if regimes else 0.0,
        "lost_capacity_share": float(lost.sum() / regimes) if regimes else 0.0,
        "mean_capacity_fraction_gap": float(values.loc[common, "capacity_fraction_gap"].mean()) if bool(common.any()) else 0.0,
        "worst_regime": str(worst[regime_col]),
        "worst_regime_capacity_stability_label": str(worst["label"]),
        "regime_capacity_stability_label": label,
    }


def passive_fill_edge_curve(
    frame: pd.DataFrame,
    *,
    bins: int = 5,
    side_col: str = "best_execution_side",
    long_return_col: str = "long_net_return_ticks",
    short_return_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Bin tradable passive opportunities by predicted fill quality.

    The curve is a lightweight calibration surface for the snapshot-based fill
    proxy. It keeps only rows where `side_col` is ``long`` or ``short``, selects
    the side-appropriate fill/adverse-fill probability and realized return, then
    reports whether higher predicted fill buckets also carry healthier realized
    execution edge. The result is intended for research review dashboards rather
    than as a formal fill simulator.
    """
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    columns = [
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "execution_adjusted_edge_ticks",
        long_return_col,
        short_return_col,
    ]
    _require_columns(frame, set(columns), "passive fill edge curve")
    if frame.empty:
        return _empty_passive_fill_edge_curve()

    values = _finite_values(
        frame,
        [
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
            "execution_adjusted_edge_ticks",
            long_return_col,
            short_return_col,
        ],
        "passive fill edge curve",
    )
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_passive_fill_edge_curve()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["side"] = selected_side
    selected["predicted_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["adverse_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_adverse_fill_probability"],
        values.loc[tradable, "ask_adverse_fill_probability"],
    )
    selected["realized_edge_ticks"] = np.where(
        selected_side == "long",
        values.loc[tradable, long_return_col],
        values.loc[tradable, short_return_col],
    )
    selected["execution_adjusted_edge_ticks"] = values.loc[tradable, "execution_adjusted_edge_ticks"]
    selected["bin"] = _rank_probability_bins(selected["predicted_fill_probability"], bins)

    rows: list[dict[str, float | int]] = []
    for bin_id, group in selected.groupby("bin", sort=True):
        rows.append(
            {
                "bin": int(bin_id),
                "rows": len(group),
                "long_rows": int((group["side"] == "long").sum()),
                "short_rows": int((group["side"] == "short").sum()),
                "mean_predicted_fill_probability": float(group["predicted_fill_probability"].mean()),
                "mean_adverse_fill_probability": float(group["adverse_fill_probability"].mean()),
                "mean_realized_edge_ticks": float(group["realized_edge_ticks"].mean()),
                "positive_edge_rate": float((group["realized_edge_ticks"] > 0.0).mean()),
                "mean_execution_adjusted_edge_ticks": float(
                    group["execution_adjusted_edge_ticks"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_passive_fill_edge_curve().columns)]


def passive_fill_threshold_policy_curve(
    frame: pd.DataFrame,
    *,
    thresholds: list[float] | tuple[float, ...] = (0.50, 0.60, 0.70, 0.80),
    side_col: str = "best_execution_side",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    long_return_col: str = "long_net_return_ticks",
    short_return_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Evaluate executable passive-fill cutoffs as a threshold policy curve.

    Calibration bins show whether probabilities are honest; execution review also
    needs the trade-off between selectivity and realized edge. This diagnostic
    selects the side-specific passive-fill probability for rows where the execution
    engine would trade, applies each cutoff, and reports coverage, realized fills,
    Brier loss, and realized/ex-ante edge so demos can defend an actionable fill
    threshold instead of cherry-picking one cutoff.
    """
    if isinstance(thresholds, (str, bytes)):
        raise ValueError("thresholds must be a non-empty sequence of finite values")
    thresholds = list(thresholds)
    if not thresholds:
        raise ValueError("thresholds must be a non-empty sequence")
    for threshold in thresholds:
        if not math.isfinite(float(threshold)):
            raise ValueError("threshold values must be finite")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold values must be in [0.0, 1.0]")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
        long_return_col,
        short_return_col,
        "execution_adjusted_edge_ticks",
    }
    _require_columns(frame, required, "passive fill threshold policy curve")
    if frame.empty:
        return _empty_passive_fill_threshold_policy_curve()

    values = _finite_values(
        frame,
        [
            "bid_fill_probability",
            "ask_fill_probability",
            bid_realized_col,
            ask_realized_col,
            long_return_col,
            short_return_col,
            "execution_adjusted_edge_ticks",
        ],
        "passive fill threshold policy curve",
    )
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_passive_fill_threshold_policy_curve()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["side"] = selected_side
    selected["predicted_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["realized_fill"] = np.where(
        selected_side == "long",
        values.loc[tradable, bid_realized_col],
        values.loc[tradable, ask_realized_col],
    )
    selected["realized_edge_ticks"] = np.where(
        selected_side == "long",
        values.loc[tradable, long_return_col],
        values.loc[tradable, short_return_col],
    )
    selected["execution_adjusted_edge_ticks"] = values.loc[tradable, "execution_adjusted_edge_ticks"]

    rows: list[dict[str, float | int | str]] = []
    total_rows = len(frame)
    for threshold in sorted(float(value) for value in thresholds):
        candidate = selected[selected["predicted_fill_probability"] >= threshold]
        candidate_rows = len(candidate)
        if candidate_rows:
            realized_fill = candidate["realized_fill"]
            predicted = candidate["predicted_fill_probability"]
            realized_edge = candidate["realized_edge_ticks"]
            adjusted_edge = candidate["execution_adjusted_edge_ticks"]
            long_rows = int((candidate["side"] == "long").sum())
            short_rows = int((candidate["side"] == "short").sum())
            mean_predicted = float(predicted.mean())
            realized_fill_rate = float(realized_fill.mean())
            brier = float(np.mean((predicted - realized_fill) ** 2))
            mean_realized_edge = float(realized_edge.mean())
            positive_edge_rate = float((realized_edge > 0.0).mean())
            mean_adjusted_edge = float(adjusted_edge.mean())
        else:
            long_rows = 0
            short_rows = 0
            mean_predicted = 0.0
            realized_fill_rate = 0.0
            brier = 0.0
            mean_realized_edge = 0.0
            positive_edge_rate = 0.0
            mean_adjusted_edge = 0.0
        rows.append(
            {
                "threshold": threshold,
                "candidate_rows": int(candidate_rows),
                "trade_share": float(candidate_rows / total_rows) if total_rows else 0.0,
                "long_rows": long_rows,
                "short_rows": short_rows,
                "mean_predicted_fill_probability": mean_predicted,
                "realized_fill_rate": realized_fill_rate,
                "weighted_brier_score": brier,
                "mean_realized_edge_ticks": mean_realized_edge,
                "positive_edge_rate": positive_edge_rate,
                "mean_execution_adjusted_edge_ticks": mean_adjusted_edge,
                "policy_label": _passive_fill_threshold_policy_label(
                    trade_share=float(candidate_rows / total_rows) if total_rows else 0.0,
                    realized_fill_rate=realized_fill_rate,
                    mean_realized_edge=mean_realized_edge,
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_passive_fill_threshold_policy_curve().columns)]


def queue_position_fill_surface(
    frame: pd.DataFrame,
    *,
    queue_bins: int = 4,
    probability_bins: int = 4,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
) -> pd.DataFrame:
    """Cross queue position depth with passive-fill calibration quality.

    This diagnostic surfaces whether the fill proxy is only calibrated for front-
    of-queue quotes or remains reliable deeper in the visible queue. Tradable rows
    are mapped to side-specific queue share, fill probability, and realized fill,
    then rank-binned within each regime across queue depth and predicted fill.
    """
    if not isinstance(queue_bins, int) or isinstance(queue_bins, bool):
        raise ValueError("queue_bins must be an integer")
    if not isinstance(probability_bins, int) or isinstance(probability_bins, bool):
        raise ValueError("probability_bins must be an integer")
    if queue_bins < 1:
        raise ValueError("queue_bins must be at least 1")
    if probability_bins < 1:
        raise ValueError("probability_bins must be at least 1")

    required = {
        side_col,
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
        "execution_adjusted_edge_ticks",
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position fill surface")
    if frame.empty:
        return _empty_queue_position_fill_surface()

    numeric_columns = [
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
        "execution_adjusted_edge_ticks",
    ]
    values = _finite_values(frame, numeric_columns, "queue position fill surface")
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_queue_position_fill_surface()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["regime"] = (
        frame.loc[tradable, regime_col].astype(str) if regime_col is not None else "all"
    )
    selected["queue_share"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_queue_share"],
        values.loc[tradable, "ask_queue_share"],
    )
    selected["predicted_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["realized_fill"] = np.where(
        selected_side == "long",
        values.loc[tradable, bid_realized_col],
        values.loc[tradable, ask_realized_col],
    )
    selected["execution_adjusted_edge_ticks"] = values.loc[
        tradable, "execution_adjusted_edge_ticks"
    ]
    if not selected["queue_share"].ge(0.0).all():
        raise ValueError("queue position fill surface queue shares must be non-negative")
    if not selected["predicted_fill_probability"].between(0.0, 1.0).all():
        raise ValueError("queue position fill surface probabilities must be in [0, 1]")
    if not selected["realized_fill"].between(0.0, 1.0).all():
        raise ValueError("queue position fill surface realized fills must be in [0, 1]")

    rows: list[dict[str, float | int | str]] = []
    for regime, regime_group in selected.groupby("regime", sort=True):
        regime_group = regime_group.copy()
        regime_group["queue_bin"] = _rank_probability_bins(
            regime_group["queue_share"], queue_bins
        )
        regime_group["fill_probability_bin"] = _rank_probability_bins(
            regime_group["predicted_fill_probability"], probability_bins
        )
        for (queue_bin, fill_bin), group in regime_group.groupby(
            ["queue_bin", "fill_probability_bin"], sort=True
        ):
            predicted = group["predicted_fill_probability"]
            realized = group["realized_fill"]
            fill_rate = float(realized.mean())
            mean_prediction = float(predicted.mean())
            error = fill_rate - mean_prediction
            rows.append(
                {
                    "regime": str(regime),
                    "queue_bin": int(queue_bin),
                    "fill_probability_bin": int(fill_bin),
                    "rows": int(len(group)),
                    "mean_queue_share": float(group["queue_share"].mean()),
                    "mean_predicted_fill_probability": mean_prediction,
                    "realized_fill_rate": fill_rate,
                    "calibration_error": error,
                    "absolute_calibration_error": abs(error),
                    "brier_score": float(((predicted - realized) ** 2).mean()),
                    "mean_execution_adjusted_edge_ticks": float(
                        group["execution_adjusted_edge_ticks"].mean()
                    ),
                }
            )
    if not rows:
        return _empty_queue_position_fill_surface()
    return pd.DataFrame(rows)[list(_empty_queue_position_fill_surface().columns)]


def queue_position_fill_calibration_surface(
    frame: pd.DataFrame,
    *,
    queue_bins: int = 4,
    probability_bins: int = 4,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
) -> pd.DataFrame:
    """Diagnose realized passive fills across side-specific queue depth.

    ``queue_position_fill_surface`` collapses long and short opportunities inside
    a regime-level grid. This side-aware companion keeps the execution side in the
    artifact so reviewers can see whether predicted bid/ask fill probabilities are
    calibrated differently as orders sit deeper in their respective visible queues.
    ``execution_adjusted_edge_ticks`` is optional; when present it is averaged into
    the surface so queue-depth calibration can be tied directly to realized edge.
    """
    if not isinstance(queue_bins, int) or isinstance(queue_bins, bool):
        raise ValueError("queue_bins must be an integer")
    if not isinstance(probability_bins, int) or isinstance(probability_bins, bool):
        raise ValueError("probability_bins must be an integer")
    if queue_bins < 1:
        raise ValueError("queue_bins must be at least 1")
    if probability_bins < 1:
        raise ValueError("probability_bins must be at least 1")

    required = {
        side_col,
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position fill calibration surface")
    columns = list(_empty_queue_position_fill_calibration_surface().columns)
    if frame.empty:
        return _empty_queue_position_fill_calibration_surface()

    numeric_columns = [
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
    ]
    has_edge = "execution_adjusted_edge_ticks" in frame.columns
    if has_edge:
        numeric_columns.append("execution_adjusted_edge_ticks")
    values = _finite_values(frame, numeric_columns, "queue position fill calibration surface")
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_queue_position_fill_calibration_surface()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["regime"] = (
        frame.loc[tradable, regime_col].astype(str) if regime_col is not None else "all"
    )
    selected["best_execution_side"] = selected_side
    selected["queue_share"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_queue_share"],
        values.loc[tradable, "ask_queue_share"],
    )
    selected["predicted_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["realized_fill"] = np.where(
        selected_side == "long",
        values.loc[tradable, bid_realized_col],
        values.loc[tradable, ask_realized_col],
    )
    selected["execution_adjusted_edge_ticks"] = (
        values.loc[tradable, "execution_adjusted_edge_ticks"] if has_edge else 0.0
    )
    if not selected["queue_share"].ge(0.0).all():
        raise ValueError("queue position fill calibration surface queue shares must be non-negative")
    if not selected["predicted_fill_probability"].between(0.0, 1.0).all():
        raise ValueError("queue position fill calibration surface probabilities must be in [0, 1]")
    if not selected["realized_fill"].between(0.0, 1.0).all():
        raise ValueError("queue position fill calibration surface realized fills must be in [0, 1]")

    rows: list[dict[str, float | int | str]] = []
    for (regime, execution_side), side_group in selected.groupby(
        ["regime", "best_execution_side"], sort=True
    ):
        side_group = side_group.copy()
        side_group["queue_share_bin"] = _rank_probability_bins(
            side_group["queue_share"], queue_bins
        )
        side_group["fill_probability_bin"] = _rank_probability_bins(
            side_group["predicted_fill_probability"], probability_bins
        )
        for (queue_bin, fill_bin), group in side_group.groupby(
            ["queue_share_bin", "fill_probability_bin"], sort=True
        ):
            predicted = group["predicted_fill_probability"]
            realized = group["realized_fill"]
            fill_rate = float(realized.mean())
            mean_prediction = float(predicted.mean())
            error = fill_rate - mean_prediction
            rows.append(
                {
                    "regime": str(regime),
                    "best_execution_side": str(execution_side),
                    "queue_share_bin": int(queue_bin),
                    "fill_probability_bin": int(fill_bin),
                    "rows": int(len(group)),
                    "mean_queue_share": float(group["queue_share"].mean()),
                    "mean_predicted_fill_probability": mean_prediction,
                    "realized_fill_rate": fill_rate,
                    "calibration_error": error,
                    "absolute_calibration_error": abs(error),
                    "brier_score": float(((predicted - realized) ** 2).mean()),
                    "mean_execution_adjusted_edge_ticks": float(
                        group["execution_adjusted_edge_ticks"].mean()
                    ),
                }
            )
    if not rows:
        return _empty_queue_position_fill_calibration_surface()
    output = pd.DataFrame(rows)[columns]
    if regime_col is not None:
        output = output.rename(columns={"regime": regime_col})
    return output


def queue_position_calibration_residual_summary(
    surface: pd.DataFrame,
    *,
    regime_col: str = "regime",
    error_threshold: float = 0.15,
) -> pd.DataFrame:
    """Rank queue-position fill model residuals by regime and execution side.

    The calibration surface shows individual queue/probability cells, but release
    review also needs an aggregate answer: which regime-side slices systematically
    overstate or understate passive fills, and does that residual coincide with
    execution edge drag? This reducer preserves side and regime, weights cells by
    their opportunity count, and labels underfilled/overfilled slices for queue
    model recalibration or conservative execution gating.
    """
    if not regime_col:
        raise ValueError("regime_col must be non-empty")
    if not math.isfinite(error_threshold) or error_threshold < 0.0:
        raise ValueError("error_threshold must be a finite non-negative value")

    columns = list(_empty_queue_position_calibration_residual_summary().columns)
    if surface.empty:
        return _empty_queue_position_calibration_residual_summary()

    required = {
        regime_col,
        "best_execution_side",
        "queue_share_bin",
        "fill_probability_bin",
        "rows",
        "mean_queue_share",
        "calibration_error",
        "absolute_calibration_error",
        "mean_execution_adjusted_edge_ticks",
    }
    _require_columns(surface, required, "queue position calibration residual summary")
    numeric_columns = [
        "queue_share_bin",
        "fill_probability_bin",
        "rows",
        "mean_queue_share",
        "calibration_error",
        "absolute_calibration_error",
        "mean_execution_adjusted_edge_ticks",
    ]
    values = _finite_values(surface, numeric_columns, "queue position calibration residual summary")
    if not values["rows"].ge(0.0).all():
        raise ValueError("queue position calibration residual summary rows must be non-negative")
    if not values["mean_queue_share"].ge(0.0).all():
        raise ValueError("queue position calibration residual summary queue shares must be non-negative")
    if not values["calibration_error"].between(-1.0, 1.0).all():
        raise ValueError("queue position calibration residual summary errors must be in [-1, 1]")
    if not values["absolute_calibration_error"].between(0.0, 1.0).all():
        raise ValueError("queue position calibration residual summary absolute errors must be in [0, 1]")

    working = surface.copy()
    working[regime_col] = working[regime_col].astype(str)
    working["best_execution_side"] = working["best_execution_side"].astype(str)
    for column in numeric_columns:
        working[column] = values[column]
    working = working[working["rows"] > 0.0]
    if working.empty:
        return _empty_queue_position_calibration_residual_summary()

    rows: list[dict[str, float | int | str]] = []
    for (regime, execution_side), group in working.groupby(
        [regime_col, "best_execution_side"], sort=True
    ):
        weights = group["rows"].to_numpy(dtype=float)
        weighted_error = float(np.average(group["calibration_error"], weights=weights))
        weighted_abs_error = float(np.average(group["absolute_calibration_error"], weights=weights))
        weighted_queue_share = float(np.average(group["mean_queue_share"], weights=weights))
        weighted_edge = float(
            np.average(group["mean_execution_adjusted_edge_ticks"], weights=weights)
        )
        worst = group.sort_values(
            ["absolute_calibration_error", "rows", "queue_share_bin", "fill_probability_bin"],
            ascending=[False, False, True, True],
        ).iloc[0]
        rows.append(
            {
                "regime": str(regime),
                "best_execution_side": str(execution_side),
                "bins": int(len(group)),
                "rows": int(group["rows"].sum()),
                "underfilled_bins": int((group["calibration_error"] < -error_threshold).sum()),
                "overfilled_bins": int((group["calibration_error"] > error_threshold).sum()),
                "weighted_mean_queue_share": weighted_queue_share,
                "weighted_calibration_error": weighted_error,
                "weighted_absolute_calibration_error": weighted_abs_error,
                "weighted_mean_execution_adjusted_edge_ticks": weighted_edge,
                "worst_queue_share_bin": int(worst["queue_share_bin"]),
                "worst_fill_probability_bin": int(worst["fill_probability_bin"]),
                "worst_absolute_calibration_error": float(worst["absolute_calibration_error"]),
                "residual_label": _queue_calibration_residual_label(
                    weighted_error=weighted_error,
                    weighted_abs_error=weighted_abs_error,
                    weighted_edge=weighted_edge,
                    error_threshold=error_threshold,
                ),
            }
        )
    output = pd.DataFrame(rows, columns=columns)
    if regime_col != "regime":
        output = output.rename(columns={"regime": regime_col})
    return output.sort_values(
        ["weighted_absolute_calibration_error", "rows", regime_col, "best_execution_side"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)



def queue_position_calibration_drift(
    surface: pd.DataFrame,
    *,
    regime_col: str = "regime",
    min_regimes: int = 2,
    min_rows: int = 1,
) -> pd.DataFrame:
    """Audit queue-position fill calibration drift across regimes.

    The side-aware calibration surface is useful only if a bin's reliability is
    stable across regimes. This reducer holds execution side, queue bin, and
    predicted-fill bin fixed, then measures the cross-regime range in realized
    fill rates and absolute calibration error. It highlights bins where passive
    queue placement looks calibrated in aggregate but fractures in stress/open
    regimes, making the execution-adjusted LCRI evidence easier to review.
    """
    if not isinstance(min_regimes, int) or isinstance(min_regimes, bool):
        raise ValueError("min_regimes must be an integer")
    if min_regimes < 2:
        raise ValueError("min_regimes must be at least 2")
    if not isinstance(min_rows, int) or isinstance(min_rows, bool):
        raise ValueError("min_rows must be an integer")
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1")
    if not regime_col:
        raise ValueError("regime_col must be non-empty")

    columns = list(_empty_queue_position_calibration_drift().columns)
    if surface.empty:
        return _empty_queue_position_calibration_drift()

    required = {
        regime_col,
        "best_execution_side",
        "queue_share_bin",
        "fill_probability_bin",
        "rows",
        "realized_fill_rate",
        "absolute_calibration_error",
    }
    _require_columns(surface, required, "queue position calibration drift")
    values = _finite_values(
        surface,
        [
            "queue_share_bin",
            "fill_probability_bin",
            "rows",
            "realized_fill_rate",
            "absolute_calibration_error",
        ],
        "queue position calibration drift",
    )
    if not values["rows"].ge(0.0).all():
        raise ValueError("queue position calibration drift rows must be non-negative")
    if not values["realized_fill_rate"].between(0.0, 1.0).all():
        raise ValueError("queue position calibration drift fill rates must be in [0, 1]")
    if not values["absolute_calibration_error"].between(0.0, 1.0).all():
        raise ValueError("queue position calibration drift errors must be in [0, 1]")

    working = surface.copy()
    working[regime_col] = working[regime_col].astype(str)
    working["best_execution_side"] = working["best_execution_side"].astype(str)
    for column in [
        "queue_share_bin",
        "fill_probability_bin",
        "rows",
        "realized_fill_rate",
        "absolute_calibration_error",
    ]:
        working[column] = values[column]
    working = working[working["rows"] >= float(min_rows)]
    if working.empty:
        return _empty_queue_position_calibration_drift()

    rows: list[dict[str, float | int | str]] = []
    group_cols = ["best_execution_side", "queue_share_bin", "fill_probability_bin"]
    for (execution_side, queue_bin, fill_bin), group in working.groupby(group_cols, sort=True):
        regimes = int(group[regime_col].nunique())
        if regimes < min_regimes:
            continue
        total_rows = int(group["rows"].sum())
        if total_rows <= 0:
            continue
        worst = group.sort_values(
            ["absolute_calibration_error", "rows", regime_col], ascending=[False, False, True]
        ).iloc[0]
        fill_range = float(group["realized_fill_rate"].max() - group["realized_fill_rate"].min())
        calibration_range = float(
            group["absolute_calibration_error"].max() - group["absolute_calibration_error"].min()
        )
        weighted_error = float(
            np.average(group["absolute_calibration_error"], weights=group["rows"])
        )
        rows.append(
            {
                "best_execution_side": str(execution_side),
                "queue_share_bin": int(queue_bin),
                "fill_probability_bin": int(fill_bin),
                "regimes": regimes,
                "rows": total_rows,
                "min_realized_fill_rate": float(group["realized_fill_rate"].min()),
                "max_realized_fill_rate": float(group["realized_fill_rate"].max()),
                "fill_rate_range": fill_range,
                "min_absolute_calibration_error": float(group["absolute_calibration_error"].min()),
                "max_absolute_calibration_error": float(group["absolute_calibration_error"].max()),
                "calibration_error_range": calibration_range,
                "weighted_mean_absolute_calibration_error": weighted_error,
                "worst_regime": str(worst[regime_col]),
                "drift_label": _queue_calibration_drift_label(
                    fill_rate_range=fill_range,
                    calibration_error_range=calibration_range,
                    weighted_error=weighted_error,
                ),
            }
        )
    if not rows:
        return _empty_queue_position_calibration_drift()
    return pd.DataFrame(rows, columns=columns)


def queue_position_edge_decay(surface: pd.DataFrame, *, min_rows: int = 1) -> pd.DataFrame:
    """Summarize how passive execution edge decays as queue position gets deeper.

    ``queue_position_fill_surface`` gives a two-dimensional calibration grid. This
    reducer collapses the fill-probability axis to a queue-depth frontier per
    regime, then compares front-of-queue and back-of-queue realized fill, predicted
    fill, calibration error, and execution-adjusted edge. The output is designed
    to answer the placement question: does a signal remain investable deeper in
    the visible queue, or is it mostly a front-of-queue opportunity?
    """
    if not isinstance(min_rows, int) or isinstance(min_rows, bool):
        raise ValueError("min_rows must be an integer")
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1")
    columns = list(_empty_queue_position_edge_decay().columns)
    if surface.empty:
        return _empty_queue_position_edge_decay()

    required = {
        "regime",
        "queue_bin",
        "rows",
        "mean_queue_share",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "absolute_calibration_error",
        "mean_execution_adjusted_edge_ticks",
    }
    _require_columns(surface, required, "queue position edge decay")
    values = _finite_values(
        surface,
        [
            "queue_bin",
            "rows",
            "mean_queue_share",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "absolute_calibration_error",
            "mean_execution_adjusted_edge_ticks",
        ],
        "queue position edge decay",
    )
    if (values["rows"] < 0.0).any():
        raise ValueError("queue position edge decay rows must be non-negative")
    if not values["mean_predicted_fill_probability"].between(0.0, 1.0).all():
        raise ValueError("queue position edge decay predicted probabilities must be in [0, 1]")
    if not values["realized_fill_rate"].between(0.0, 1.0).all():
        raise ValueError("queue position edge decay realized fill rates must be in [0, 1]")

    data = values.copy()
    data["regime"] = surface["regime"].astype(str)
    rows: list[dict[str, float | int | str | bool]] = []
    for regime, regime_group in data.groupby("regime", sort=True):
        queue_rows = []
        for queue_bin, queue_group in regime_group.groupby("queue_bin", sort=True):
            total_rows = float(queue_group["rows"].sum())
            if total_rows < float(min_rows):
                continue
            weights = queue_group["rows"] / total_rows if total_rows > 0.0 else queue_group["rows"]
            queue_rows.append(
                {
                    "queue_bin": int(queue_bin),
                    "rows": int(total_rows),
                    "mean_queue_share": float((queue_group["mean_queue_share"] * weights).sum()),
                    "mean_predicted_fill_probability": float(
                        (queue_group["mean_predicted_fill_probability"] * weights).sum()
                    ),
                    "realized_fill_rate": float((queue_group["realized_fill_rate"] * weights).sum()),
                    "absolute_calibration_error": float(
                        (queue_group["absolute_calibration_error"] * weights).sum()
                    ),
                    "mean_execution_adjusted_edge_ticks": float(
                        (queue_group["mean_execution_adjusted_edge_ticks"] * weights).sum()
                    ),
                }
            )
        if not queue_rows:
            continue
        frontier = pd.DataFrame(queue_rows).sort_values("queue_bin").reset_index(drop=True)
        front = frontier.iloc[0]
        back = frontier.iloc[-1]
        edges = frontier["mean_execution_adjusted_edge_ticks"]
        worst_idx = edges.idxmin()
        edge_decay = float(front["mean_execution_adjusted_edge_ticks"] - back["mean_execution_adjusted_edge_ticks"])
        fill_decay = float(front["realized_fill_rate"] - back["realized_fill_rate"])
        calibration_widening = float(
            back["absolute_calibration_error"] - front["absolute_calibration_error"]
        )
        monotonic_edge_decay = bool((edges.diff().fillna(0.0) <= 1e-12).all())
        rows.append(
            {
                "regime": str(regime),
                "queue_bins": int(len(frontier)),
                "rows": int(frontier["rows"].sum()),
                "front_queue_bin": int(front["queue_bin"]),
                "back_queue_bin": int(back["queue_bin"]),
                "front_mean_queue_share": float(front["mean_queue_share"]),
                "back_mean_queue_share": float(back["mean_queue_share"]),
                "fill_rate_decay": fill_decay,
                "predicted_fill_decay": float(
                    front["mean_predicted_fill_probability"]
                    - back["mean_predicted_fill_probability"]
                ),
                "edge_decay_ticks": edge_decay,
                "calibration_error_widening": calibration_widening,
                "monotonic_edge_decay": monotonic_edge_decay,
                "worst_queue_bin": int(frontier.loc[worst_idx, "queue_bin"]),
                "worst_mean_execution_adjusted_edge_ticks": float(edges.loc[worst_idx]),
                "queue_decay_label": _queue_decay_label(
                    edge_decay=edge_decay,
                    fill_decay=fill_decay,
                    calibration_widening=calibration_widening,
                ),
            }
        )
    if not rows:
        return _empty_queue_position_edge_decay()
    return pd.DataFrame(rows)[columns].sort_values(
        ["edge_decay_ticks", "fill_rate_decay", "regime"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def _empty_queue_position_execution_quality_gate() -> dict[str, float | int | str]:
    return {
        "surface_rows": 0,
        "decay_rows": 0,
        "surface_regimes": 0,
        "decay_regimes": 0,
        "eligible_regimes": 0,
        "blocked_regimes": 0,
        "weighted_absolute_calibration_error": 0.0,
        "weighted_brier_score": 0.0,
        "weighted_edge_decay_ticks": 0.0,
        "worst_calibration_regime": "none",
        "worst_decay_regime": "none",
        "max_regime_absolute_calibration_error": 0.0,
        "max_calibration_error_widening": 0.0,
        "non_monotonic_decay_regimes": 0,
        "quality_gate_label": "empty_queue_execution_surface",
    }


def queue_position_execution_quality_gate(
    surface: pd.DataFrame,
    decay: pd.DataFrame,
    *,
    drift: pd.DataFrame | None = None,
    max_expected_calibration_error: float = 0.15,
    max_expected_brier_score: float = 0.15,
    max_calibration_widening: float = 0.10,
    max_drift_fill_rate_range: float = 0.25,
    max_drift_calibration_error_range: float = 0.15,
) -> dict[str, float | int | str]:
    """Gate queue-position fill surfaces before treating passive edge as publishable.

    The surface-level calibration check asks whether predicted passive fills are
    empirically reliable across queue-depth buckets. The decay check asks whether
    edge degrades monotonically as quote placement moves deeper into the visible
    queue. When supplied, the drift table additionally blocks bins whose realized
    fill reliability fractures across regimes at a fixed side/queue/fill bin.
    Combining these prevents a superficially attractive LCRI signal from passing
    review when its execution evidence is driven by miscalibrated, regime-fragile,
    or non-monotone queue-depth regimes.
    """
    for name, value in {
        "max_expected_calibration_error": max_expected_calibration_error,
        "max_expected_brier_score": max_expected_brier_score,
        "max_calibration_widening": max_calibration_widening,
        "max_drift_fill_rate_range": max_drift_fill_rate_range,
        "max_drift_calibration_error_range": max_drift_calibration_error_range,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative value")
    if surface.empty and decay.empty and (drift is None or drift.empty):
        return _empty_queue_position_execution_quality_gate()

    surface_required = {"regime", "rows", "absolute_calibration_error", "brier_score"}
    decay_required = {
        "regime",
        "rows",
        "edge_decay_ticks",
        "calibration_error_widening",
        "monotonic_edge_decay",
    }
    _require_columns(surface, surface_required, "queue execution quality surface")
    _require_columns(decay, decay_required, "queue execution quality decay")
    surface_values = _finite_values(
        surface,
        ["rows", "absolute_calibration_error", "brier_score"],
        "queue execution quality surface",
    )
    decay_values = _finite_values(
        decay,
        ["rows", "edge_decay_ticks", "calibration_error_widening"],
        "queue execution quality decay",
    )
    if (surface_values["rows"] < 0.0).any() or (decay_values["rows"] < 0.0).any():
        raise ValueError("queue execution quality rows must be non-negative")
    if (surface_values[["absolute_calibration_error", "brier_score"]] < 0.0).any().any():
        raise ValueError("queue execution quality calibration metrics must be non-negative")

    surface_data = surface_values.copy()
    surface_data["regime"] = surface["regime"].astype(str)
    regime_surface = surface_data.groupby("regime", sort=True).apply(
        _weighted_queue_quality_surface_row,
        include_groups=False,
    )
    if isinstance(regime_surface, pd.Series):
        regime_surface = regime_surface.unstack()
    regime_surface = regime_surface.reset_index()

    decay_data = decay_values.copy()
    decay_data["regime"] = decay["regime"].astype(str)
    decay_data["monotonic_edge_decay"] = decay["monotonic_edge_decay"].astype(bool)

    surface_rows = int(surface_values["rows"].sum())
    decay_rows = int(decay_values["rows"].sum())
    surface_weights = surface_values["rows"] / surface_rows if surface_rows > 0 else surface_values["rows"]
    decay_weights = decay_values["rows"] / decay_rows if decay_rows > 0 else decay_values["rows"]

    if regime_surface.empty:
        worst_calibration_regime = "none"
        max_regime_error = 0.0
        high_error_regimes: set[str] = set()
    else:
        worst_idx = regime_surface["absolute_calibration_error"].idxmax()
        worst_calibration_regime = str(regime_surface.loc[worst_idx, "regime"])
        max_regime_error = float(regime_surface["absolute_calibration_error"].max())
        high_error_regimes = set(
            regime_surface.loc[
                (regime_surface["absolute_calibration_error"] > max_expected_calibration_error)
                | (regime_surface["brier_score"] > max_expected_brier_score),
                "regime",
            ].astype(str)
        )

    if decay_data.empty:
        worst_decay_regime = "none"
        max_widening = 0.0
        non_monotonic = 0
        decay_block_regimes: set[str] = set()
    else:
        worst_decay_idx = decay_data["calibration_error_widening"].idxmax()
        worst_decay_regime = str(decay_data.loc[worst_decay_idx, "regime"])
        max_widening = float(decay_data["calibration_error_widening"].max())
        non_monotonic = int((~decay_data["monotonic_edge_decay"]).sum())
        decay_block_regimes = set(
            decay_data.loc[
                (decay_data["calibration_error_widening"] > max_calibration_widening)
                | (~decay_data["monotonic_edge_decay"]),
                "regime",
            ].astype(str)
        )

    drift_metrics: dict[str, float | int | str] = {}
    drift_block_regimes: set[str] = set()
    drift_watch_bins = 0
    drift_unstable_bins = 0
    if drift is not None:
        drift_required = {
            "rows",
            "fill_rate_range",
            "calibration_error_range",
            "weighted_mean_absolute_calibration_error",
            "worst_regime",
            "drift_label",
        }
        _require_columns(drift, drift_required, "queue execution quality drift")
        drift_values = _finite_values(
            drift,
            [
                "rows",
                "fill_rate_range",
                "calibration_error_range",
                "weighted_mean_absolute_calibration_error",
            ],
            "queue execution quality drift",
        )
        if (drift_values["rows"] < 0.0).any():
            raise ValueError("queue execution quality drift rows must be non-negative")
        if (drift_values.drop(columns=["rows"]) < 0.0).any().any():
            raise ValueError("queue execution quality drift metrics must be non-negative")
        if drift.empty:
            drift_metrics = {
                "drift_rows": 0,
                "drift_bins": 0,
                "unstable_drift_bins": 0,
                "watch_drift_bins": 0,
                "worst_drift_regime": "none",
                "max_drift_fill_rate_range": 0.0,
                "max_drift_calibration_error_range": 0.0,
                "weighted_drift_absolute_calibration_error": 0.0,
            }
        else:
            drift_labels = drift["drift_label"].astype(str)
            drift_block_mask = (
                (drift_labels == "calibration_unstable")
                | (drift_values["fill_rate_range"] > max_drift_fill_rate_range)
                | (drift_values["calibration_error_range"] > max_drift_calibration_error_range)
            )
            drift_watch_bins = int((drift_labels == "calibration_watch").sum())
            drift_unstable_bins = int(drift_block_mask.sum())
            drift_block_regimes = set(drift.loc[drift_block_mask, "worst_regime"].astype(str))
            worst_drift_idx = drift_values["calibration_error_range"].idxmax()
            drift_rows = int(drift_values["rows"].sum())
            drift_weights = drift_values["rows"] / drift_rows if drift_rows > 0 else drift_values["rows"]
            drift_metrics = {
                "drift_rows": drift_rows,
                "drift_bins": int(len(drift)),
                "unstable_drift_bins": drift_unstable_bins,
                "watch_drift_bins": drift_watch_bins,
                "worst_drift_regime": str(drift.loc[worst_drift_idx, "worst_regime"]),
                "max_drift_fill_rate_range": float(drift_values["fill_rate_range"].max()),
                "max_drift_calibration_error_range": float(
                    drift_values["calibration_error_range"].max()
                ),
                "weighted_drift_absolute_calibration_error": float(
                    (
                        drift_values["weighted_mean_absolute_calibration_error"]
                        * drift_weights
                    ).sum()
                ),
            }

    blocked_regimes = high_error_regimes | decay_block_regimes | drift_block_regimes
    eligible_regimes = set(surface_data["regime"]) | set(decay_data["regime"])
    if drift is not None and not drift.empty:
        eligible_regimes |= set(drift["worst_regime"].astype(str))
    label = _queue_execution_quality_label(
        blocked_regimes=len(blocked_regimes),
        eligible_regimes=len(eligible_regimes),
        weighted_calibration_error=float(
            (surface_values["absolute_calibration_error"] * surface_weights).sum()
        ),
        weighted_brier_score=float((surface_values["brier_score"] * surface_weights).sum()),
        max_expected_calibration_error=max_expected_calibration_error,
        max_expected_brier_score=max_expected_brier_score,
    )

    gate = {
        "surface_rows": surface_rows,
        "decay_rows": decay_rows,
        "surface_regimes": int(surface_data["regime"].nunique()),
        "decay_regimes": int(decay_data["regime"].nunique()),
        "eligible_regimes": len(eligible_regimes),
        "blocked_regimes": len(blocked_regimes),
        "weighted_absolute_calibration_error": float(
            (surface_values["absolute_calibration_error"] * surface_weights).sum()
        ),
        "weighted_brier_score": float((surface_values["brier_score"] * surface_weights).sum()),
        "weighted_edge_decay_ticks": float((decay_values["edge_decay_ticks"] * decay_weights).sum()),
        "worst_calibration_regime": worst_calibration_regime,
        "worst_decay_regime": worst_decay_regime,
        "max_regime_absolute_calibration_error": max_regime_error,
        "max_calibration_error_widening": max_widening,
        "non_monotonic_decay_regimes": non_monotonic,
        "quality_gate_label": label,
    }
    gate.update(drift_metrics)
    return gate


def queue_position_execution_readiness_scorecard(
    quality_gate: dict[str, float | int | str | bool],
    capacity_stability: dict[str, float | int | str | bool],
    capacity_concentration: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str | bool]:
    """Combine queue execution quality, stability, and capacity into one release gate.

    Execution-adjusted LCRI should not be published solely because a fill model is
    calibrated on average. This scorecard joins three orthogonal blockers:
    side-aware queue calibration quality, out-of-sample capacity stability, and
    regime concentration of viable passive capacity. The result is a compact
    artifact for demos/review packets that states whether execution evidence is
    publishable or still needs queue-model/capacity remediation.
    """
    quality_required = {
        "quality_gate_label",
        "blocked_regimes",
        "eligible_regimes",
        "weighted_absolute_calibration_error",
        "weighted_brier_score",
        "max_regime_absolute_calibration_error",
        "worst_calibration_regime",
        "worst_decay_regime",
    }
    stability_required = {
        "capacity_stability_label",
        "capacity_fraction_gap",
        "capacity_edge_gap_ticks",
        "capacity_tradable_share_gap",
        "dominant_side_changed",
    }
    missing_quality = sorted(quality_required - set(quality_gate))
    if missing_quality:
        raise ValueError(f"missing queue execution readiness quality keys: {missing_quality}")
    missing_stability = sorted(stability_required - set(capacity_stability))
    if missing_stability:
        raise ValueError(f"missing queue execution readiness stability keys: {missing_stability}")

    quality_numeric = {
        "blocked_regimes": float(quality_gate["blocked_regimes"]),
        "eligible_regimes": float(quality_gate["eligible_regimes"]),
        "weighted_absolute_calibration_error": float(
            quality_gate["weighted_absolute_calibration_error"]
        ),
        "weighted_brier_score": float(quality_gate["weighted_brier_score"]),
        "max_regime_absolute_calibration_error": float(
            quality_gate["max_regime_absolute_calibration_error"]
        ),
    }
    stability_numeric = {
        "capacity_fraction_gap": float(capacity_stability["capacity_fraction_gap"]),
        "capacity_edge_gap_ticks": float(capacity_stability["capacity_edge_gap_ticks"]),
        "capacity_tradable_share_gap": float(capacity_stability["capacity_tradable_share_gap"]),
    }
    if not all(math.isfinite(value) for value in quality_numeric.values()):
        raise ValueError("queue execution readiness quality values must be finite")
    if not all(math.isfinite(value) for value in stability_numeric.values()):
        raise ValueError("queue execution readiness stability values must be finite")
    if quality_numeric["blocked_regimes"] < 0.0 or quality_numeric["eligible_regimes"] < 0.0:
        raise ValueError("queue execution readiness regime counts must be non-negative")
    if (
        quality_numeric["weighted_absolute_calibration_error"] < 0.0
        or quality_numeric["weighted_brier_score"] < 0.0
        or quality_numeric["max_regime_absolute_calibration_error"] < 0.0
    ):
        raise ValueError("queue execution readiness quality errors must be non-negative")

    concentration_label = "not_supplied"
    front_only_share = 0.0
    worst_capacity_regime = "none"
    concentration_blocked = False
    if capacity_concentration is not None:
        concentration_required = {
            "capacity_concentration_label",
            "front_only_or_no_capacity_share",
            "worst_capacity_regime",
        }
        missing_concentration = sorted(concentration_required - set(capacity_concentration))
        if missing_concentration:
            raise ValueError(
                f"missing queue execution readiness concentration keys: {missing_concentration}"
            )
        concentration_label = str(capacity_concentration["capacity_concentration_label"])
        front_only_share = float(capacity_concentration["front_only_or_no_capacity_share"])
        worst_capacity_regime = str(capacity_concentration["worst_capacity_regime"])
        if not math.isfinite(front_only_share) or not 0.0 <= front_only_share <= 1.0:
            raise ValueError("queue execution readiness concentration share must be in [0, 1]")
        concentration_blocked = concentration_label in {
            "capacity_regime_concentrated",
            "capacity_regime_blocked",
        }

    quality_blocked = str(quality_gate["quality_gate_label"]) == "queue_execution_blocked"
    stability_blocked = str(capacity_stability["capacity_stability_label"]) == "capacity_fragile"
    blocker_count = int(quality_blocked) + int(stability_blocked) + int(concentration_blocked)
    if blocker_count > 0:
        readiness_label = "execution_not_publishable"
    elif str(quality_gate["quality_gate_label"]) == "queue_execution_review":
        readiness_label = "execution_review"
    else:
        readiness_label = "execution_publishable"

    return {
        "quality_gate_label": str(quality_gate["quality_gate_label"]),
        "capacity_stability_label": str(capacity_stability["capacity_stability_label"]),
        "capacity_concentration_label": concentration_label,
        "blocked_regimes": int(quality_numeric["blocked_regimes"]),
        "eligible_regimes": int(quality_numeric["eligible_regimes"]),
        "execution_blocker_count": blocker_count,
        "worst_calibration_regime": str(quality_gate["worst_calibration_regime"]),
        "worst_decay_regime": str(quality_gate["worst_decay_regime"]),
        "worst_capacity_regime": worst_capacity_regime,
        "weighted_absolute_calibration_error": quality_numeric[
            "weighted_absolute_calibration_error"
        ],
        "weighted_brier_score": quality_numeric["weighted_brier_score"],
        "max_regime_absolute_calibration_error": quality_numeric[
            "max_regime_absolute_calibration_error"
        ],
        "capacity_fraction_gap": stability_numeric["capacity_fraction_gap"],
        "capacity_edge_gap_ticks": stability_numeric["capacity_edge_gap_ticks"],
        "capacity_tradable_share_gap": stability_numeric["capacity_tradable_share_gap"],
        "front_only_or_no_capacity_share": front_only_share,
        "dominant_side_changed": bool(capacity_stability["dominant_side_changed"]),
        "execution_readiness_label": readiness_label,
    }


def _weighted_queue_quality_surface_row(group: pd.DataFrame) -> pd.Series:
    total_rows = float(group["rows"].sum())
    weights = group["rows"] / total_rows if total_rows > 0.0 else group["rows"]
    return pd.Series(
        {
            "rows": int(total_rows),
            "absolute_calibration_error": float((group["absolute_calibration_error"] * weights).sum()),
            "brier_score": float((group["brier_score"] * weights).sum()),
        }
    )


def _queue_execution_quality_label(
    *,
    blocked_regimes: int,
    eligible_regimes: int,
    weighted_calibration_error: float,
    weighted_brier_score: float,
    max_expected_calibration_error: float,
    max_expected_brier_score: float,
) -> str:
    if eligible_regimes == 0:
        return "empty_queue_execution_surface"
    if blocked_regimes > 0:
        return "queue_execution_blocked"
    if (
        weighted_calibration_error > max_expected_calibration_error * 0.75
        or weighted_brier_score > max_expected_brier_score * 0.75
    ):
        return "queue_execution_review"
    return "queue_execution_publishable"


def passive_fill_calibration_curve(
    frame: pd.DataFrame,
    *,
    bins: int = 5,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
) -> pd.DataFrame:
    """Calibrate side-specific passive fill probabilities against realized fills.

    Rows are reduced to executable ``long``/``short`` decisions, mapped to the
    matching bid/ask predicted fill probability and realized fill flag, then
    rank-binned within each regime. This is the bridge from the snapshot proxy to
    event-level add/cancel/trade validation: high predicted buckets should carry
    higher realized fill rates with small calibration error and Brier loss.
    """
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "passive fill calibration")
    if frame.empty:
        return _empty_passive_fill_calibration_curve()

    values = _finite_values(
        frame,
        ["bid_fill_probability", "ask_fill_probability", bid_realized_col, ask_realized_col],
        "passive fill calibration",
    )
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_passive_fill_calibration_curve()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["side"] = selected_side
    selected["regime"] = (
        frame.loc[tradable, regime_col].astype(str) if regime_col is not None else "all"
    )
    selected["predicted_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["realized_fill"] = np.where(
        selected_side == "long",
        values.loc[tradable, bid_realized_col],
        values.loc[tradable, ask_realized_col],
    )
    if not selected["realized_fill"].between(0.0, 1.0).all():
        raise ValueError("passive fill calibration realized fills must be in [0, 1]")
    if not selected["predicted_fill_probability"].between(0.0, 1.0).all():
        raise ValueError("passive fill calibration probabilities must be in [0, 1]")

    rows: list[dict[str, float | int | str]] = []
    for regime, regime_group in selected.groupby("regime", sort=True):
        regime_group = regime_group.copy()
        regime_group["bin"] = _rank_probability_bins(
            regime_group["predicted_fill_probability"], bins
        )
        for bin_id, group in regime_group.groupby("bin", sort=True):
            predicted = group["predicted_fill_probability"]
            realized = group["realized_fill"]
            fill_rate = float(realized.mean())
            mean_prediction = float(predicted.mean())
            error = fill_rate - mean_prediction
            rows.append(
                {
                    "regime": str(regime),
                    "bin": int(bin_id),
                    "rows": int(len(group)),
                    "long_rows": int((group["side"] == "long").sum()),
                    "short_rows": int((group["side"] == "short").sum()),
                    "mean_predicted_fill_probability": mean_prediction,
                    "realized_fill_rate": fill_rate,
                    "calibration_error": error,
                    "absolute_calibration_error": abs(error),
                    "brier_score": float(((predicted - realized) ** 2).mean()),
                }
            )
    return pd.DataFrame(rows)[list(_empty_passive_fill_calibration_curve().columns)]


def passive_fill_calibration_summary(curve: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize passive-fill calibration curve quality with row weighting."""
    if curve.empty:
        return _empty_passive_fill_calibration_summary()
    required = {
        "regime",
        "bin",
        "rows",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "calibration_error",
        "absolute_calibration_error",
        "brier_score",
    }
    _require_columns(curve, required, "passive fill calibration summary")
    values = _finite_values(
        curve,
        [
            "rows",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "calibration_error",
            "absolute_calibration_error",
            "brier_score",
        ],
        "passive fill calibration summary",
    )
    if (values["rows"] < 0.0).any():
        raise ValueError("passive fill calibration summary rows must be non-negative")
    total_rows = int(values["rows"].sum())
    if total_rows == 0:
        return _empty_passive_fill_calibration_summary()
    weights = values["rows"] / total_rows
    worst_idx = values["absolute_calibration_error"].idxmax()
    return {
        "rows": total_rows,
        "bins": int(len(curve)),
        "regimes": int(curve["regime"].astype(str).nunique()),
        "weighted_mean_predicted_fill_probability": float(
            (values["mean_predicted_fill_probability"] * weights).sum()
        ),
        "weighted_realized_fill_rate": float((values["realized_fill_rate"] * weights).sum()),
        "weighted_calibration_error": float((values["calibration_error"] * weights).sum()),
        "expected_calibration_error": float(
            (values["absolute_calibration_error"] * weights).sum()
        ),
        "weighted_brier_score": float((values["brier_score"] * weights).sum()),
        "worst_regime": str(curve.loc[worst_idx, "regime"]),
        "worst_absolute_calibration_error": float(values.loc[worst_idx, "absolute_calibration_error"]),
    }


def event_level_passive_fill_horizon_sweep(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons: list[float] | tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
    bins: int = 5,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
    event_side_col: str = "side",
    event_price_col: str = "price",
    event_size_col: str = "size",
) -> pd.DataFrame:
    """Calibrate passive fill probabilities against event-level queue depletion horizons.

    Snapshot horizon sweeps are useful when only L2 states are available, but a
    publishable execution claim should prefer add/cancel/trade message evidence
    when available. This sweep relabels each passive quote with event-level queue
    depletion over multiple time horizons, then reports calibration drift versus
    the shortest event window.
    """
    if isinstance(horizons, (str, bytes)):
        raise ValueError("horizons must be a non-empty sequence of finite positive values")
    horizons = list(horizons)
    if not horizons:
        raise ValueError("horizons must be a non-empty sequence")
    for horizon in horizons:
        if not math.isfinite(float(horizon)) or float(horizon) <= 0.0:
            raise ValueError("horizon values must be finite positive values")
    if len(set(float(horizon) for horizon in horizons)) != len(horizons):
        raise ValueError("horizon values must be unique")
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    rows: list[dict[str, float | int | str]] = []
    anchor_realized_rate: float | None = None
    anchor_brier: float | None = None
    for index, horizon in enumerate(sorted(float(value) for value in horizons)):
        realized = add_event_level_realized_fill_proxy(
            snapshots,
            events,
            horizon=horizon,
            group_cols=group_cols,
            timestamp_col=timestamp_col,
            event_type_col=event_type_col,
            event_side_col=event_side_col,
            event_price_col=event_price_col,
            event_size_col=event_size_col,
            bid_realized_col="_event_sweep_bid_realized_fill",
            ask_realized_col="_event_sweep_ask_realized_fill",
        )
        curve = passive_fill_calibration_curve(
            realized,
            bins=bins,
            side_col=side_col,
            regime_col=regime_col,
            bid_realized_col="_event_sweep_bid_realized_fill",
            ask_realized_col="_event_sweep_ask_realized_fill",
        )
        summary = passive_fill_calibration_summary(curve)
        realized_rate = float(summary["weighted_realized_fill_rate"])
        brier = float(summary["weighted_brier_score"])
        if index == 0:
            anchor_realized_rate = realized_rate
            anchor_brier = brier
        assert anchor_realized_rate is not None
        assert anchor_brier is not None
        realized_gap = realized_rate - anchor_realized_rate
        brier_gap = brier - anchor_brier
        rows.append(
            {
                "horizon": float(horizon),
                "event_depletion_source": "events",
                "rows": int(summary["rows"]),
                "bins": int(summary["bins"]),
                "regimes": int(summary["regimes"]),
                "weighted_mean_predicted_fill_probability": float(
                    summary["weighted_mean_predicted_fill_probability"]
                ),
                "weighted_realized_fill_rate": realized_rate,
                "weighted_calibration_error": float(summary["weighted_calibration_error"]),
                "expected_calibration_error": float(summary["expected_calibration_error"]),
                "weighted_brier_score": brier,
                "worst_absolute_calibration_error": float(
                    summary["worst_absolute_calibration_error"]
                ),
                "realized_fill_rate_gap_vs_shortest": float(realized_gap),
                "brier_score_gap_vs_shortest": float(brier_gap),
                "horizon_stability_label": _passive_fill_horizon_stability_label(
                    index=index,
                    realized_gap=realized_gap,
                    brier_gap=brier_gap,
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_event_level_passive_fill_horizon_sweep().columns)]


def passive_fill_realization_horizon_sweep(
    frame: pd.DataFrame,
    *,
    horizons: list[int] | tuple[int, ...] = (1, 2, 3, 5),
    bins: int = 5,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
) -> pd.DataFrame:
    """Stress-test passive fill calibration across realized-depletion horizons.

    Queue-position labels inferred from snapshots are horizon-sensitive: one-step
    depletion is conservative, while longer horizons can include slower queue
    clearance that a passive order could plausibly capture. This sweep recomputes
    realized bid/ask fill labels for each horizon, recalibrates the side-selected
    fill probabilities, and reports drift versus the shortest horizon so research
    review can spot alpha that only looks tradable after delayed fills.
    """
    if isinstance(horizons, (str, bytes)):
        raise ValueError("horizons must be a non-empty sequence of positive integers")
    horizons = list(horizons)
    if not horizons:
        raise ValueError("horizons must be a non-empty sequence")
    for horizon in horizons:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon values must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizon values must be unique")
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    rows: list[dict[str, float | int | str]] = []
    anchor_realized_rate: float | None = None
    anchor_brier: float | None = None
    for index, horizon in enumerate(sorted(horizons)):
        realized = add_queue_position_realized_fill_proxy(
            frame,
            horizon=horizon,
            group_cols=group_cols,
            bid_realized_col="_sweep_bid_realized_fill",
            ask_realized_col="_sweep_ask_realized_fill",
        )
        curve = passive_fill_calibration_curve(
            realized,
            bins=bins,
            side_col=side_col,
            regime_col=regime_col,
            bid_realized_col="_sweep_bid_realized_fill",
            ask_realized_col="_sweep_ask_realized_fill",
        )
        summary = passive_fill_calibration_summary(curve)
        realized_rate = float(summary["weighted_realized_fill_rate"])
        brier = float(summary["weighted_brier_score"])
        if index == 0:
            anchor_realized_rate = realized_rate
            anchor_brier = brier
        assert anchor_realized_rate is not None
        assert anchor_brier is not None
        realized_gap = realized_rate - anchor_realized_rate
        brier_gap = brier - anchor_brier
        rows.append(
            {
                "horizon": int(horizon),
                "rows": int(summary["rows"]),
                "bins": int(summary["bins"]),
                "regimes": int(summary["regimes"]),
                "weighted_mean_predicted_fill_probability": float(
                    summary["weighted_mean_predicted_fill_probability"]
                ),
                "weighted_realized_fill_rate": realized_rate,
                "weighted_calibration_error": float(summary["weighted_calibration_error"]),
                "expected_calibration_error": float(summary["expected_calibration_error"]),
                "weighted_brier_score": brier,
                "worst_absolute_calibration_error": float(
                    summary["worst_absolute_calibration_error"]
                ),
                "realized_fill_rate_gap_vs_shortest": float(realized_gap),
                "brier_score_gap_vs_shortest": float(brier_gap),
                "horizon_stability_label": _passive_fill_horizon_stability_label(
                    index=index,
                    realized_gap=realized_gap,
                    brier_gap=brier_gap,
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_passive_fill_realization_horizon_sweep().columns)]


def add_passive_fill_event_window_regimes(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.75,
    window: int = 3,
    side_col: str = "best_execution_side",
    group_cols: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Label rows by proximity to high-probability passive-fill execution events.

    Sparse passive-fill event diagnostics are useful for review, but evaluation and
    demo artifacts also need row-level strata: pre-event rows show queue pressure
    buildup, event rows capture the tradable passive-fill opportunity, and
    post-event rows expose immediate adverse-selection aftermath. The event side,
    side-specific fill probability, and adverse-fill probability are copied from
    the nearest triggering event so downstream slices can attribute toxicity to
    the executable passive side. ``group_cols`` isolates independent symbols,
    venues, dates, or sessions in panel data.
    """
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValueError("window must be an integer")
    if window < 1:
        raise ValueError("window must be at least 1")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    grouping_columns = _normalize_group_columns(
        frame, group_cols, "passive fill event window regime group"
    )
    _require_columns(frame, required, "passive fill event window regime")

    output = frame.copy()
    distance = pd.Series(len(frame) + window, index=frame.index, dtype=int)
    event_side = pd.Series("none", index=frame.index, dtype=object)
    event_fill_out = pd.Series(0.0, index=frame.index, dtype=float)
    event_toxicity_out = pd.Series(0.0, index=frame.index, dtype=float)
    if frame.empty:
        output["passive_fill_event_distance"] = distance
        output["passive_fill_event_window_position"] = pd.Series(dtype=float)
        output["passive_fill_event_window_regime"] = pd.Series(dtype=object)
        output["passive_fill_event_side"] = event_side
        output["passive_fill_event_fill_probability"] = event_fill_out
        output["passive_fill_event_toxicity_probability"] = event_toxicity_out
        return output

    values = _finite_values(
        frame,
        [
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ],
        "passive fill event window regime",
    ).reset_index(drop=True)
    side = frame[side_col].astype(str).reset_index(drop=True)
    fill_probability = pd.Series(
        np.select(
            [side == "long", side == "short"],
            [values["bid_fill_probability"], values["ask_fill_probability"]],
            default=0.0,
        ),
        index=range(len(frame)),
    )
    adverse_probability = pd.Series(
        np.select(
            [side == "long", side == "short"],
            [values["bid_adverse_fill_probability"], values["ask_adverse_fill_probability"]],
            default=0.0,
        ),
        index=range(len(frame)),
    )
    if grouping_columns:
        grouped_positions = (
            frame[grouping_columns]
            .reset_index(drop=True)
            .groupby(grouping_columns, sort=False, dropna=False)
            .indices
            .values()
        )
    else:
        grouped_positions = [np.arange(len(frame))]

    event_mask = (side.isin(["long", "short"])) & (fill_probability >= threshold)
    for positions in grouped_positions:
        positions_array = np.asarray(positions, dtype=int)
        event_positions = positions_array[event_mask.iloc[positions_array].to_numpy(dtype=bool)]
        if len(event_positions) == 0:
            continue
        for position in positions_array:
            deltas = position - event_positions
            abs_deltas = np.abs(deltas)
            nearest_distance = int(abs_deltas.min())
            if nearest_distance > window:
                continue
            tied = event_positions[abs_deltas == nearest_distance]
            previous_tied = tied[tied <= position]
            if len(previous_tied):
                nearest_event = int(previous_tied.max())
            else:
                nearest_event = int(tied.min())
            original_label = frame.index[position]
            distance.loc[original_label] = int(position - nearest_event)
            event_side.loc[original_label] = str(side.iloc[nearest_event])
            event_fill_out.loc[original_label] = float(fill_probability.iloc[nearest_event])
            event_toxicity_out.loc[original_label] = float(adverse_probability.iloc[nearest_event])

    distance_values = distance.to_numpy(dtype=int)
    regimes = np.select(
        [
            distance_values == 0,
            (distance_values < 0) & (np.abs(distance_values) <= window),
            (distance_values > 0) & (distance_values <= window),
        ],
        ["event", "pre_event", "post_event"],
        default="calm",
    )
    output["passive_fill_event_distance"] = distance.astype(int)
    output["passive_fill_event_window_position"] = distance.astype(float) / float(window)
    output["passive_fill_event_window_regime"] = regimes.astype(str)
    output["passive_fill_event_side"] = event_side.astype(str)
    output["passive_fill_event_fill_probability"] = event_fill_out.astype(float)
    output["passive_fill_event_toxicity_probability"] = event_toxicity_out.astype(float)
    return output


def passive_fill_event_window_regime_summary(
    frame: pd.DataFrame,
    *,
    regime_col: str = "passive_fill_event_window_regime",
    event_side_col: str = "passive_fill_event_side",
    fill_probability_col: str = "passive_fill_event_fill_probability",
    toxicity_probability_col: str = "passive_fill_event_toxicity_probability",
    edge_col: str = "execution_adjusted_edge_ticks",
) -> pd.DataFrame:
    """Summarize execution quality by passive-fill event-window regime.

    The row-level event-window labels isolate buildup, executable fill rows, and
    immediate aftermath around high-probability passive fills. This summary ranks
    those neighborhoods by toxicity and realized execution drag so review packets
    can spot where passive fills are likely available but economically fragile.
    """
    columns = [
        "passive_fill_event_window_regime",
        "rows",
        "event_rows",
        "row_share",
        "mean_passive_fill_event_fill_probability",
        "mean_passive_fill_event_toxicity_probability",
        "mean_execution_adjusted_edge_ticks",
        "negative_edge_share",
        "dominant_passive_fill_event_side",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {
        regime_col,
        event_side_col,
        fill_probability_col,
        toxicity_probability_col,
        edge_col,
    }
    _require_columns(frame, required, "passive fill event window regime summary")
    data = frame[
        [regime_col, event_side_col, fill_probability_col, toxicity_probability_col, edge_col]
    ].copy()
    values = _finite_values(
        data,
        [fill_probability_col, toxicity_probability_col, edge_col],
        "passive fill event window regime summary",
    )
    data[fill_probability_col] = values[fill_probability_col]
    data[toxicity_probability_col] = values[toxicity_probability_col]
    data[edge_col] = values[edge_col]

    total_rows = float(len(data))
    rows = []
    for regime, group in data.groupby(regime_col, sort=True):
        side_counts = group[event_side_col].astype(str).value_counts()
        if "none" in side_counts.index and len(side_counts) > 1:
            side_counts = side_counts.drop(index="none")
        dominant_side = "none" if side_counts.empty else str(side_counts.idxmax())
        rows.append(
            {
                "passive_fill_event_window_regime": str(regime),
                "rows": int(len(group)),
                "event_rows": int((group[regime_col].astype(str) == "event").sum()),
                "row_share": float(len(group)) / total_rows if total_rows else 0.0,
                "mean_passive_fill_event_fill_probability": float(group[fill_probability_col].mean()),
                "mean_passive_fill_event_toxicity_probability": float(
                    group[toxicity_probability_col].mean()
                ),
                "mean_execution_adjusted_edge_ticks": float(group[edge_col].mean()),
                "negative_edge_share": float((group[edge_col] < 0.0).sum()) / float(len(group)),
                "dominant_passive_fill_event_side": dominant_side,
            }
        )
    return (
        pd.DataFrame(rows)[columns]
        .sort_values(
            [
                "mean_passive_fill_event_toxicity_probability",
                "negative_edge_share",
                "event_rows",
                "mean_execution_adjusted_edge_ticks",
                "rows",
            ],
            ascending=[False, False, False, True, False],
        )
        .reset_index(drop=True)
    )



def passive_fill_event_window_diagnostics(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.75,
    window: int = 3,
    side_col: str = "best_execution_side",
    long_return_col: str = "long_net_return_ticks",
    short_return_col: str = "short_net_return_ticks",
    regime_col: str | None = None,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Measure realized edge drift around high-probability passive-fill events.

    The fill model can look attractive exactly when adverse-selection risk is
    highest. This diagnostic isolates rows where the side-specific passive-fill
    probability breaches ``threshold`` and compares realized, side-consistent
    edge before and after the event. Grouping by an optional regime column turns
    the output into an event-window regime table for publishability review.
    ``group_cols`` prevents pre/post windows from leaking across symbols,
    venues, dates, or sessions in batched order-book panels.
    """
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValueError("window must be an integer")
    if window < 1:
        raise ValueError("window must be at least 1")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "execution_adjusted_edge_ticks",
        long_return_col,
        short_return_col,
    }
    if regime_col is not None:
        required.add(regime_col)
    grouping_columns = _normalize_group_columns(
        frame, group_cols, "passive fill event window group"
    )
    _require_columns(frame, required, "passive fill event window")
    if frame.empty:
        return _empty_passive_fill_event_windows()

    numeric_columns = [
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "execution_adjusted_edge_ticks",
        long_return_col,
        short_return_col,
    ]
    values = _finite_values(frame, numeric_columns, "passive fill event window")
    side = frame[side_col].astype(str).reset_index(drop=True)
    original_index = pd.Series(frame.index, index=range(len(frame)))
    event_fill = pd.Series(
        np.select(
            [side == "long", side == "short"],
            [values["bid_fill_probability"], values["ask_fill_probability"]],
            default=0.0,
        ),
        index=range(len(frame)),
    )
    event_adverse = pd.Series(
        np.select(
            [side == "long", side == "short"],
            [values["bid_adverse_fill_probability"], values["ask_adverse_fill_probability"]],
            default=0.0,
        ),
        index=range(len(frame)),
    )
    event_edge = values["execution_adjusted_edge_ticks"].reset_index(drop=True)
    long_returns = values[long_return_col].reset_index(drop=True)
    short_returns = values[short_return_col].reset_index(drop=True)
    regimes = (
        frame[regime_col].astype(str).reset_index(drop=True)
        if regime_col is not None
        else pd.Series("all", index=range(len(frame)))
    )
    if grouping_columns:
        group_keys = pd.Series(
            list(frame[grouping_columns].astype(str).itertuples(index=False, name=None)),
            index=range(len(frame)),
        )
    else:
        group_keys = pd.Series("__all__", index=range(len(frame)))

    event_positions = event_fill.index[(side.isin(["long", "short"])) & (event_fill >= threshold)]
    rows: list[dict[str, float | int | str]] = []
    for position in event_positions:
        event_side = side.iloc[position]
        realized = long_returns if event_side == "long" else short_returns
        event_group = group_keys.iloc[int(position)]
        pre_candidates = range(max(0, int(position) - window), int(position))
        post_candidates = range(int(position) + 1, min(len(frame), int(position) + window + 1))
        pre_positions = [row for row in pre_candidates if group_keys.iloc[row] == event_group]
        post_positions = [row for row in post_candidates if group_keys.iloc[row] == event_group]
        pre = realized.iloc[pre_positions]
        post = realized.iloc[post_positions]
        pre_sum = float(pre.sum()) if len(pre) else 0.0
        post_sum = float(post.sum()) if len(post) else 0.0
        event_regime = str(regimes.iloc[position])
        pre_regime = _modal_window_regime(regimes.iloc[pre_positions], fallback=event_regime)
        post_regime = _modal_window_regime(regimes.iloc[post_positions], fallback=event_regime)
        rows.append(
            {
                "event_index": original_index.iloc[position],
                "event_side": str(event_side),
                "event_regime": event_regime,
                "pre_window_regime": pre_regime,
                "post_window_regime": post_regime,
                "regime_transition": f"{pre_regime}->{post_regime}",
                "event_fill_probability": float(event_fill.iloc[position]),
                "event_adverse_fill_probability": float(event_adverse.iloc[position]),
                "event_edge_ticks": float(event_edge.iloc[position]),
                "pre_realized_edge_sum": pre_sum,
                "post_realized_edge_sum": post_sum,
                "post_minus_pre_realized_edge": float(post_sum - pre_sum),
                "window_rows": int(len(pre) + len(post) + 1),
            }
        )
    if not rows:
        return _empty_passive_fill_event_windows()
    return pd.DataFrame(rows)[list(_empty_passive_fill_event_windows().columns)]


def passive_fill_event_lead_lag_profile(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.75,
    window: int = 3,
    side_col: str = "best_execution_side",
    long_return_col: str = "long_net_return_ticks",
    short_return_col: str = "short_net_return_ticks",
    regime_col: str | None = None,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Profile side-consistent realized edge at each offset around fill events.

    Aggregate event-window sums can hide whether toxicity arrives before the fill,
    at the fill snapshot, or only after queue priority is achieved. This profile
    keeps relative offsets explicit and groups by the event regime so execution
    reviews can identify lead/lag adverse-selection structure rather than only a
    pre-vs-post total. ``group_cols`` bounds offsets within symbols, venues, dates,
    or sessions for panel-safe event-window analysis.
    """
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValueError("window must be an integer")
    if window < 1:
        raise ValueError("window must be at least 1")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        long_return_col,
        short_return_col,
    }
    if regime_col is not None:
        required.add(regime_col)
    grouping_columns = _normalize_group_columns(
        frame, group_cols, "passive fill event lead lag profile group"
    )
    _require_columns(frame, required, "passive fill event lead lag profile")

    columns = [
        "event_regime",
        "relative_offset",
        "observations",
        "mean_realized_edge_ticks",
        "adverse_realized_edge_share",
        "cumulative_mean_realized_edge_ticks",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    values = _finite_values(
        frame,
        [
            "bid_fill_probability",
            "ask_fill_probability",
            long_return_col,
            short_return_col,
        ],
        "passive fill event lead lag profile",
    )
    side = frame[side_col].astype(str).reset_index(drop=True)
    long_returns = values[long_return_col].reset_index(drop=True)
    short_returns = values[short_return_col].reset_index(drop=True)
    regimes = (
        frame[regime_col].astype(str).reset_index(drop=True)
        if regime_col is not None
        else pd.Series("all", index=range(len(frame)))
    )
    if grouping_columns:
        group_keys = pd.Series(
            list(frame[grouping_columns].astype(str).itertuples(index=False, name=None)),
            index=range(len(frame)),
        )
    else:
        group_keys = pd.Series("__all__", index=range(len(frame)))
    event_fill = pd.Series(
        np.select(
            [side == "long", side == "short"],
            [values["bid_fill_probability"], values["ask_fill_probability"]],
            default=0.0,
        ),
        index=range(len(frame)),
    )
    event_positions = event_fill.index[(side.isin(["long", "short"])) & (event_fill >= threshold)]

    observations: list[dict[str, float | int | str]] = []
    for position in event_positions:
        position_int = int(position)
        event_side = side.iloc[position_int]
        realized = long_returns if event_side == "long" else short_returns
        event_regime = str(regimes.iloc[position_int])
        event_group = group_keys.iloc[position_int]
        for relative_offset in range(-window, window + 1):
            row_position = position_int + relative_offset
            if 0 <= row_position < len(frame) and group_keys.iloc[row_position] == event_group:
                observations.append(
                    {
                        "event_regime": event_regime,
                        "relative_offset": int(relative_offset),
                        "realized_edge_ticks": float(realized.iloc[row_position]),
                    }
                )
    if not observations:
        return pd.DataFrame(columns=columns)

    data = pd.DataFrame(observations)
    rows: list[dict[str, float | int | str]] = []
    for (event_regime, relative_offset), group in data.groupby(
        ["event_regime", "relative_offset"], sort=True
    ):
        edge = group["realized_edge_ticks"]
        rows.append(
            {
                "event_regime": str(event_regime),
                "relative_offset": int(relative_offset),
                "observations": int(len(group)),
                "mean_realized_edge_ticks": float(edge.mean()),
                "adverse_realized_edge_share": float((edge < 0.0).mean()),
            }
        )
    profile = pd.DataFrame(rows).sort_values(
        ["event_regime", "relative_offset"], ignore_index=True
    )
    profile["cumulative_mean_realized_edge_ticks"] = profile.groupby(
        "event_regime", sort=False
    )["mean_realized_edge_ticks"].cumsum()
    return profile[columns]


def passive_fill_event_lead_lag_scorecard(profile: pd.DataFrame) -> pd.DataFrame:
    """Condense lead/lag fill-event profiles into regime-level toxicity warnings.

    High fill probabilities are only publishable when the realized edge survives
    after queue priority is achieved. This scorecard flags regimes where the
    pre-fill mean edge is positive but post-fill edge turns negative, a classic
    passive-execution adverse-selection inversion.
    """
    columns = [
        "event_regime",
        "offset_observations",
        "min_offset_observations",
        "pre_cumulative_mean_edge_ticks",
        "event_mean_edge_ticks",
        "post_cumulative_mean_edge_ticks",
        "post_adverse_realized_edge_share",
        "lead_lag_decay_ticks",
        "toxicity_inversion",
        "warning_label",
    ]
    required = {
        "event_regime",
        "relative_offset",
        "observations",
        "mean_realized_edge_ticks",
        "adverse_realized_edge_share",
        "cumulative_mean_realized_edge_ticks",
    }
    _require_columns(profile, required, "passive fill event lead lag scorecard")
    if profile.empty:
        return pd.DataFrame(columns=columns)

    values = _finite_values(
        profile,
        [
            "relative_offset",
            "observations",
            "mean_realized_edge_ticks",
            "adverse_realized_edge_share",
            "cumulative_mean_realized_edge_ticks",
        ],
        "passive fill event lead lag scorecard",
    )
    if (values["observations"] < 0.0).any():
        raise ValueError("passive fill event lead lag scorecard observations must be non-negative")
    if (
        (values["adverse_realized_edge_share"] < 0.0)
        | (values["adverse_realized_edge_share"] > 1.0)
    ).any():
        raise ValueError(
            "passive fill event lead lag scorecard adverse shares must be between 0 and 1"
        )

    data = values.copy()
    data["event_regime"] = profile["event_regime"].astype(str)
    rows: list[dict[str, bool | float | int | str]] = []
    for regime, group in data.groupby("event_regime", sort=True):
        pre = group[group["relative_offset"] < 0.0]
        event = group[group["relative_offset"] == 0.0]
        post = group[group["relative_offset"] > 0.0]
        offset_observations = int(group["observations"].sum())
        min_offset_observations = int(group["observations"].min()) if len(group) else 0
        pre_edge = float(pre["mean_realized_edge_ticks"].sum()) if len(pre) else 0.0
        event_edge = float(event["mean_realized_edge_ticks"].mean()) if len(event) else 0.0
        post_edge = float(post["mean_realized_edge_ticks"].sum()) if len(post) else 0.0
        post_adverse_share = (
            float(np.average(post["adverse_realized_edge_share"], weights=post["observations"]))
            if len(post) and float(post["observations"].sum()) > 0.0
            else 0.0
        )
        toxicity_inversion = bool(pre_edge > 0.0 and post_edge < 0.0)
        if toxicity_inversion:
            warning_label = "toxic_reversal"
        elif post_edge < 0.0:
            warning_label = "post_fill_adverse"
        elif not len(pre) or not len(post):
            warning_label = "insufficient_pre_post"
        else:
            warning_label = "edge_persistent"
        rows.append(
            {
                "event_regime": str(regime),
                "offset_observations": offset_observations,
                "min_offset_observations": min_offset_observations,
                "pre_cumulative_mean_edge_ticks": pre_edge,
                "event_mean_edge_ticks": event_edge,
                "post_cumulative_mean_edge_ticks": post_edge,
                "post_adverse_realized_edge_share": post_adverse_share,
                "lead_lag_decay_ticks": float(post_edge - pre_edge),
                "toxicity_inversion": toxicity_inversion,
                "warning_label": warning_label,
            }
        )
    scorecard = pd.DataFrame(rows)[columns]
    return scorecard.sort_values(
        ["toxicity_inversion", "post_cumulative_mean_edge_ticks", "lead_lag_decay_ticks"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def passive_fill_event_regime_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passive-fill event windows by execution regime."""
    columns = list(_empty_passive_fill_event_regime_summary().columns)
    if events.empty:
        return _empty_passive_fill_event_regime_summary()
    required = {
        "event_regime",
        "event_fill_probability",
        "event_adverse_fill_probability",
        "event_edge_ticks",
        "post_minus_pre_realized_edge",
    }
    _require_columns(events, required, "passive fill event regime summary")
    values = _finite_values(
        events,
        [
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "post_minus_pre_realized_edge",
        ],
        "passive fill event regime summary",
    )
    data = values.copy()
    data["event_regime"] = events["event_regime"].astype(str)

    rows: list[dict[str, float | int | str]] = []
    for regime, group in data.groupby("event_regime", sort=True):
        drift = group["post_minus_pre_realized_edge"]
        adverse = int((drift < 0.0).sum())
        rows.append(
            {
                "event_regime": str(regime),
                "events": int(len(group)),
                "adverse_post_edge_events": adverse,
                "adverse_post_edge_share": float(adverse / len(group)),
                "mean_event_fill_probability": float(group["event_fill_probability"].mean()),
                "mean_event_adverse_fill_probability": float(
                    group["event_adverse_fill_probability"].mean()
                ),
                "mean_event_edge_ticks": float(group["event_edge_ticks"].mean()),
                "mean_post_minus_pre_realized_edge": float(drift.mean()),
                "worst_post_minus_pre_realized_edge": float(drift.min()),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values(
        ["adverse_post_edge_share", "worst_post_minus_pre_realized_edge"],
        ascending=[False, True],
        ignore_index=True,
    )


def passive_fill_event_transition_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passive-fill event toxicity by pre/post regime transition.

    Event-window failures often appear at liquidity-state boundaries rather than
    inside a single named regime. This table groups high-probability passive-fill
    events by the modal pre-window to post-window regime path so reviewers can see
    whether queue fills become toxic specifically during calm→thin or thin→stress
    transitions.
    """
    columns = list(_empty_passive_fill_event_transition_summary().columns)
    if events.empty:
        return _empty_passive_fill_event_transition_summary()
    required = {
        "event_regime",
        "regime_transition",
        "event_fill_probability",
        "event_adverse_fill_probability",
        "event_edge_ticks",
        "post_minus_pre_realized_edge",
    }
    _require_columns(events, required, "passive fill event transition summary")
    values = _finite_values(
        events,
        [
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "post_minus_pre_realized_edge",
        ],
        "passive fill event transition summary",
    )
    data = values.copy()
    data["event_regime"] = events["event_regime"].astype(str)
    data["regime_transition"] = events["regime_transition"].astype(str)

    rows: list[dict[str, float | int | str]] = []
    for transition, group in data.groupby("regime_transition", sort=True):
        drift = group["post_minus_pre_realized_edge"]
        adverse = int((drift < 0.0).sum())
        rows.append(
            {
                "regime_transition": str(transition),
                "event_regimes": int(group["event_regime"].nunique()),
                "events": int(len(group)),
                "adverse_post_edge_events": adverse,
                "adverse_post_edge_share": float(adverse / len(group)),
                "mean_event_fill_probability": float(group["event_fill_probability"].mean()),
                "mean_event_adverse_fill_probability": float(
                    group["event_adverse_fill_probability"].mean()
                ),
                "mean_event_edge_ticks": float(group["event_edge_ticks"].mean()),
                "mean_post_minus_pre_realized_edge": float(drift.mean()),
                "worst_post_minus_pre_realized_edge": float(drift.min()),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values(
        [
            "adverse_post_edge_share",
            "worst_post_minus_pre_realized_edge",
            "mean_event_adverse_fill_probability",
        ],
        ascending=[False, True, False],
        ignore_index=True,
    )


def passive_fill_event_lifecycle_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passive-fill event toxicity by full pre/event/post regime path.

    Transition summaries collapse the event regime out of the state path. This
    lifecycle view keeps the modal pre-window regime, event-row regime, and modal
    post-window regime together so reviewers can spot toxic fills that are benign
    for the same pre→post transition in one event state but fragile in another.
    """
    columns = list(_empty_passive_fill_event_lifecycle_summary().columns)
    if events.empty:
        return _empty_passive_fill_event_lifecycle_summary()
    required = {
        "pre_window_regime",
        "event_regime",
        "post_window_regime",
        "regime_transition",
        "event_fill_probability",
        "event_adverse_fill_probability",
        "event_edge_ticks",
        "pre_realized_edge_sum",
        "post_realized_edge_sum",
        "post_minus_pre_realized_edge",
        "window_rows",
    }
    _require_columns(events, required, "passive fill event lifecycle summary")
    values = _finite_values(
        events,
        [
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "pre_realized_edge_sum",
            "post_realized_edge_sum",
            "post_minus_pre_realized_edge",
            "window_rows",
        ],
        "passive fill event lifecycle summary",
    )
    if (values["window_rows"] < 1.0).any():
        raise ValueError("passive fill event lifecycle summary window_rows must be positive")

    data = values.copy()
    data["pre_window_regime"] = events["pre_window_regime"].astype(str)
    data["event_regime"] = events["event_regime"].astype(str)
    data["post_window_regime"] = events["post_window_regime"].astype(str)
    data["regime_transition"] = events["regime_transition"].astype(str)
    data["lifecycle_path"] = (
        data["pre_window_regime"]
        + "|"
        + data["event_regime"]
        + "|"
        + data["post_window_regime"]
    )

    rows: list[dict[str, float | int | str]] = []
    for lifecycle_path, group in data.groupby("lifecycle_path", sort=True):
        drift = group["post_minus_pre_realized_edge"]
        adverse = int((drift < 0.0).sum())
        adverse_share = float(adverse / len(group))
        mean_drift = float(drift.mean())
        rows.append(
            {
                "lifecycle_path": str(lifecycle_path),
                "pre_window_regime": str(group["pre_window_regime"].iloc[0]),
                "event_regime": str(group["event_regime"].iloc[0]),
                "post_window_regime": str(group["post_window_regime"].iloc[0]),
                "regime_transitions": int(group["regime_transition"].nunique()),
                "events": int(len(group)),
                "adverse_post_edge_events": adverse,
                "adverse_post_edge_share": adverse_share,
                "mean_event_fill_probability": float(group["event_fill_probability"].mean()),
                "mean_event_adverse_fill_probability": float(
                    group["event_adverse_fill_probability"].mean()
                ),
                "mean_event_edge_ticks": float(group["event_edge_ticks"].mean()),
                "mean_pre_realized_edge_sum": float(group["pre_realized_edge_sum"].mean()),
                "mean_post_realized_edge_sum": float(group["post_realized_edge_sum"].mean()),
                "mean_post_minus_pre_realized_edge": mean_drift,
                "worst_post_minus_pre_realized_edge": float(drift.min()),
                "mean_window_rows": float(group["window_rows"].mean()),
                "lifecycle_toxicity_label": _passive_fill_lifecycle_toxicity_label(
                    adverse_share=adverse_share,
                    mean_post_minus_pre_edge=mean_drift,
                ),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values(
        [
            "adverse_post_edge_share",
            "worst_post_minus_pre_realized_edge",
            "mean_event_adverse_fill_probability",
        ],
        ascending=[False, True, False],
        ignore_index=True,
    )


def _passive_fill_lifecycle_toxicity_label(
    *, adverse_share: float, mean_post_minus_pre_edge: float
) -> str:
    if adverse_share >= 0.5 and mean_post_minus_pre_edge < 0.0:
        return "toxic_transition_lifecycle"
    if adverse_share > 0.0 or mean_post_minus_pre_edge < 0.0:
        return "mixed_transition_lifecycle"
    return "benign_transition_lifecycle"


def _modal_window_regime(values: pd.Series, *, fallback: str) -> str:
    if values.empty:
        return fallback
    counts = values.astype(str).value_counts(sort=True)
    return str(counts.index[0])


def passive_fill_event_toxicity_scorecard(
    regime_summary: pd.DataFrame,
    *,
    max_adverse_post_edge_share: float = 0.60,
    min_mean_post_minus_pre_edge: float = -0.25,
    min_events: int = 1,
) -> dict[str, float | int | str]:
    """Gate passive-fill event windows for execution-aware publishability.

    High fill probability is only useful if the subsequent event window is not
    systematically toxic. This scorecard consumes ``passive_fill_event_regime_summary``
    and flags regimes where high-probability fills are followed by excessive
    adverse edge drift or a strongly negative mean post-minus-pre realized edge.
    """
    if not math.isfinite(max_adverse_post_edge_share) or not 0.0 <= max_adverse_post_edge_share <= 1.0:
        raise ValueError("max_adverse_post_edge_share must be finite and between 0 and 1")
    if not math.isfinite(min_mean_post_minus_pre_edge):
        raise ValueError("min_mean_post_minus_pre_edge must be finite")
    if not isinstance(min_events, int) or isinstance(min_events, bool) or min_events < 1:
        raise ValueError("min_events must be a positive integer")
    if regime_summary.empty:
        return _empty_passive_fill_event_toxicity_scorecard()

    required = {
        "event_regime",
        "events",
        "adverse_post_edge_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_post_minus_pre_realized_edge",
        "worst_post_minus_pre_realized_edge",
    }
    _require_columns(regime_summary, required, "passive fill event toxicity")
    values = _finite_values(
        regime_summary,
        [
            "events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
        ],
        "passive fill event toxicity",
    )
    if (values["events"] < 0.0).any():
        raise ValueError("passive fill event toxicity events must be non-negative")

    total_events = int(values["events"].sum())
    if total_events == 0:
        return _empty_passive_fill_event_toxicity_scorecard()

    data = values.copy()
    data["event_regime"] = regime_summary["event_regime"].astype(str)
    weights = data["events"] / total_events
    eligible = data[data["events"] >= min_events]
    if eligible.empty:
        scorecard = _empty_passive_fill_event_toxicity_scorecard()
        scorecard.update(
            {
                "rows": int(len(regime_summary)),
                "regimes": int(data["event_regime"].nunique()),
                "total_events": total_events,
                "weighted_mean_event_fill_probability": float(
                    (data["mean_event_fill_probability"] * weights).sum()
                ),
                "weighted_mean_event_adverse_fill_probability": float(
                    (data["mean_event_adverse_fill_probability"] * weights).sum()
                ),
                "weighted_mean_post_minus_pre_realized_edge": float(
                    (data["mean_post_minus_pre_realized_edge"] * weights).sum()
                ),
                "event_toxicity_label": "insufficient_event_windows",
            }
        )
        return scorecard

    blocked = eligible[
        (eligible["adverse_post_edge_share"] > max_adverse_post_edge_share)
        | (eligible["mean_post_minus_pre_realized_edge"] < min_mean_post_minus_pre_edge)
    ]
    worst_idx = eligible.sort_values(
        ["adverse_post_edge_share", "mean_post_minus_pre_realized_edge"],
        ascending=[False, True],
    ).index[0]
    label = "event_window_blocker" if not blocked.empty else "event_window_pass"

    return {
        "rows": int(len(regime_summary)),
        "regimes": int(data["event_regime"].nunique()),
        "total_events": total_events,
        "eligible_regimes": int(len(eligible)),
        "blocked_regimes": int(len(blocked)),
        "worst_regime": str(data.loc[worst_idx, "event_regime"]),
        "worst_adverse_post_edge_share": float(data.loc[worst_idx, "adverse_post_edge_share"]),
        "worst_mean_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "mean_post_minus_pre_realized_edge"]
        ),
        "worst_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "worst_post_minus_pre_realized_edge"]
        ),
        "weighted_mean_event_fill_probability": float(
            (data["mean_event_fill_probability"] * weights).sum()
        ),
        "weighted_mean_event_adverse_fill_probability": float(
            (data["mean_event_adverse_fill_probability"] * weights).sum()
        ),
        "weighted_mean_post_minus_pre_realized_edge": float(
            (data["mean_post_minus_pre_realized_edge"] * weights).sum()
        ),
        "event_toxicity_label": label,
    }


def passive_fill_event_lifecycle_policy_curve(
    events: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90),
    max_adverse_post_edge_share: float = 0.60,
    min_mean_post_minus_pre_edge: float = -0.25,
) -> pd.DataFrame:
    """Sweep full lifecycle path cutoffs for passive-fill event policies.

    Transition-level policy curves can hide that the same pre→post move is benign
    in one event-row liquidity state and toxic in another. This surface keeps the
    pre-window, event-row, and post-window regimes together so execution reviewers
    can suppress only the fragile lifecycle paths rather than blocking an entire
    transition family.
    """
    columns = [
        "lifecycle_path",
        "pre_window_regime",
        "event_regime",
        "post_window_regime",
        "threshold",
        "total_events",
        "candidate_events",
        "event_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_event_edge_ticks",
        "mean_pre_realized_edge_sum",
        "mean_post_realized_edge_sum",
        "mean_post_minus_pre_realized_edge",
        "adverse_post_edge_share",
        "policy_label",
    ]
    if not thresholds:
        raise ValueError("thresholds must be a non-empty sequence")
    clean_thresholds = [float(threshold) for threshold in thresholds]
    if any(not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0 for threshold in clean_thresholds):
        raise ValueError("threshold values must be in [0.0, 1.0]")
    for name, value in {
        "max_adverse_post_edge_share": max_adverse_post_edge_share,
        "min_mean_post_minus_pre_edge": min_mean_post_minus_pre_edge,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_adverse_post_edge_share <= 1.0:
        raise ValueError("max_adverse_post_edge_share must be in [0.0, 1.0]")

    required = {
        "pre_window_regime",
        "event_regime",
        "post_window_regime",
        "event_fill_probability",
        "event_adverse_fill_probability",
        "event_edge_ticks",
        "pre_realized_edge_sum",
        "post_realized_edge_sum",
        "post_minus_pre_realized_edge",
    }
    if events.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(events, required, "passive fill event lifecycle policy")
    values = _finite_values(
        events,
        [
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "pre_realized_edge_sum",
            "post_realized_edge_sum",
            "post_minus_pre_realized_edge",
        ],
        "passive fill event lifecycle policy",
    )
    if (
        (values[["event_fill_probability", "event_adverse_fill_probability"]] < 0.0)
        | (values[["event_fill_probability", "event_adverse_fill_probability"]] > 1.0)
    ).any().any():
        raise ValueError("event fill probabilities must be in [0.0, 1.0]")

    state = values.copy()
    state["pre_window_regime"] = events["pre_window_regime"].astype(str)
    state["event_regime"] = events["event_regime"].astype(str)
    state["post_window_regime"] = events["post_window_regime"].astype(str)
    state["lifecycle_path"] = (
        state["pre_window_regime"]
        + "|"
        + state["event_regime"]
        + "|"
        + state["post_window_regime"]
    )

    rows: list[dict[str, float | int | str]] = []
    for lifecycle_path, group in state.groupby("lifecycle_path", sort=True):
        total_events = len(group)
        for threshold in sorted(clean_thresholds):
            candidate = group[group["event_fill_probability"] >= threshold]
            candidate_events = len(candidate)
            common = {
                "lifecycle_path": str(lifecycle_path),
                "pre_window_regime": str(group["pre_window_regime"].iloc[0]),
                "event_regime": str(group["event_regime"].iloc[0]),
                "post_window_regime": str(group["post_window_regime"].iloc[0]),
                "threshold": float(threshold),
                "total_events": int(total_events),
                "candidate_events": int(candidate_events),
                "event_share": float(candidate_events / total_events),
            }
            if candidate_events == 0:
                rows.append(
                    {
                        **common,
                        "mean_event_fill_probability": 0.0,
                        "mean_event_adverse_fill_probability": 0.0,
                        "mean_event_edge_ticks": 0.0,
                        "mean_pre_realized_edge_sum": 0.0,
                        "mean_post_realized_edge_sum": 0.0,
                        "mean_post_minus_pre_realized_edge": 0.0,
                        "adverse_post_edge_share": 0.0,
                        "policy_label": "no_lifecycle_policy_events",
                    }
                )
                continue
            adverse_share = float((candidate["post_minus_pre_realized_edge"] < 0.0).mean())
            mean_post_delta = float(candidate["post_minus_pre_realized_edge"].mean())
            if mean_post_delta < min_mean_post_minus_pre_edge:
                label = "lifecycle_policy_blocked"
            elif adverse_share > max_adverse_post_edge_share:
                label = "lifecycle_policy_review"
            elif threshold >= 0.80:
                label = "selective_lifecycle_policy"
            else:
                label = "broad_lifecycle_policy"
            rows.append(
                {
                    **common,
                    "mean_event_fill_probability": float(candidate["event_fill_probability"].mean()),
                    "mean_event_adverse_fill_probability": float(
                        candidate["event_adverse_fill_probability"].mean()
                    ),
                    "mean_event_edge_ticks": float(candidate["event_edge_ticks"].mean()),
                    "mean_pre_realized_edge_sum": float(candidate["pre_realized_edge_sum"].mean()),
                    "mean_post_realized_edge_sum": float(candidate["post_realized_edge_sum"].mean()),
                    "mean_post_minus_pre_realized_edge": mean_post_delta,
                    "adverse_post_edge_share": adverse_share,
                    "policy_label": label,
                }
            )
    return pd.DataFrame(rows, columns=columns)



def passive_fill_event_transition_policy_curve(
    events: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90),
    transition_col: str = "regime_transition",
    max_adverse_post_edge_share: float = 0.60,
    min_mean_post_minus_pre_edge: float = -0.25,
) -> pd.DataFrame:
    """Sweep transition-conditioned passive-fill cutoffs for event-window policies.

    High predicted passive-fill probability is not enough when fills cluster around
    toxic regime transitions. This artifact turns event-window diagnostics into an
    execution policy surface: for each regime transition and fill cutoff, quantify
    how many events remain tradable, whether post-window realized edge deteriorates,
    and whether the transition should pass, review, or block a passive policy.
    """
    columns = [
        transition_col,
        "threshold",
        "total_events",
        "candidate_events",
        "event_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_event_edge_ticks",
        "mean_post_minus_pre_realized_edge",
        "adverse_post_edge_share",
        "policy_label",
    ]
    if not thresholds:
        raise ValueError("thresholds must be a non-empty sequence")
    clean_thresholds = [float(threshold) for threshold in thresholds]
    if any(not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0 for threshold in clean_thresholds):
        raise ValueError("threshold values must be in [0.0, 1.0]")
    for name, value in {
        "max_adverse_post_edge_share": max_adverse_post_edge_share,
        "min_mean_post_minus_pre_edge": min_mean_post_minus_pre_edge,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_adverse_post_edge_share <= 1.0:
        raise ValueError("max_adverse_post_edge_share must be in [0.0, 1.0]")

    required = {
        transition_col,
        "event_fill_probability",
        "event_adverse_fill_probability",
        "event_edge_ticks",
        "post_minus_pre_realized_edge",
    }
    if events.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(events, required, "passive fill event transition policy")
    values = _finite_values(
        events,
        [
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "post_minus_pre_realized_edge",
        ],
        "passive fill event transition policy",
    )
    if (
        (values[["event_fill_probability", "event_adverse_fill_probability"]] < 0.0)
        | (values[["event_fill_probability", "event_adverse_fill_probability"]] > 1.0)
    ).any().any():
        raise ValueError("event fill probabilities must be in [0.0, 1.0]")

    state = pd.DataFrame(
        {
            transition_col: events[transition_col].astype(str),
            "event_fill_probability": values["event_fill_probability"],
            "event_adverse_fill_probability": values["event_adverse_fill_probability"],
            "event_edge_ticks": values["event_edge_ticks"],
            "post_minus_pre_realized_edge": values["post_minus_pre_realized_edge"],
        }
    )

    rows: list[dict[str, float | int | str]] = []
    for transition, group in state.groupby(transition_col, sort=True):
        total_events = len(group)
        for threshold in sorted(clean_thresholds):
            candidate = group[group["event_fill_probability"] >= threshold]
            candidate_events = len(candidate)
            if candidate_events == 0:
                rows.append(
                    {
                        transition_col: str(transition),
                        "threshold": float(threshold),
                        "total_events": int(total_events),
                        "candidate_events": 0,
                        "event_share": 0.0,
                        "mean_event_fill_probability": 0.0,
                        "mean_event_adverse_fill_probability": 0.0,
                        "mean_event_edge_ticks": 0.0,
                        "mean_post_minus_pre_realized_edge": 0.0,
                        "adverse_post_edge_share": 0.0,
                        "policy_label": "no_transition_policy_events",
                    }
                )
                continue
            adverse_share = float((candidate["post_minus_pre_realized_edge"] < 0.0).mean())
            mean_post_delta = float(candidate["post_minus_pre_realized_edge"].mean())
            if mean_post_delta < min_mean_post_minus_pre_edge:
                label = "transition_policy_blocked"
            elif adverse_share > max_adverse_post_edge_share:
                label = "transition_policy_review"
            elif threshold >= 0.80:
                label = "selective_transition_policy"
            else:
                label = "broad_transition_policy"
            rows.append(
                {
                    transition_col: str(transition),
                    "threshold": float(threshold),
                    "total_events": int(total_events),
                    "candidate_events": int(candidate_events),
                    "event_share": float(candidate_events / total_events),
                    "mean_event_fill_probability": float(candidate["event_fill_probability"].mean()),
                    "mean_event_adverse_fill_probability": float(
                        candidate["event_adverse_fill_probability"].mean()
                    ),
                    "mean_event_edge_ticks": float(candidate["event_edge_ticks"].mean()),
                    "mean_post_minus_pre_realized_edge": mean_post_delta,
                    "adverse_post_edge_share": adverse_share,
                    "policy_label": label,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def passive_fill_event_policy_stability(
    train_policy: pd.DataFrame,
    heldout_policy: pd.DataFrame,
    *,
    path_col: str = "lifecycle_path",
    threshold_col: str = "threshold",
) -> pd.DataFrame:
    """Compare event-window passive-fill policy curves across train/heldout splits.

    In-sample lifecycle or transition policies can look tradable because they find
    narrow high-fill cutoffs, then fail exactly where passive execution becomes
    toxic out of sample. This stability table joins policy curves by path and
    threshold, measures candidate retention and toxicity deltas, and labels rows
    where a train-approved passive policy becomes blocked, review-only, or empty
    on heldout data.
    """
    columns = [
        path_col,
        threshold_col,
        "train_total_events",
        "heldout_total_events",
        "train_candidate_events",
        "heldout_candidate_events",
        "candidate_event_retention",
        "train_event_share",
        "heldout_event_share",
        "event_share_delta",
        "train_mean_event_fill_probability",
        "heldout_mean_event_fill_probability",
        "mean_event_fill_probability_delta",
        "train_mean_event_adverse_fill_probability",
        "heldout_mean_event_adverse_fill_probability",
        "mean_event_adverse_fill_probability_delta",
        "train_mean_post_minus_pre_realized_edge",
        "heldout_mean_post_minus_pre_realized_edge",
        "mean_post_minus_pre_realized_edge_delta",
        "train_adverse_post_edge_share",
        "heldout_adverse_post_edge_share",
        "adverse_post_edge_share_delta",
        "train_policy_label",
        "heldout_policy_label",
        "heldout_stability_label",
    ]
    required = {
        path_col,
        threshold_col,
        "total_events",
        "candidate_events",
        "event_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_post_minus_pre_realized_edge",
        "adverse_post_edge_share",
        "policy_label",
    }
    _require_columns(train_policy, required, "passive fill event policy stability train")
    _require_columns(heldout_policy, required, "passive fill event policy stability heldout")
    if train_policy.empty:
        return pd.DataFrame(columns=columns)

    numeric_columns = [
        threshold_col,
        "total_events",
        "candidate_events",
        "event_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_post_minus_pre_realized_edge",
        "adverse_post_edge_share",
    ]
    train_values = _finite_values(train_policy, numeric_columns, "passive fill event policy stability train")
    heldout_values = _finite_values(
        heldout_policy, numeric_columns, "passive fill event policy stability heldout"
    )
    for label, values in {"train": train_values, "heldout": heldout_values}.items():
        if (values[["total_events", "candidate_events"]] < 0.0).any().any():
            raise ValueError(f"passive fill event policy stability {label} counts must be non-negative")
        if (values["candidate_events"] > values["total_events"]).any():
            raise ValueError(
                f"passive fill event policy stability {label} candidates exceed total events"
            )
        probability_columns = [
            "event_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "adverse_post_edge_share",
        ]
        if not values[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
            raise ValueError(
                f"passive fill event policy stability {label} probabilities must be in [0.0, 1.0]"
            )

    train = train_values.copy()
    train[path_col] = train_policy[path_col].astype(str)
    train["policy_label"] = train_policy["policy_label"].astype(str)
    heldout = heldout_values.copy()
    heldout[path_col] = heldout_policy[path_col].astype(str)
    heldout["policy_label"] = heldout_policy["policy_label"].astype(str)

    merged = train.merge(
        heldout,
        on=[path_col, threshold_col],
        how="left",
        suffixes=("_train", "_heldout"),
    )
    rows: list[dict[str, float | int | str]] = []
    for _, row in merged.iterrows():
        heldout_missing = pd.isna(row.get("total_events_heldout"))
        train_candidates = float(row["candidate_events_train"])
        heldout_candidates = 0.0 if heldout_missing else float(row["candidate_events_heldout"])
        heldout_label = "missing_heldout_policy_path" if heldout_missing else str(row["policy_label_heldout"])
        stability_label = _passive_fill_event_policy_stability_label(heldout_label)
        train_edge = float(row["mean_post_minus_pre_realized_edge_train"])
        heldout_edge = 0.0 if heldout_missing else float(row["mean_post_minus_pre_realized_edge_heldout"])
        train_adverse = float(row["adverse_post_edge_share_train"])
        heldout_adverse = 1.0 if heldout_missing else float(row["adverse_post_edge_share_heldout"])
        rows.append(
            {
                path_col: str(row[path_col]),
                threshold_col: float(row[threshold_col]),
                "train_total_events": int(row["total_events_train"]),
                "heldout_total_events": 0 if heldout_missing else int(row["total_events_heldout"]),
                "train_candidate_events": int(row["candidate_events_train"]),
                "heldout_candidate_events": int(heldout_candidates),
                "candidate_event_retention": float(
                    heldout_candidates / train_candidates if train_candidates else 0.0
                ),
                "train_event_share": float(row["event_share_train"]),
                "heldout_event_share": 0.0 if heldout_missing else float(row["event_share_heldout"]),
                "event_share_delta": float(
                    (0.0 if heldout_missing else row["event_share_heldout"]) - row["event_share_train"]
                ),
                "train_mean_event_fill_probability": float(row["mean_event_fill_probability_train"]),
                "heldout_mean_event_fill_probability": 0.0
                if heldout_missing
                else float(row["mean_event_fill_probability_heldout"]),
                "mean_event_fill_probability_delta": float(
                    (0.0 if heldout_missing else row["mean_event_fill_probability_heldout"])
                    - row["mean_event_fill_probability_train"]
                ),
                "train_mean_event_adverse_fill_probability": float(
                    row["mean_event_adverse_fill_probability_train"]
                ),
                "heldout_mean_event_adverse_fill_probability": 1.0
                if heldout_missing
                else float(row["mean_event_adverse_fill_probability_heldout"]),
                "mean_event_adverse_fill_probability_delta": float(
                    (1.0 if heldout_missing else row["mean_event_adverse_fill_probability_heldout"])
                    - row["mean_event_adverse_fill_probability_train"]
                ),
                "train_mean_post_minus_pre_realized_edge": train_edge,
                "heldout_mean_post_minus_pre_realized_edge": heldout_edge,
                "mean_post_minus_pre_realized_edge_delta": float(heldout_edge - train_edge),
                "train_adverse_post_edge_share": train_adverse,
                "heldout_adverse_post_edge_share": heldout_adverse,
                "adverse_post_edge_share_delta": float(heldout_adverse - train_adverse),
                "train_policy_label": str(row["policy_label_train"]),
                "heldout_policy_label": heldout_label,
                "heldout_stability_label": stability_label,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    priority = result["heldout_stability_label"].map(
        {
            "heldout_policy_blocker": 4,
            "heldout_policy_review": 3,
            "heldout_policy_no_events": 2,
            "heldout_policy_missing": 2,
            "heldout_policy_stable": 1,
        }
    )
    return (
        result.assign(_priority=priority.fillna(0))
        .sort_values(
            [
                "_priority",
                "adverse_post_edge_share_delta",
                "mean_post_minus_pre_realized_edge_delta",
                path_col,
                threshold_col,
            ],
            ascending=[False, False, True, True, True],
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def _passive_fill_event_policy_stability_label(policy_label: str) -> str:
    if policy_label == "missing_heldout_policy_path":
        return "heldout_policy_missing"
    if "blocked" in policy_label:
        return "heldout_policy_blocker"
    if "review" in policy_label:
        return "heldout_policy_review"
    if policy_label.startswith("no_"):
        return "heldout_policy_no_events"
    return "heldout_policy_stable"


def _empty_passive_fill_event_policy_stability_scorecard() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "policy_paths": 0,
        "total_train_candidate_events": 0,
        "total_heldout_candidate_events": 0,
        "candidate_event_retention": 0.0,
        "blocker_rows": 0,
        "review_rows": 0,
        "no_event_rows": 0,
        "missing_rows": 0,
        "blocker_train_candidate_share": 0.0,
        "review_train_candidate_share": 0.0,
        "no_event_train_candidate_share": 0.0,
        "missing_train_candidate_share": 0.0,
        "weighted_mean_post_minus_pre_realized_edge_delta": 0.0,
        "weighted_adverse_post_edge_share_delta": 0.0,
        "worst_policy_path": "none",
        "worst_threshold": 0.0,
        "worst_heldout_stability_label": "none",
        "policy_stability_decision": "review",
        "policy_stability_label": "insufficient_passive_fill_policy_stability_evidence",
        "blocking_reasons": "none",
        "review_reasons": "empty_stability_table",
    }


def passive_fill_event_policy_stability_scorecard(
    stability: pd.DataFrame,
    *,
    path_col: str = "lifecycle_path",
    threshold_col: str = "threshold",
    max_blocker_candidate_share: float = 0.10,
    max_review_candidate_share: float = 0.25,
    min_weighted_edge_delta: float = -0.25,
    max_weighted_adverse_delta: float = 0.20,
) -> dict[str, float | int | str]:
    """Reduce train/heldout passive-fill policy stability into a release gate.

    ``passive_fill_event_policy_stability`` is row-level evidence. This scorecard
    converts it into a candidate-weighted publishability decision so a policy that
    only works in sample cannot pass merely because its toxic heldout rows are
    numerically few but carry most of the train candidate capacity.
    """
    for name, value in {
        "max_blocker_candidate_share": max_blocker_candidate_share,
        "max_review_candidate_share": max_review_candidate_share,
        "min_weighted_edge_delta": min_weighted_edge_delta,
        "max_weighted_adverse_delta": max_weighted_adverse_delta,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_blocker_candidate_share <= 1.0:
        raise ValueError("max_blocker_candidate_share must be in [0.0, 1.0]")
    if not 0.0 <= max_review_candidate_share <= 1.0:
        raise ValueError("max_review_candidate_share must be in [0.0, 1.0]")
    if max_weighted_adverse_delta < 0.0:
        raise ValueError("max_weighted_adverse_delta must be non-negative")

    if stability.empty:
        return _empty_passive_fill_event_policy_stability_scorecard()

    required = {
        path_col,
        threshold_col,
        "train_candidate_events",
        "heldout_candidate_events",
        "mean_post_minus_pre_realized_edge_delta",
        "adverse_post_edge_share_delta",
        "heldout_stability_label",
    }
    _require_columns(stability, required, "passive fill event policy stability scorecard")
    values = _finite_values(
        stability,
        [
            threshold_col,
            "train_candidate_events",
            "heldout_candidate_events",
            "mean_post_minus_pre_realized_edge_delta",
            "adverse_post_edge_share_delta",
        ],
        "passive fill event policy stability scorecard",
    )
    if (values[["train_candidate_events", "heldout_candidate_events"]] < 0.0).any().any():
        raise ValueError("passive fill event policy stability scorecard counts must be non-negative")

    data = values.copy()
    data[path_col] = stability[path_col].astype(str)
    data["heldout_stability_label"] = stability["heldout_stability_label"].astype(str)
    total_train_candidates = float(data["train_candidate_events"].sum())
    total_heldout_candidates = float(data["heldout_candidate_events"].sum())
    weights = (
        data["train_candidate_events"] / total_train_candidates
        if total_train_candidates > 0.0
        else pd.Series(0.0, index=data.index)
    )

    labels = data["heldout_stability_label"]
    blocker = labels == "heldout_policy_blocker"
    review = labels == "heldout_policy_review"
    no_event = labels == "heldout_policy_no_events"
    missing = labels == "heldout_policy_missing"

    def candidate_share(mask: pd.Series) -> float:
        if total_train_candidates <= 0.0:
            return 0.0
        return float(data.loc[mask, "train_candidate_events"].sum() / total_train_candidates)

    blocker_share = candidate_share(blocker)
    review_share = candidate_share(review)
    no_event_share = candidate_share(no_event)
    missing_share = candidate_share(missing)
    weighted_edge_delta = float((data["mean_post_minus_pre_realized_edge_delta"] * weights).sum())
    weighted_adverse_delta = float((data["adverse_post_edge_share_delta"] * weights).sum())

    blocking_reasons: list[str] = []
    review_reasons: list[str] = []
    if blocker_share > max_blocker_candidate_share:
        blocking_reasons.append("heldout_blocker_candidate_share")
    if weighted_edge_delta < min_weighted_edge_delta:
        blocking_reasons.append("weighted_edge_delta_breach")
    if review_share + no_event_share + missing_share > max_review_candidate_share:
        review_reasons.append("heldout_review_or_empty_candidate_share")
    if weighted_adverse_delta > max_weighted_adverse_delta:
        review_reasons.append("weighted_adverse_delta_breach")
    if missing_share > 0.0:
        review_reasons.append("missing_heldout_policy_paths")

    if blocking_reasons:
        decision = "block"
        gate_label = "passive_fill_policy_stability_blocked"
    elif review_reasons:
        decision = "review"
        gate_label = "passive_fill_policy_stability_review"
    else:
        decision = "pass"
        gate_label = "passive_fill_policy_stability_pass"

    priority = labels.map(
        {
            "heldout_policy_blocker": 4,
            "heldout_policy_review": 3,
            "heldout_policy_no_events": 2,
            "heldout_policy_missing": 2,
            "heldout_policy_stable": 1,
        }
    ).fillna(0)
    worst = data.assign(_priority=priority).sort_values(
        [
            "_priority",
            "train_candidate_events",
            "adverse_post_edge_share_delta",
            "mean_post_minus_pre_realized_edge_delta",
            path_col,
            threshold_col,
        ],
        ascending=[False, False, False, True, True, True],
    ).iloc[0]

    return {
        "rows": int(len(data)),
        "policy_paths": int(data[path_col].nunique()),
        "total_train_candidate_events": int(total_train_candidates),
        "total_heldout_candidate_events": int(total_heldout_candidates),
        "candidate_event_retention": float(
            total_heldout_candidates / total_train_candidates if total_train_candidates > 0.0 else 0.0
        ),
        "blocker_rows": int(blocker.sum()),
        "review_rows": int(review.sum()),
        "no_event_rows": int(no_event.sum()),
        "missing_rows": int(missing.sum()),
        "blocker_train_candidate_share": blocker_share,
        "review_train_candidate_share": review_share,
        "no_event_train_candidate_share": no_event_share,
        "missing_train_candidate_share": missing_share,
        "weighted_mean_post_minus_pre_realized_edge_delta": weighted_edge_delta,
        "weighted_adverse_post_edge_share_delta": weighted_adverse_delta,
        "worst_policy_path": str(worst[path_col]),
        "worst_threshold": float(worst[threshold_col]),
        "worst_heldout_stability_label": str(worst["heldout_stability_label"]),
        "policy_stability_decision": decision,
        "policy_stability_label": gate_label,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
    }



def passive_fill_event_lifecycle_scorecard(
    lifecycle_summary: pd.DataFrame,
    *,
    max_adverse_post_edge_share: float = 0.60,
    min_mean_post_minus_pre_edge: float = -0.25,
    min_events: int = 1,
) -> dict[str, float | int | str]:
    """Gate passive-fill toxicity by full pre/event/post regime lifecycle paths.

    Transition-level scorecards can still hide toxicity that appears only when the
    fill event itself occurs in a fragile liquidity state. This scorecard consumes
    ``passive_fill_event_lifecycle_summary`` and blocks full lifecycle paths where
    high-probability passive fills are followed by adverse realized-edge drift.
    """
    if not math.isfinite(max_adverse_post_edge_share) or not 0.0 <= max_adverse_post_edge_share <= 1.0:
        raise ValueError("max_adverse_post_edge_share must be finite and between 0 and 1")
    if not math.isfinite(min_mean_post_minus_pre_edge):
        raise ValueError("min_mean_post_minus_pre_edge must be finite")
    if not isinstance(min_events, int) or isinstance(min_events, bool) or min_events < 1:
        raise ValueError("min_events must be a positive integer")
    if lifecycle_summary.empty:
        return _empty_passive_fill_event_lifecycle_scorecard()

    required = {
        "lifecycle_path",
        "events",
        "adverse_post_edge_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_post_minus_pre_realized_edge",
        "worst_post_minus_pre_realized_edge",
    }
    _require_columns(
        lifecycle_summary,
        required,
        "passive fill event lifecycle toxicity",
    )
    values = _finite_values(
        lifecycle_summary,
        [
            "events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
        ],
        "passive fill event lifecycle toxicity",
    )
    if (values["events"] < 0.0).any():
        raise ValueError("passive fill event lifecycle toxicity events must be non-negative")

    total_events = int(values["events"].sum())
    if total_events == 0:
        return _empty_passive_fill_event_lifecycle_scorecard()

    data = values.copy()
    data["lifecycle_path"] = lifecycle_summary["lifecycle_path"].astype(str)
    weights = data["events"] / total_events
    eligible = data[data["events"] >= min_events]
    if eligible.empty:
        scorecard = _empty_passive_fill_event_lifecycle_scorecard()
        scorecard.update(
            {
                "rows": int(len(lifecycle_summary)),
                "lifecycle_paths": int(data["lifecycle_path"].nunique()),
                "total_events": total_events,
                "weighted_mean_event_fill_probability": float(
                    (data["mean_event_fill_probability"] * weights).sum()
                ),
                "weighted_mean_event_adverse_fill_probability": float(
                    (data["mean_event_adverse_fill_probability"] * weights).sum()
                ),
                "weighted_mean_post_minus_pre_realized_edge": float(
                    (data["mean_post_minus_pre_realized_edge"] * weights).sum()
                ),
                "lifecycle_toxicity_gate_label": "insufficient_lifecycle_event_windows",
            }
        )
        return scorecard

    blocked = eligible[
        (eligible["adverse_post_edge_share"] > max_adverse_post_edge_share)
        | (eligible["mean_post_minus_pre_realized_edge"] < min_mean_post_minus_pre_edge)
    ]
    worst_idx = eligible.sort_values(
        ["adverse_post_edge_share", "mean_post_minus_pre_realized_edge"],
        ascending=[False, True],
    ).index[0]
    label = "lifecycle_event_window_blocker" if not blocked.empty else "lifecycle_event_window_pass"

    return {
        "rows": int(len(lifecycle_summary)),
        "lifecycle_paths": int(data["lifecycle_path"].nunique()),
        "total_events": total_events,
        "eligible_lifecycle_paths": int(len(eligible)),
        "blocked_lifecycle_paths": int(len(blocked)),
        "worst_lifecycle_path": str(data.loc[worst_idx, "lifecycle_path"]),
        "worst_adverse_post_edge_share": float(data.loc[worst_idx, "adverse_post_edge_share"]),
        "worst_mean_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "mean_post_minus_pre_realized_edge"]
        ),
        "worst_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "worst_post_minus_pre_realized_edge"]
        ),
        "weighted_mean_event_fill_probability": float(
            (data["mean_event_fill_probability"] * weights).sum()
        ),
        "weighted_mean_event_adverse_fill_probability": float(
            (data["mean_event_adverse_fill_probability"] * weights).sum()
        ),
        "weighted_mean_post_minus_pre_realized_edge": float(
            (data["mean_post_minus_pre_realized_edge"] * weights).sum()
        ),
        "lifecycle_toxicity_gate_label": label,
    }


def passive_fill_event_transition_scorecard(
    transition_summary: pd.DataFrame,
    *,
    max_adverse_post_edge_share: float = 0.60,
    min_mean_post_minus_pre_edge: float = -0.25,
    min_events: int = 1,
) -> dict[str, float | int | str]:
    """Gate passive-fill event-window toxicity at regime-transition boundaries.

    Regime summaries can hide boundary-specific failures. This scorecard consumes
    ``passive_fill_event_transition_summary`` and promotes pre→post regime paths
    to first-class publishability gates, blocking transitions where high-probability
    fills are followed by excessive adverse drift.
    """
    if not math.isfinite(max_adverse_post_edge_share) or not 0.0 <= max_adverse_post_edge_share <= 1.0:
        raise ValueError("max_adverse_post_edge_share must be finite and between 0 and 1")
    if not math.isfinite(min_mean_post_minus_pre_edge):
        raise ValueError("min_mean_post_minus_pre_edge must be finite")
    if not isinstance(min_events, int) or isinstance(min_events, bool) or min_events < 1:
        raise ValueError("min_events must be a positive integer")
    if transition_summary.empty:
        return _empty_passive_fill_event_transition_scorecard()

    required = {
        "regime_transition",
        "events",
        "adverse_post_edge_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_post_minus_pre_realized_edge",
        "worst_post_minus_pre_realized_edge",
    }
    _require_columns(
        transition_summary,
        required,
        "passive fill event transition toxicity",
    )
    values = _finite_values(
        transition_summary,
        [
            "events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
        ],
        "passive fill event transition toxicity",
    )
    if (values["events"] < 0.0).any():
        raise ValueError("passive fill event transition toxicity events must be non-negative")

    total_events = int(values["events"].sum())
    if total_events == 0:
        return _empty_passive_fill_event_transition_scorecard()

    data = values.copy()
    data["regime_transition"] = transition_summary["regime_transition"].astype(str)
    weights = data["events"] / total_events
    eligible = data[data["events"] >= min_events]
    if eligible.empty:
        scorecard = _empty_passive_fill_event_transition_scorecard()
        scorecard.update(
            {
                "rows": int(len(transition_summary)),
                "transitions": int(data["regime_transition"].nunique()),
                "total_events": total_events,
                "weighted_mean_event_fill_probability": float(
                    (data["mean_event_fill_probability"] * weights).sum()
                ),
                "weighted_mean_event_adverse_fill_probability": float(
                    (data["mean_event_adverse_fill_probability"] * weights).sum()
                ),
                "weighted_mean_post_minus_pre_realized_edge": float(
                    (data["mean_post_minus_pre_realized_edge"] * weights).sum()
                ),
                "transition_toxicity_label": "insufficient_transition_event_windows",
            }
        )
        return scorecard

    blocked = eligible[
        (eligible["adverse_post_edge_share"] > max_adverse_post_edge_share)
        | (eligible["mean_post_minus_pre_realized_edge"] < min_mean_post_minus_pre_edge)
    ]
    worst_idx = eligible.sort_values(
        ["adverse_post_edge_share", "mean_post_minus_pre_realized_edge"],
        ascending=[False, True],
    ).index[0]
    label = (
        "transition_event_window_blocker"
        if not blocked.empty
        else "transition_event_window_pass"
    )

    return {
        "rows": int(len(transition_summary)),
        "transitions": int(data["regime_transition"].nunique()),
        "total_events": total_events,
        "eligible_transitions": int(len(eligible)),
        "blocked_transitions": int(len(blocked)),
        "worst_transition": str(data.loc[worst_idx, "regime_transition"]),
        "worst_adverse_post_edge_share": float(data.loc[worst_idx, "adverse_post_edge_share"]),
        "worst_mean_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "mean_post_minus_pre_realized_edge"]
        ),
        "worst_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "worst_post_minus_pre_realized_edge"]
        ),
        "weighted_mean_event_fill_probability": float(
            (data["mean_event_fill_probability"] * weights).sum()
        ),
        "weighted_mean_event_adverse_fill_probability": float(
            (data["mean_event_adverse_fill_probability"] * weights).sum()
        ),
        "weighted_mean_post_minus_pre_realized_edge": float(
            (data["mean_post_minus_pre_realized_edge"] * weights).sum()
        ),
        "transition_toxicity_label": label,
    }


def _empty_queue_position_toxicity_surface() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "best_execution_side",
            "queue_bin",
            "rows",
            "mean_queue_share",
            "mean_predicted_fill_probability",
            "mean_adverse_fill_probability",
            "adverse_to_fill_ratio",
            "realized_fill_rate",
            "realized_loss_rate",
            "mean_realized_edge_ticks",
            "mean_execution_adjusted_edge_ticks",
            "queue_toxicity_label",
        ]
    )


def _empty_queue_position_fraction_sweep() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "queue_position_fraction",
            "rows",
            "mean_bid_queue_share",
            "mean_ask_queue_share",
            "mean_bid_fill_probability",
            "mean_ask_fill_probability",
            "mean_fill_probability_imbalance",
            "mean_bid_adverse_fill_probability",
            "mean_ask_adverse_fill_probability",
            "mean_execution_adjusted_edge_ticks",
            "tradable_share",
            "abstain_share",
            "dominant_execution_side",
        ]
    )


def _empty_queue_position_capacity_frontier() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "viable_rows": 0,
        "front_queue_position_fraction": 0.0,
        "max_viable_queue_position_fraction": 0.0,
        "front_mean_execution_adjusted_edge_ticks": 0.0,
        "max_viable_mean_execution_adjusted_edge_ticks": 0.0,
        "edge_decay_to_capacity_ticks": 0.0,
        "front_tradable_share": 0.0,
        "max_viable_tradable_share": 0.0,
        "tradable_share_decay_to_capacity": 0.0,
        "dominant_execution_side_at_capacity": "none",
        "capacity_label": "empty_sweep",
    }


def _empty_queue_position_regime_capacity_stability(*, regime_col: str = "regime") -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            regime_col,
            "research_missing",
            "heldout_missing",
            "research_viable_rows",
            "heldout_viable_rows",
            "research_capacity_label",
            "heldout_capacity_label",
            "research_max_viable_queue_position_fraction",
            "heldout_max_viable_queue_position_fraction",
            "capacity_fraction_gap",
            "capacity_edge_gap_ticks",
            "capacity_tradable_share_gap",
            "capacity_viable_row_gap",
            "dominant_side_changed",
            "lost_capacity",
            "gained_capacity",
            "regime_capacity_stability_label",
        ]
    )


def _queue_capacity_label(*, max_fraction: float, edge_decay: float, tradable_decay: float) -> str:
    if max_fraction >= 0.75 and edge_decay <= 0.25 and tradable_decay <= 0.15:
        return "deep_queue_resilient_capacity"
    if max_fraction >= 0.50:
        return "queue_capacity_constrained"
    return "front_queue_only_capacity"


def _queue_regime_capacity_brittleness_label(
    *, viable_rows: int, max_fraction: float, capacity_label: str
) -> str:
    if viable_rows <= 0 or capacity_label == "no_viable_passive_capacity":
        return "regime_capacity_not_viable"
    if max_fraction <= 0.0:
        return "regime_capacity_front_only"
    if max_fraction < 0.75:
        return "regime_capacity_partial"
    return "regime_capacity_resilient"


def _queue_capacity_stability_label(
    *,
    fraction_gap: float,
    edge_gap: float,
    tradable_gap: float,
    viable_row_gap: int,
    dominant_side_changed: bool,
    heldout_label: str,
    max_fraction_gap: float,
    max_edge_gap_ticks: float,
    max_tradable_share_gap: float,
) -> str:
    if heldout_label in {"empty_sweep", "no_viable_passive_capacity"}:
        return "capacity_not_replicated"
    if (
        fraction_gap < -max_fraction_gap
        or edge_gap < -max_edge_gap_ticks
        or tradable_gap < -max_tradable_share_gap
        or viable_row_gap < 0
        or dominant_side_changed
    ):
        return "capacity_fragile"
    return "capacity_stable"


def _empty_passive_fill_edge_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bin",
            "rows",
            "long_rows",
            "short_rows",
            "mean_predicted_fill_probability",
            "mean_adverse_fill_probability",
            "mean_realized_edge_ticks",
            "positive_edge_rate",
            "mean_execution_adjusted_edge_ticks",
        ]
    )


def _empty_passive_fill_threshold_policy_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "threshold",
            "candidate_rows",
            "trade_share",
            "long_rows",
            "short_rows",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "weighted_brier_score",
            "mean_realized_edge_ticks",
            "positive_edge_rate",
            "mean_execution_adjusted_edge_ticks",
            "policy_label",
        ]
    )


def _passive_fill_threshold_policy_label(
    *, trade_share: float, realized_fill_rate: float, mean_realized_edge: float
) -> str:
    if trade_share <= 0.0:
        return "no_executable_policy"
    if realized_fill_rate >= 0.75 and mean_realized_edge > 0.0:
        if trade_share >= 0.50:
            return "broad_execution_policy"
        return "selective_high_quality_policy"
    if trade_share >= 0.50 and realized_fill_rate >= 0.60 and mean_realized_edge > 0.0:
        return "broad_execution_policy"
    if mean_realized_edge > 0.0:
        return "edge_positive_fill_uncertain_policy"
    return "execution_policy_rejected"


def _empty_queue_position_fill_surface() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "queue_bin",
            "fill_probability_bin",
            "rows",
            "mean_queue_share",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "calibration_error",
            "absolute_calibration_error",
            "brier_score",
            "mean_execution_adjusted_edge_ticks",
        ]
    )


def _empty_queue_position_fill_calibration_surface() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "best_execution_side",
            "queue_share_bin",
            "fill_probability_bin",
            "rows",
            "mean_queue_share",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "calibration_error",
            "absolute_calibration_error",
            "brier_score",
            "mean_execution_adjusted_edge_ticks",
        ]
    )


def _empty_queue_position_calibration_drift() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "best_execution_side",
            "queue_share_bin",
            "fill_probability_bin",
            "regimes",
            "rows",
            "min_realized_fill_rate",
            "max_realized_fill_rate",
            "fill_rate_range",
            "min_absolute_calibration_error",
            "max_absolute_calibration_error",
            "calibration_error_range",
            "weighted_mean_absolute_calibration_error",
            "worst_regime",
            "drift_label",
        ]
    )


def _empty_queue_position_calibration_residual_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "best_execution_side",
            "bins",
            "rows",
            "underfilled_bins",
            "overfilled_bins",
            "weighted_mean_queue_share",
            "weighted_calibration_error",
            "weighted_absolute_calibration_error",
            "weighted_mean_execution_adjusted_edge_ticks",
            "worst_queue_share_bin",
            "worst_fill_probability_bin",
            "worst_absolute_calibration_error",
            "residual_label",
        ]
    )


def _queue_calibration_residual_label(
    *,
    weighted_error: float,
    weighted_abs_error: float,
    weighted_edge: float,
    error_threshold: float,
) -> str:
    if weighted_error < -error_threshold and weighted_edge <= 0.0:
        return "underfilled_execution_drag"
    if weighted_error < -error_threshold:
        return "underfilled_but_edge_positive"
    if weighted_error > error_threshold and weighted_edge >= 0.0:
        return "overfilled_execution_opportunity"
    if weighted_abs_error > error_threshold:
        return "calibration_residual_watch"
    return "calibration_residual_controlled"


def _queue_calibration_drift_label(
    *, fill_rate_range: float, calibration_error_range: float, weighted_error: float
) -> str:
    if calibration_error_range > 0.15 or fill_rate_range > 0.25 or weighted_error > 0.20:
        return "calibration_unstable"
    if calibration_error_range > 0.05 or fill_rate_range > 0.10 or weighted_error > 0.10:
        return "calibration_watch"
    return "calibration_stable"


def _empty_queue_position_edge_decay() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "queue_bins",
            "rows",
            "front_queue_bin",
            "back_queue_bin",
            "front_mean_queue_share",
            "back_mean_queue_share",
            "fill_rate_decay",
            "predicted_fill_decay",
            "edge_decay_ticks",
            "calibration_error_widening",
            "monotonic_edge_decay",
            "worst_queue_bin",
            "worst_mean_execution_adjusted_edge_ticks",
            "queue_decay_label",
        ]
    )


def _queue_decay_label(*, edge_decay: float, fill_decay: float, calibration_widening: float) -> str:
    if calibration_widening > 0.100000000001:
        return "calibration_watch"
    if edge_decay > 0.0 and fill_decay >= 0.0:
        return "front_queue_preferred"
    if edge_decay <= 0.0 and fill_decay <= 0.0:
        return "deep_queue_resilient"
    return "mixed_queue_response"


def _empty_passive_fill_calibration_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "bin",
            "rows",
            "long_rows",
            "short_rows",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "calibration_error",
            "absolute_calibration_error",
            "brier_score",
        ]
    )


def _empty_passive_fill_calibration_summary() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "bins": 0,
        "regimes": 0,
        "weighted_mean_predicted_fill_probability": 0.0,
        "weighted_realized_fill_rate": 0.0,
        "weighted_calibration_error": 0.0,
        "expected_calibration_error": 0.0,
        "weighted_brier_score": 0.0,
        "worst_regime": "none",
        "worst_absolute_calibration_error": 0.0,
    }


def _empty_passive_fill_realization_horizon_sweep() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "horizon",
            "rows",
            "bins",
            "regimes",
            "weighted_mean_predicted_fill_probability",
            "weighted_realized_fill_rate",
            "weighted_calibration_error",
            "expected_calibration_error",
            "weighted_brier_score",
            "worst_absolute_calibration_error",
            "realized_fill_rate_gap_vs_shortest",
            "brier_score_gap_vs_shortest",
            "horizon_stability_label",
        ]
    )


def _empty_event_level_passive_fill_horizon_sweep() -> pd.DataFrame:
    columns = list(_empty_passive_fill_realization_horizon_sweep().columns)
    columns.insert(1, "event_depletion_source")
    return pd.DataFrame(columns=columns)


def _passive_fill_horizon_stability_label(
    *, index: int, realized_gap: float, brier_gap: float
) -> str:
    if index == 0:
        return "anchor_horizon"
    if realized_gap >= 0.10 and brier_gap <= 0.0:
        return "later_fill_realization"
    if realized_gap <= -0.10 or brier_gap >= 0.05:
        return "horizon_fragile"
    return "horizon_stable"


def _empty_passive_fill_event_windows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_index",
            "event_side",
            "event_regime",
            "pre_window_regime",
            "post_window_regime",
            "regime_transition",
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "pre_realized_edge_sum",
            "post_realized_edge_sum",
            "post_minus_pre_realized_edge",
            "window_rows",
        ]
    )


def _empty_passive_fill_event_regime_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_regime",
            "events",
            "adverse_post_edge_events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_event_edge_ticks",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
        ]
    )


def _empty_passive_fill_event_transition_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime_transition",
            "event_regimes",
            "events",
            "adverse_post_edge_events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_event_edge_ticks",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
        ]
    )


def _empty_passive_fill_event_lifecycle_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "lifecycle_path",
            "pre_window_regime",
            "event_regime",
            "post_window_regime",
            "regime_transitions",
            "events",
            "adverse_post_edge_events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_event_edge_ticks",
            "mean_pre_realized_edge_sum",
            "mean_post_realized_edge_sum",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
            "mean_window_rows",
            "lifecycle_toxicity_label",
        ]
    )


def _empty_passive_fill_event_toxicity_scorecard() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "regimes": 0,
        "total_events": 0,
        "eligible_regimes": 0,
        "blocked_regimes": 0,
        "worst_regime": "none",
        "worst_adverse_post_edge_share": 0.0,
        "worst_mean_post_minus_pre_realized_edge": 0.0,
        "worst_post_minus_pre_realized_edge": 0.0,
        "weighted_mean_event_fill_probability": 0.0,
        "weighted_mean_event_adverse_fill_probability": 0.0,
        "weighted_mean_post_minus_pre_realized_edge": 0.0,
        "event_toxicity_label": "empty_event_windows",
    }


def _empty_passive_fill_event_lifecycle_scorecard() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "lifecycle_paths": 0,
        "total_events": 0,
        "eligible_lifecycle_paths": 0,
        "blocked_lifecycle_paths": 0,
        "worst_lifecycle_path": "none",
        "worst_adverse_post_edge_share": 0.0,
        "worst_mean_post_minus_pre_realized_edge": 0.0,
        "worst_post_minus_pre_realized_edge": 0.0,
        "weighted_mean_event_fill_probability": 0.0,
        "weighted_mean_event_adverse_fill_probability": 0.0,
        "weighted_mean_post_minus_pre_realized_edge": 0.0,
        "lifecycle_toxicity_gate_label": "empty_lifecycle_event_windows",
    }


def _empty_passive_fill_event_transition_scorecard() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "transitions": 0,
        "total_events": 0,
        "eligible_transitions": 0,
        "blocked_transitions": 0,
        "worst_transition": "none",
        "worst_adverse_post_edge_share": 0.0,
        "worst_mean_post_minus_pre_realized_edge": 0.0,
        "worst_post_minus_pre_realized_edge": 0.0,
        "weighted_mean_event_fill_probability": 0.0,
        "weighted_mean_event_adverse_fill_probability": 0.0,
        "weighted_mean_post_minus_pre_realized_edge": 0.0,
        "transition_toxicity_label": "empty_transition_event_windows",
    }


def _rank_probability_bins(probability: pd.Series, bins: int) -> pd.Series:
    effective_bins = min(bins, len(probability))
    ranks = probability.rank(method="first")
    bin_ids = pd.qcut(ranks, q=effective_bins, labels=False, duplicates="drop")
    return bin_ids.astype(int) + 1


def execution_publishability_release_gate(
    review_packet: pd.DataFrame,
    *,
    quality_gate: dict[str, float | int | str] | None = None,
    capacity_stability: dict[str, float | int | str | bool] | None = None,
    regime_capacity_stability: dict[str, float | int | str] | None = None,
    max_conflict_share: float = 0.25,
    max_high_priority_conflict_share: float = 0.10,
) -> dict[str, float | int | str | bool]:
    """Reduce execution-aware artifacts into an owner-facing release gate.

    The review packet catches pre/post execution gate conflicts, while queue
    quality and capacity-stability summaries test whether passive fill evidence
    is calibrated and out-of-sample durable. This reducer deliberately keeps the
    output JSON-like so demos and reports can publish a single execution release
    decision instead of scattering the evidence across multiple tables.
    """
    for name, value in {
        "max_conflict_share": max_conflict_share,
        "max_high_priority_conflict_share": max_high_priority_conflict_share,
    }.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0.0, 1.0]")

    required = {"rows", "conflict_rows", "review_priority"}
    if review_packet.empty:
        total_rows = 0
        conflict_rows = 0
        weighted_conflict_share = 0.0
        high_priority_conflict_rows = 0
        high_priority_conflict_share = 0.0
    else:
        _require_columns(review_packet, required, "execution publishability release gate")
        values = _finite_values(
            review_packet,
            ["rows", "conflict_rows", "review_priority"],
            "execution publishability release gate",
        )
        if (values[["rows", "conflict_rows", "review_priority"]] < 0.0).any().any():
            raise ValueError("execution publishability release gate counts must be non-negative")
        total_rows = int(values["rows"].sum())
        conflict_rows = int(values["conflict_rows"].sum())
        weighted_conflict_share = float(conflict_rows / total_rows) if total_rows else 0.0
        high_priority = values["review_priority"] >= 3.0
        high_priority_conflict_rows = int(values.loc[high_priority, "conflict_rows"].sum())
        high_priority_conflict_share = (
            float(high_priority_conflict_rows / total_rows) if total_rows else 0.0
        )

    quality_label = str(
        (quality_gate or {}).get("quality_gate_label", "missing_queue_execution_quality_gate")
    )
    capacity_label = str(
        (capacity_stability or {}).get(
            "capacity_stability_label", "missing_capacity_stability_gate"
        )
    )
    regime_capacity = regime_capacity_stability or {}
    regime_capacity_label = str(
        regime_capacity.get(
            "regime_capacity_stability_label",
            "regime_capacity_stability_not_evaluated",
        )
    )
    lost_capacity_regimes = int(regime_capacity.get("lost_capacity_regimes", 0))
    stable_regime_share = float(regime_capacity.get("stable_regime_share", 0.0))
    worst_capacity_regime = str(regime_capacity.get("worst_regime", "none"))

    blocking_reasons: list[str] = []
    review_reasons: list[str] = []
    if total_rows == 0:
        review_reasons.append("empty_execution_review_packet")
    if weighted_conflict_share > max_conflict_share:
        blocking_reasons.append("execution_conflict_share_exceeds_limit")
    if high_priority_conflict_share > max_high_priority_conflict_share:
        blocking_reasons.append("high_priority_execution_conflicts_exceed_limit")
    if quality_label in {"empty_queue_execution_surface", "queue_execution_blocked"}:
        blocking_reasons.append(quality_label)
    elif quality_label in {"queue_execution_review", "missing_queue_execution_quality_gate"}:
        review_reasons.append(quality_label)
    if capacity_label in {"capacity_not_replicated", "capacity_fragile"}:
        blocking_reasons.append(capacity_label)
    elif capacity_label == "missing_capacity_stability_gate":
        review_reasons.append(capacity_label)
    if regime_capacity_label == "regime_capacity_not_replicated":
        blocking_reasons.append(regime_capacity_label)
    elif regime_capacity_label in {"regime_capacity_fragile", "regime_capacity_mixed"}:
        review_reasons.append(regime_capacity_label)

    if blocking_reasons:
        decision = "block"
        label = "execution_release_blocked"
        passes = False
    elif review_reasons:
        decision = "review"
        label = "execution_release_review"
        passes = False
    else:
        decision = "pass"
        label = "execution_release_publishable"
        passes = True

    return {
        "total_rows": total_rows,
        "conflict_rows": conflict_rows,
        "weighted_conflict_share": weighted_conflict_share,
        "high_priority_conflict_rows": high_priority_conflict_rows,
        "high_priority_conflict_share": high_priority_conflict_share,
        "quality_gate_label": quality_label,
        "capacity_stability_label": capacity_label,
        "regime_capacity_stability_label": regime_capacity_label,
        "lost_capacity_regimes": lost_capacity_regimes,
        "stable_regime_share": stable_regime_share,
        "worst_capacity_regime": worst_capacity_regime,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
        "decision": decision,
        "passes": passes,
        "release_gate_label": label,
    }


def execution_publishability_review_packet(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pre/post-execution publishability conflicts for review.

    The packet cross-tabs the pre-execution ``publishable_side`` gate against the
    queue/adverse-fill-aware ``best_execution_side``. It is intentionally small
    enough to ship as a demo/report artifact while still exposing the failure
    modes that matter for market microstructure review: signals that abstain
    after queue-position adjustment, side flips, and opportunities surfaced only
    by the execution layer.
    """
    columns = list(_empty_execution_publishability_review_packet().columns)
    required = {
        "publishable_side",
        "best_execution_side",
        "execution_adjusted_edge_ticks",
        "long_fill_adjusted_edge_ticks",
        "short_fill_adjusted_edge_ticks",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    if frame.empty:
        return _empty_execution_publishability_review_packet()
    _require_columns(frame, required, "execution publishability review")
    values = _finite_values(
        frame,
        [
            "execution_adjusted_edge_ticks",
            "long_fill_adjusted_edge_ticks",
            "short_fill_adjusted_edge_ticks",
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ],
        "execution publishability review",
    )

    publishable_side = frame["publishable_side"].astype(str)
    best_side = frame["best_execution_side"].astype(str)
    row_state = pd.DataFrame(
        {
            "publishable_side": publishable_side,
            "best_execution_side": best_side,
            "execution_adjusted_edge_ticks": values["execution_adjusted_edge_ticks"],
            "best_fill_probability": _side_probability(
                best_side,
                bid=values["bid_fill_probability"],
                ask=values["ask_fill_probability"],
            ),
            "best_adverse_fill_probability": _side_probability(
                best_side,
                bid=values["bid_adverse_fill_probability"],
                ask=values["ask_adverse_fill_probability"],
            ),
            "publishable_fill_probability": _side_probability(
                publishable_side,
                bid=values["bid_fill_probability"],
                ask=values["ask_fill_probability"],
            ),
            "publishable_edge_ticks": _side_probability(
                publishable_side,
                bid=values["long_fill_adjusted_edge_ticks"],
                ask=values["short_fill_adjusted_edge_ticks"],
            ),
        }
    )
    row_state["edge_drag_ticks"] = (
        row_state["execution_adjusted_edge_ticks"] - row_state["publishable_edge_ticks"]
    )
    row_state["is_conflict"] = row_state["publishable_side"] != row_state["best_execution_side"]

    rows: list[dict[str, float | int | str]] = []
    for (published, best), group in row_state.groupby(
        ["publishable_side", "best_execution_side"], sort=True
    ):
        conflict_rows = int(group["is_conflict"].sum())
        rows.append(
            {
                "publishable_side": str(published),
                "best_execution_side": str(best),
                "rows": len(group),
                "conflict_rows": conflict_rows,
                "conflict_share": float(group["is_conflict"].mean()),
                "mean_execution_adjusted_edge_ticks": float(
                    group["execution_adjusted_edge_ticks"].mean()
                ),
                "mean_best_fill_probability": float(group["best_fill_probability"].mean()),
                "mean_best_adverse_fill_probability": float(
                    group["best_adverse_fill_probability"].mean()
                ),
                "mean_publishable_fill_probability": float(
                    group["publishable_fill_probability"].mean()
                ),
                "mean_edge_drag_ticks": float(group["edge_drag_ticks"].mean()),
                "review_priority": _execution_review_priority(str(published), str(best)),
                "review_note": _execution_review_note(str(published), str(best)),
            }
        )

    packet = pd.DataFrame(rows)[columns]
    return packet.sort_values(
        ["review_priority", "conflict_rows", "rows", "publishable_side", "best_execution_side"],
        ascending=[False, False, False, True, True],
        ignore_index=True,
    )


def _side_probability(side: pd.Series, *, bid: pd.Series, ask: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([side == "long", side == "short"], [bid, ask], default=0.0),
        index=side.index,
    )


def _execution_review_priority(publishable_side: str, best_execution_side: str) -> int:
    if publishable_side == best_execution_side:
        return 0
    if publishable_side in {"long", "short"}:
        return 3
    if best_execution_side in {"long", "short"}:
        return 2
    return 1


def _execution_review_note(publishable_side: str, best_execution_side: str) -> str:
    if publishable_side == best_execution_side:
        return f"pre-execution and execution-aware gates agree on {best_execution_side}"
    if publishable_side in {"long", "short"} and best_execution_side == "abstain":
        return (
            f"pre-execution {publishable_side} signal abstains after "
            "queue/adverse-fill adjustment"
        )
    if publishable_side in {"long", "short"} and best_execution_side in {"long", "short"}:
        return f"pre-execution {publishable_side} signal flips to {best_execution_side} after execution adjustment"
    if publishable_side == "abstain" and best_execution_side in {"long", "short"}:
        return f"execution layer surfaces {best_execution_side} opportunity despite pre-execution abstain"
    return f"publishability gate changes from {publishable_side} to {best_execution_side} after execution adjustment"


def _empty_execution_publishability_review_packet() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "publishable_side",
            "best_execution_side",
            "rows",
            "conflict_rows",
            "conflict_share",
            "mean_execution_adjusted_edge_ticks",
            "mean_best_fill_probability",
            "mean_best_adverse_fill_probability",
            "mean_publishable_fill_probability",
            "mean_edge_drag_ticks",
            "review_priority",
            "review_note",
        ]
    )


def _empty_execution_summary() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "tradable_rows": 0,
        "abstain_rows": 0,
        "tradable_share": 0.0,
        "mean_execution_adjusted_edge_ticks": 0.0,
        "median_execution_adjusted_edge_ticks": 0.0,
        "mean_bid_fill_probability": 0.0,
        "mean_ask_fill_probability": 0.0,
        "mean_adverse_fill_probability": 0.0,
        "publishable_side_conflict_rows": 0,
        "publishable_side_conflict_share": 0.0,
        "dominant_execution_side": "none",
    }


def execution_adjusted_edge_summary(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize execution-aware tradability after queue-fill adjustment.

    The optional `publishable_side` column is treated as the pre-execution gate.
    When present, conflict counts expose rows where a signal that looked
    publishable before queue/adverse-fill adjustment changes side or abstains.
    """
    if frame.empty:
        return _empty_execution_summary()

    required = {
        "best_execution_side",
        "execution_adjusted_edge_ticks",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    _require_columns(frame, required, "execution summary")
    values = _finite_values(
        frame,
        [
            "execution_adjusted_edge_ticks",
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ],
        "execution summary",
    )

    side = frame["best_execution_side"].astype(str)
    tradable = side != "abstain"
    adverse = 0.5 * (values["bid_adverse_fill_probability"] + values["ask_adverse_fill_probability"])
    side_counts = side[tradable].value_counts()
    dominant_side = "none" if side_counts.empty else str(side_counts.idxmax())

    conflict_rows = 0
    conflict_share = 0.0
    if "publishable_side" in frame.columns:
        publishable_side = frame["publishable_side"].astype(str)
        conflict = publishable_side != side
        conflict_rows = int(conflict.sum())
        conflict_share = float(conflict.mean())

    return {
        "rows": len(frame),
        "tradable_rows": int(tradable.sum()),
        "abstain_rows": int((~tradable).sum()),
        "tradable_share": float(tradable.mean()),
        "mean_execution_adjusted_edge_ticks": float(values["execution_adjusted_edge_ticks"].mean()),
        "median_execution_adjusted_edge_ticks": float(values["execution_adjusted_edge_ticks"].median()),
        "mean_bid_fill_probability": float(values["bid_fill_probability"].mean()),
        "mean_ask_fill_probability": float(values["ask_fill_probability"].mean()),
        "mean_adverse_fill_probability": float(adverse.mean()),
        "publishable_side_conflict_rows": conflict_rows,
        "publishable_side_conflict_share": conflict_share,
        "dominant_execution_side": dominant_side,
    }


def _empty_execution_adjusted_lcri_side_attribution() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "lcri_side",
            "rows",
            "tradable_rows",
            "execution_conflict_rows",
            "execution_conflict_share",
            "mean_signal_confidence",
            "mean_execution_adjusted_edge_ticks",
            "mean_fill_probability_advantage",
            "mean_adverse_fill_probability_advantage",
            "dominant_execution_side",
            "review_label",
        ]
    )


def execution_adjusted_lcri_side_attribution(
    frame: pd.DataFrame,
    *,
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
) -> pd.DataFrame:
    """Attribute raw LCRI side survival, inversion, and friction after execution gating.

    This diagnostic turns execution-adjusted LCRI into an auditable review artifact:
    it groups rows by the raw LCRI side, then quantifies whether queue-aware passive
    execution preserves that side, forces abstention, or inverts the tradable side.
    The fill advantage is same-side minus opposite-side passive fill probability;
    the adverse metric is the same-side adverse-fill drag that must be paid to trade.
    """
    if frame.empty:
        return _empty_execution_adjusted_lcri_side_attribution()

    required = {
        signal_col,
        probability_col,
        "best_execution_side",
        "execution_adjusted_edge_ticks",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    _require_columns(frame, required, "execution-adjusted LCRI side attribution")
    values = _finite_values(
        frame,
        [
            signal_col,
            probability_col,
            "execution_adjusted_edge_ticks",
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ],
        "execution-adjusted LCRI side attribution",
    )
    probability = values[probability_col]
    if not probability.between(0.0, 1.0).all():
        raise ValueError("execution-adjusted LCRI side attribution probabilities must be in [0, 1]")

    signal = values[signal_col]
    side = pd.Series(
        np.select([signal > 0.0, signal < 0.0], ["long", "short"], default="neutral"),
        index=frame.index,
    )
    best_side = frame["best_execution_side"].astype(str)
    valid_sides = {"long", "short", "abstain"}
    unknown_sides = sorted(set(best_side) - valid_sides)
    if unknown_sides:
        raise ValueError(f"unknown execution sides: {unknown_sides}")

    diagnostics = pd.DataFrame(index=frame.index)
    diagnostics["lcri_side"] = side
    diagnostics["best_execution_side"] = best_side
    diagnostics["execution_adjusted_edge_ticks"] = values["execution_adjusted_edge_ticks"]
    diagnostics["tradable"] = best_side != "abstain"
    diagnostics["execution_conflict"] = (side != "neutral") & (best_side != side)
    diagnostics["signal_confidence"] = np.select(
        [side == "long", side == "short"],
        [probability, 1.0 - probability],
        default=0.50,
    )
    diagnostics["fill_probability_advantage"] = np.select(
        [side == "long", side == "short"],
        [values["bid_fill_probability"] - values["ask_fill_probability"],
         values["ask_fill_probability"] - values["bid_fill_probability"]],
        default=0.0,
    )
    diagnostics["adverse_fill_probability_advantage"] = np.select(
        [side == "long", side == "short"],
        [values["bid_adverse_fill_probability"], values["ask_adverse_fill_probability"]],
        default=0.0,
    )

    rows: list[dict[str, float | int | str]] = []
    for side_name in ["long", "short", "neutral"]:
        group = diagnostics[diagnostics["lcri_side"] == side_name]
        if group.empty:
            continue
        side_counts = group.loc[group["tradable"], "best_execution_side"].value_counts()
        dominant_side = "none" if side_counts.empty else str(side_counts.idxmax())
        conflict_share = float(group["execution_conflict"].mean())
        if side_name == "neutral":
            review_label = "neutral_signal"
        elif dominant_side not in {"none", side_name}:
            review_label = "execution_side_inversion_review"
        elif conflict_share > 0.0:
            review_label = "execution_friction_review"
        else:
            review_label = "execution_side_preserved"
        rows.append(
            {
                "lcri_side": side_name,
                "rows": int(len(group)),
                "tradable_rows": int(group["tradable"].sum()),
                "execution_conflict_rows": int(group["execution_conflict"].sum()),
                "execution_conflict_share": conflict_share,
                "mean_signal_confidence": float(group["signal_confidence"].mean()),
                "mean_execution_adjusted_edge_ticks": float(
                    group["execution_adjusted_edge_ticks"].mean()
                ),
                "mean_fill_probability_advantage": float(
                    group["fill_probability_advantage"].mean()
                ),
                "mean_adverse_fill_probability_advantage": float(
                    group["adverse_fill_probability_advantage"].mean()
                ),
                "dominant_execution_side": dominant_side,
                "review_label": review_label,
            }
        )
    return pd.DataFrame(rows)[list(_empty_execution_adjusted_lcri_side_attribution().columns)]


def _empty_execution_adjusted_lcri_regime_attribution() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "lcri_side",
            "rows",
            "tradable_rows",
            "execution_survival_rows",
            "execution_conflict_rows",
            "execution_survival_share",
            "execution_conflict_share",
            "mean_signal_confidence",
            "mean_execution_adjusted_edge_ticks",
            "mean_fill_probability_advantage",
            "mean_adverse_fill_probability_advantage",
            "dominant_execution_side",
            "review_label",
        ]
    )


def execution_adjusted_lcri_regime_attribution(
    frame: pd.DataFrame,
    *,
    regime_col: str = "regime",
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
) -> pd.DataFrame:
    """Attribute execution-adjusted LCRI side survival within liquidity regimes.

    Aggregate side attribution can hide that queue friction preserves a residual
    imbalance in one regime while inverting or abstaining it in another. This
    diagnostic groups by liquidity regime and raw LCRI side, then measures where
    the queue-aware execution layer preserves, blocks, or flips the signal.
    """
    if not regime_col:
        raise ValueError("regime_col must be non-empty")
    if frame.empty:
        return _empty_execution_adjusted_lcri_regime_attribution()

    required = {
        regime_col,
        signal_col,
        probability_col,
        "best_execution_side",
        "execution_adjusted_edge_ticks",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    _require_columns(frame, required, "execution-adjusted LCRI regime attribution")
    values = _finite_values(
        frame,
        [
            signal_col,
            probability_col,
            "execution_adjusted_edge_ticks",
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ],
        "execution-adjusted LCRI regime attribution",
    )
    probability = values[probability_col]
    if not probability.between(0.0, 1.0).all():
        raise ValueError("execution-adjusted LCRI regime probabilities must be in [0, 1]")

    signal = values[signal_col]
    side = pd.Series(
        np.select([signal > 0.0, signal < 0.0], ["long", "short"], default="neutral"),
        index=frame.index,
    )
    best_side = frame["best_execution_side"].astype(str)
    valid_sides = {"long", "short", "abstain"}
    unknown_sides = sorted(set(best_side) - valid_sides)
    if unknown_sides:
        raise ValueError(f"unknown execution sides: {unknown_sides}")

    diagnostics = pd.DataFrame(index=frame.index)
    diagnostics["regime"] = frame[regime_col].astype(str)
    diagnostics["lcri_side"] = side
    diagnostics["best_execution_side"] = best_side
    diagnostics["execution_adjusted_edge_ticks"] = values["execution_adjusted_edge_ticks"]
    diagnostics["tradable"] = best_side != "abstain"
    diagnostics["execution_survival"] = (side != "neutral") & (best_side == side)
    diagnostics["execution_conflict"] = (side != "neutral") & (best_side != side)
    diagnostics["signal_confidence"] = np.select(
        [side == "long", side == "short"],
        [probability, 1.0 - probability],
        default=0.50,
    )
    diagnostics["fill_probability_advantage"] = np.select(
        [side == "long", side == "short"],
        [
            values["bid_fill_probability"] - values["ask_fill_probability"],
            values["ask_fill_probability"] - values["bid_fill_probability"],
        ],
        default=0.0,
    )
    diagnostics["adverse_fill_probability_advantage"] = np.select(
        [side == "long", side == "short"],
        [values["bid_adverse_fill_probability"], values["ask_adverse_fill_probability"]],
        default=0.0,
    )

    rows: list[dict[str, float | int | str]] = []
    for (regime, side_name), group in diagnostics.groupby(["regime", "lcri_side"], sort=True):
        side_counts = group.loc[group["tradable"], "best_execution_side"].value_counts()
        dominant_side = "none" if side_counts.empty else str(side_counts.idxmax())
        survival_share = float(group["execution_survival"].mean())
        conflict_share = float(group["execution_conflict"].mean())
        if side_name == "neutral":
            review_label = "neutral_signal"
        elif dominant_side not in {"none", str(side_name)}:
            review_label = "execution_side_inversion_review"
        elif conflict_share > 0.0:
            review_label = "execution_friction_review"
        else:
            review_label = "execution_side_preserved"
        rows.append(
            {
                "regime": str(regime),
                "lcri_side": str(side_name),
                "rows": int(len(group)),
                "tradable_rows": int(group["tradable"].sum()),
                "execution_survival_rows": int(group["execution_survival"].sum()),
                "execution_conflict_rows": int(group["execution_conflict"].sum()),
                "execution_survival_share": survival_share,
                "execution_conflict_share": conflict_share,
                "mean_signal_confidence": float(group["signal_confidence"].mean()),
                "mean_execution_adjusted_edge_ticks": float(
                    group["execution_adjusted_edge_ticks"].mean()
                ),
                "mean_fill_probability_advantage": float(
                    group["fill_probability_advantage"].mean()
                ),
                "mean_adverse_fill_probability_advantage": float(
                    group["adverse_fill_probability_advantage"].mean()
                ),
                "dominant_execution_side": dominant_side,
                "review_label": review_label,
            }
        )
    output = pd.DataFrame(rows)[list(_empty_execution_adjusted_lcri_regime_attribution().columns)]
    if regime_col != "regime":
        output = output.rename(columns={"regime": regime_col})
    return output.sort_values(
        [regime_col, "execution_conflict_share", "rows", "lcri_side"],
        ascending=[True, False, False, True],
        ignore_index=True,
    )


def _empty_execution_adjusted_lcri_quantile_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bucket",
            "rows",
            "mean_abs_lcri",
            "mean_abs_execution_adjusted_lcri_score",
            "signal_survival_ratio",
            "tradable_share",
            "mean_selected_fill_probability",
            "mean_selected_adverse_fill_probability",
            "fill_minus_adverse_probability_spread",
            "mean_execution_adjusted_edge_ticks",
            "edge_drag_vs_raw_abs_lcri",
        ]
    )


def execution_adjusted_lcri_quantile_diagnostics(
    frame: pd.DataFrame,
    *,
    bins: int = 5,
    signal_col: str = "lcri",
    execution_signal_col: str = "execution_adjusted_lcri_score",
) -> pd.DataFrame:
    """Measure how much raw LCRI strength survives passive execution constraints.

    The table sorts rows by absolute raw LCRI, buckets them into equal-count quantiles,
    and reports the execution-adjusted signal retained after fill/adverse-fill gating.
    It makes the key publishability question auditable: are the largest residual
    imbalances still tradable after queue position, or is apparent alpha mostly eaten
    by passive-fill friction and abstention?
    """
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if frame.empty:
        return _empty_execution_adjusted_lcri_quantile_diagnostics()

    required = {
        signal_col,
        execution_signal_col,
        "execution_adjusted_edge_ticks",
        "best_execution_side",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    _require_columns(frame, required, "execution-adjusted LCRI quantile diagnostics")
    values = _finite_values(
        frame,
        [
            signal_col,
            execution_signal_col,
            "execution_adjusted_edge_ticks",
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ],
        "execution-adjusted LCRI quantile diagnostics",
    )

    diagnostics_frame = pd.DataFrame(index=frame.index)
    diagnostics_frame["abs_lcri"] = values[signal_col].abs()
    diagnostics_frame["abs_execution_adjusted_lcri_score"] = values[execution_signal_col].abs()
    diagnostics_frame["execution_adjusted_edge_ticks"] = values["execution_adjusted_edge_ticks"]
    best_side = frame["best_execution_side"].astype(str)
    diagnostics_frame["tradable"] = best_side != "abstain"
    diagnostics_frame["selected_fill_probability"] = np.select(
        [best_side == "long", best_side == "short"],
        [values["bid_fill_probability"], values["ask_fill_probability"]],
        default=0.0,
    )
    diagnostics_frame["selected_adverse_fill_probability"] = np.select(
        [best_side == "long", best_side == "short"],
        [values["bid_adverse_fill_probability"], values["ask_adverse_fill_probability"]],
        default=0.0,
    )

    actual_bins = min(bins, len(diagnostics_frame))
    ranks = diagnostics_frame["abs_lcri"].rank(method="first")
    diagnostics_frame["bucket_id"] = pd.qcut(ranks, q=actual_bins, labels=False, duplicates="drop")
    actual_bins = int(diagnostics_frame["bucket_id"].max()) + 1

    if actual_bins == 1:
        labels = ["all_abs_lcri"]
    elif actual_bins == 3:
        labels = ["low_abs_lcri", "mid_abs_lcri", "high_abs_lcri"]
    else:
        labels = [f"abs_lcri_q{index + 1:02d}" for index in range(actual_bins)]

    rows: list[dict[str, float | int | str]] = []
    for bucket_id, group in diagnostics_frame.groupby("bucket_id", sort=True):
        mean_abs_lcri = float(group["abs_lcri"].mean())
        mean_abs_execution_signal = float(group["abs_execution_adjusted_lcri_score"].mean())
        mean_selected_fill_probability = float(group["selected_fill_probability"].mean())
        mean_selected_adverse_fill_probability = float(
            group["selected_adverse_fill_probability"].mean()
        )
        rows.append(
            {
                "bucket": labels[int(bucket_id)],
                "rows": int(len(group)),
                "mean_abs_lcri": mean_abs_lcri,
                "mean_abs_execution_adjusted_lcri_score": mean_abs_execution_signal,
                "signal_survival_ratio": (
                    mean_abs_execution_signal / mean_abs_lcri if mean_abs_lcri > 0.0 else 0.0
                ),
                "tradable_share": float(group["tradable"].mean()),
                "mean_selected_fill_probability": mean_selected_fill_probability,
                "mean_selected_adverse_fill_probability": mean_selected_adverse_fill_probability,
                "fill_minus_adverse_probability_spread": float(
                    mean_selected_fill_probability - mean_selected_adverse_fill_probability
                ),
                "mean_execution_adjusted_edge_ticks": float(
                    group["execution_adjusted_edge_ticks"].mean()
                ),
                "edge_drag_vs_raw_abs_lcri": float(
                    mean_abs_lcri - group["execution_adjusted_edge_ticks"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_execution_adjusted_lcri_quantile_diagnostics().columns)]
