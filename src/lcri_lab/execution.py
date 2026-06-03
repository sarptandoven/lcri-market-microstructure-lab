from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

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


def add_queue_position_order_size_features(
    frame: pd.DataFrame,
    *,
    levels: int = 5,
    order_size_fraction: float = 0.10,
    bid_order_size_col: str | None = None,
    ask_order_size_col: str | None = None,
) -> pd.DataFrame:
    """Add child-order-size-aware passive queue clearance features.

    ``add_queue_position_features`` estimates queue ahead for a passive order. Real
    passive execution also depends on the order's own displayed size: a larger
    child order needs queue-ahead *plus* child size to clear before it receives a
    full fill. This helper keeps the snapshot approximation transparent by either
    taking explicit bid/ask child order columns or, when absent, sizing each child
    order as a fraction of the current best displayed size.
    """
    if levels < 1:
        raise ValueError("levels must be positive")
    if not math.isfinite(order_size_fraction):
        raise ValueError("order_size_fraction must be finite")
    if order_size_fraction < 0.0:
        raise ValueError("order_size_fraction must be non-negative")

    required = {"bid_queue_ahead", "ask_queue_ahead"}
    required.update({f"bid_sz_{level}" for level in range(1, levels + 1)})
    required.update({f"ask_sz_{level}" for level in range(1, levels + 1)})
    if bid_order_size_col is not None:
        required.add(bid_order_size_col)
    if ask_order_size_col is not None:
        required.add(ask_order_size_col)
    _require_columns(frame, required, "queue position order size")

    values = _finite_values(frame, sorted(required), "queue position order size")
    if (values < 0.0).any().any():
        raise ValueError("queue position order sizes must be non-negative")

    output = frame.copy()
    bid_total = pd.Series(0.0, index=frame.index)
    ask_total = pd.Series(0.0, index=frame.index)
    for level in range(1, levels + 1):
        bid_total = bid_total + values[f"bid_sz_{level}"]
        ask_total = ask_total + values[f"ask_sz_{level}"]

    if bid_order_size_col is None:
        bid_order_size = values["bid_sz_1"] * order_size_fraction
    else:
        bid_order_size = values[bid_order_size_col]
    if ask_order_size_col is None:
        ask_order_size = values["ask_sz_1"] * order_size_fraction
    else:
        ask_order_size = values[ask_order_size_col]

    bid_clear_size = values["bid_queue_ahead"] + bid_order_size
    ask_clear_size = values["ask_queue_ahead"] + ask_order_size
    output["bid_child_order_size"] = bid_order_size
    output["ask_child_order_size"] = ask_order_size
    output["bid_queue_clear_size"] = bid_clear_size
    output["ask_queue_clear_size"] = ask_clear_size
    output["bid_order_size_share"] = np.divide(
        bid_order_size,
        bid_total.replace(0.0, np.nan),
    ).fillna(0.0)
    output["ask_order_size_share"] = np.divide(
        ask_order_size,
        ask_total.replace(0.0, np.nan),
    ).fillna(0.0)
    output["bid_queue_clear_share"] = np.divide(
        bid_clear_size,
        bid_total.replace(0.0, np.nan),
    ).fillna(0.0)
    output["ask_queue_clear_share"] = np.divide(
        ask_clear_size,
        ask_total.replace(0.0, np.nan),
    ).fillna(0.0)
    output["queue_clear_size_imbalance"] = bid_clear_size - ask_clear_size
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
    clear_size_columns = {"bid_queue_clear_size", "ask_queue_clear_size"}
    if clear_size_columns & set(frame.columns):
        required.update(clear_size_columns)
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
    if clear_size_columns.issubset(required):
        size_columns.extend(["bid_queue_clear_size", "ask_queue_clear_size"])
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
    if {"bid_queue_clear_size", "ask_queue_clear_size"}.issubset(values.columns):
        bid_queue = values["bid_queue_clear_size"]
        ask_queue = values["ask_queue_clear_size"]
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
    clear_size_columns = {"bid_queue_clear_size", "ask_queue_clear_size"}
    if clear_size_columns & set(snapshots.columns):
        snapshot_required.update(clear_size_columns)
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

    snapshot_numeric_columns = [timestamp_col, "bid_px_1", "ask_px_1", "bid_queue_ahead", "ask_queue_ahead"]
    if clear_size_columns.issubset(snapshot_required):
        snapshot_numeric_columns.extend(["bid_queue_clear_size", "ask_queue_clear_size"])
    snapshot_numeric = _finite_values(
        snapshots,
        snapshot_numeric_columns,
        "event-level realized fill snapshot",
    )
    event_numeric = _finite_values(
        events,
        [timestamp_col, event_price_col, event_size_col],
        "event-level realized fill event",
    )
    snapshot_size_columns = ["bid_queue_ahead", "ask_queue_ahead"]
    if clear_size_columns.issubset(snapshot_required):
        snapshot_size_columns.extend(["bid_queue_clear_size", "ask_queue_clear_size"])
    if (snapshot_numeric[snapshot_size_columns] < 0.0).any().any():
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
    if {"bid_queue_clear_size", "ask_queue_clear_size"}.issubset(snapshot_numeric.columns):
        bid_queue = snapshot_numeric["bid_queue_clear_size"]
        ask_queue = snapshot_numeric["ask_queue_clear_size"]
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


def add_event_level_trade_confirmed_fill_proxy(
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
    bid_realized_col: str = "bid_trade_confirmed_fill",
    ask_realized_col: str = "ask_trade_confirmed_fill",
    trade_event_types: tuple[str, ...] = ("trade",),
    cancel_event_types: tuple[str, ...] = ("cancel", "delete"),
    bid_trade_sides: tuple[str, ...] = ("sell", "bid"),
    ask_trade_sides: tuple[str, ...] = ("buy", "ask"),
    bid_cancel_sides: tuple[str, ...] = ("bid",),
    ask_cancel_sides: tuple[str, ...] = ("ask",),
) -> pd.DataFrame:
    """Label passive fills only when queue advancement is trade-confirmed.

    ``add_event_level_realized_fill_proxy`` treats both trades and cancels as
    queue-depleting evidence. This stricter companion separates trade depletion
    from cancel depletion and only marks a fill when the cumulative same-price
    queue advance reaches the order's queue threshold on, or before, a confirming
    trade. Rows whose queue would clear from cancels alone are surfaced through
    ``*_queue_advance_without_trade`` so execution-adjusted LCRI audits can spot
    optimistic cancel-only passive-fill labels before publication.
    """
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be a finite positive value")
    grouping_columns = _normalize_group_columns(snapshots, group_cols, "event-level trade-confirmed snapshot")

    snapshot_required = {
        timestamp_col,
        "bid_px_1",
        "ask_px_1",
        "bid_queue_ahead",
        "ask_queue_ahead",
        *grouping_columns,
    }
    clear_size_columns = {"bid_queue_clear_size", "ask_queue_clear_size"}
    if clear_size_columns & set(snapshots.columns):
        snapshot_required.update(clear_size_columns)
    event_required = {
        timestamp_col,
        event_type_col,
        event_side_col,
        event_price_col,
        event_size_col,
        *grouping_columns,
    }
    _require_columns(snapshots, snapshot_required, "event-level trade-confirmed snapshot")
    _require_columns(events, event_required, "event-level trade-confirmed event")

    output = snapshots.copy()
    output_columns = [
        "bid_event_trade_depletion",
        "ask_event_trade_depletion",
        "bid_event_cancel_depletion",
        "ask_event_cancel_depletion",
        "bid_event_total_queue_advance",
        "ask_event_total_queue_advance",
        bid_realized_col,
        ask_realized_col,
        "bid_queue_advance_without_trade",
        "ask_queue_advance_without_trade",
    ]
    for column in output_columns:
        output[column] = pd.Series(0.0, index=snapshots.index)
    output["bid_trade_confirmed_fill_latency"] = pd.Series(np.nan, index=snapshots.index)
    output["ask_trade_confirmed_fill_latency"] = pd.Series(np.nan, index=snapshots.index)
    if snapshots.empty:
        return output

    snapshot_numeric_columns = [timestamp_col, "bid_px_1", "ask_px_1", "bid_queue_ahead", "ask_queue_ahead"]
    if clear_size_columns.issubset(snapshot_required):
        snapshot_numeric_columns.extend(["bid_queue_clear_size", "ask_queue_clear_size"])
    snapshot_numeric = _finite_values(
        snapshots,
        snapshot_numeric_columns,
        "event-level trade-confirmed snapshot",
    )
    event_numeric = _finite_values(
        events,
        [timestamp_col, event_price_col, event_size_col],
        "event-level trade-confirmed event",
    )
    snapshot_size_columns = ["bid_queue_ahead", "ask_queue_ahead"]
    if clear_size_columns.issubset(snapshot_required):
        snapshot_size_columns.extend(["bid_queue_clear_size", "ask_queue_clear_size"])
    if (snapshot_numeric[snapshot_size_columns] < 0.0).any().any():
        raise ValueError("event-level trade-confirmed snapshot queues must be non-negative")
    if (event_numeric[event_size_col] < 0.0).any():
        raise ValueError("event-level trade-confirmed event sizes must be non-negative")

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
    bid_trades = event_types.isin(trade_type_aliases) & event_sides.isin(bid_trade_side_aliases)
    ask_trades = event_types.isin(trade_type_aliases) & event_sides.isin(ask_trade_side_aliases)
    bid_cancels = event_types.isin(cancel_type_aliases) & event_sides.isin(bid_cancel_side_aliases)
    ask_cancels = event_types.isin(cancel_type_aliases) & event_sides.isin(ask_cancel_side_aliases)

    bid_queue = snapshot_numeric["bid_queue_ahead"]
    ask_queue = snapshot_numeric["ask_queue_ahead"]
    if {"bid_queue_clear_size", "ask_queue_clear_size"}.issubset(snapshot_numeric.columns):
        bid_queue = snapshot_numeric["bid_queue_clear_size"]
        ask_queue = snapshot_numeric["ask_queue_clear_size"]

    def scan_side(row: pd.Series, side: str, threshold: float) -> tuple[float, float, float, float, float]:
        start = float(row[timestamp_col])
        event_window = (event_numeric[timestamp_col] > start) & (event_numeric[timestamp_col] <= start + horizon)
        if grouping_columns:
            for group_col in grouping_columns:
                event_window = event_window & (events[group_col] == row[group_col])
        price_col = "bid_px_1" if side == "bid" else "ask_px_1"
        at_price = event_numeric[event_price_col] == float(row[price_col])
        if side == "bid":
            trade_mask = bid_trades
            cancel_mask = bid_cancels
        else:
            trade_mask = ask_trades
            cancel_mask = ask_cancels
        side_events = events.loc[event_window & at_price & (trade_mask | cancel_mask)].copy()
        if side_events.empty:
            return 0.0, 0.0, 0.0, 0.0, math.nan
        side_events = side_events.assign(
            _event_time=event_numeric.loc[side_events.index, timestamp_col],
            _event_size=event_numeric.loc[side_events.index, event_size_col],
            _is_trade=trade_mask.loc[side_events.index].to_numpy(),
        ).sort_values("_event_time")
        trade_depletion = 0.0
        cancel_depletion = 0.0
        cumulative_advance = 0.0
        fill_latency = math.nan
        trade_confirmed = 0.0
        for event in side_events[["_event_time", "_event_size", "_is_trade"]].to_dict("records"):
            size = float(event["_event_size"])
            cumulative_advance += size
            if bool(event["_is_trade"]):
                trade_depletion += size
                if trade_confirmed == 0.0 and (cumulative_advance >= threshold if threshold > 0.0 else cumulative_advance > 0.0):
                    trade_confirmed = 1.0
                    fill_latency = float(event["_event_time"]) - start
            else:
                cancel_depletion += size
        total_advance = trade_depletion + cancel_depletion
        return trade_depletion, cancel_depletion, total_advance, trade_confirmed, fill_latency if trade_confirmed else math.nan

    for index, row in snapshots.iterrows():
        bid_trade, bid_cancel, bid_total, bid_fill, bid_latency = scan_side(row, "bid", float(bid_queue.loc[index]))
        ask_trade, ask_cancel, ask_total, ask_fill, ask_latency = scan_side(row, "ask", float(ask_queue.loc[index]))
        output.loc[index, "bid_event_trade_depletion"] = bid_trade
        output.loc[index, "bid_event_cancel_depletion"] = bid_cancel
        output.loc[index, "bid_event_total_queue_advance"] = bid_total
        output.loc[index, bid_realized_col] = bid_fill
        output.loc[index, "bid_trade_confirmed_fill_latency"] = bid_latency
        output.loc[index, "bid_queue_advance_without_trade"] = float(
            bid_fill == 0.0 and (bid_total >= float(bid_queue.loc[index]) if float(bid_queue.loc[index]) > 0.0 else bid_total > 0.0)
        )
        output.loc[index, "ask_event_trade_depletion"] = ask_trade
        output.loc[index, "ask_event_cancel_depletion"] = ask_cancel
        output.loc[index, "ask_event_total_queue_advance"] = ask_total
        output.loc[index, ask_realized_col] = ask_fill
        output.loc[index, "ask_trade_confirmed_fill_latency"] = ask_latency
        output.loc[index, "ask_queue_advance_without_trade"] = float(
            ask_fill == 0.0 and (ask_total >= float(ask_queue.loc[index]) if float(ask_queue.loc[index]) > 0.0 else ask_total > 0.0)
        )
    return output


def trade_confirmed_passive_fill_latency_summary(
    frame: pd.DataFrame,
    *,
    fill_cols: tuple[str, ...] = ("bid_trade_confirmed_fill", "ask_trade_confirmed_fill"),
    latency_cols: tuple[str, ...] = ("bid_trade_confirmed_fill_latency", "ask_trade_confirmed_fill_latency"),
    cancel_only_cols: tuple[str, ...] = ("bid_queue_advance_without_trade", "ask_queue_advance_without_trade"),
    trade_depletion_cols: tuple[str, ...] = ("bid_event_trade_depletion", "ask_event_trade_depletion"),
    cancel_depletion_cols: tuple[str, ...] = ("bid_event_cancel_depletion", "ask_event_cancel_depletion"),
    sides: tuple[str, ...] = ("bid", "ask"),
    max_mean_latency: float = 0.50,
    max_cancel_only_clear_rate: float = 0.05,
) -> pd.DataFrame:
    """Summarize trade-confirmed passive-fill latency and cancel-only clearance risk.

    This diagnostic is designed for rows produced by
    ``add_event_level_trade_confirmed_fill_proxy``. It quantifies whether passive
    fills are promptly trade-confirmed or mostly inferred from cancel-only queue
    advancement, which is a key publishability risk for queue-position-aware
    execution-adjusted LCRI.
    """
    column_groups = [
        fill_cols,
        latency_cols,
        cancel_only_cols,
        trade_depletion_cols,
        cancel_depletion_cols,
        sides,
    ]
    lengths = {len(group) for group in column_groups}
    if lengths != {len(sides)} or not sides:
        raise ValueError("all side column tuples must be non-empty and have the same length")
    if not math.isfinite(max_mean_latency) or max_mean_latency < 0.0:
        raise ValueError("max_mean_latency must be finite and non-negative")
    if not math.isfinite(max_cancel_only_clear_rate) or not 0.0 <= max_cancel_only_clear_rate <= 1.0:
        raise ValueError("max_cancel_only_clear_rate must be finite and in [0, 1]")

    required = set(fill_cols) | set(latency_cols) | set(cancel_only_cols) | set(trade_depletion_cols) | set(cancel_depletion_cols)
    _require_columns(frame, required, "trade-confirmed passive fill latency summary")
    columns = [
        "side",
        "rows",
        "trade_confirmed_fill_rate",
        "cancel_only_clear_rate",
        "mean_fill_latency",
        "p95_fill_latency",
        "mean_trade_depletion",
        "mean_cancel_depletion",
        "review_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    fill_values = _finite_values(frame, list(fill_cols), "trade-confirmed passive fill latency summary")
    cancel_only_values = _finite_values(frame, list(cancel_only_cols), "trade-confirmed passive fill latency summary")
    depletion_values = _finite_values(
        frame,
        list(trade_depletion_cols) + list(cancel_depletion_cols),
        "trade-confirmed passive fill latency summary",
    )
    if not fill_values.apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("trade-confirmed passive fill latency summary fill flags must be in [0, 1]")
    if not cancel_only_values.apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("trade-confirmed passive fill latency summary cancel-only flags must be in [0, 1]")
    if (depletion_values < 0.0).any().any():
        raise ValueError("trade-confirmed passive fill latency summary depletions must be non-negative")

    latency_values = pd.DataFrame(index=frame.index)
    for latency_col in latency_cols:
        latency = pd.to_numeric(frame[latency_col], errors="coerce")
        if (latency.dropna() < 0.0).any() or not np.isfinite(latency.dropna().to_numpy()).all():
            raise ValueError("trade-confirmed passive fill latency summary latencies must be non-negative")
        latency_values[latency_col] = latency

    def summarize(
        side: str,
        fill: pd.Series,
        latency: pd.Series,
        cancel_only: pd.Series,
        trade_depletion: pd.Series,
        cancel_depletion: pd.Series,
    ) -> dict[str, float | int | str]:
        fill_bool = fill >= 0.5
        rows = int(len(fill))
        filled_latency = latency[fill_bool].dropna()
        mean_latency = float(filled_latency.mean()) if not filled_latency.empty else 0.0
        p95_latency = float(filled_latency.quantile(0.95)) if not filled_latency.empty else 0.0
        cancel_only_rate = float((cancel_only >= 0.5).mean())
        latency_risk = mean_latency > max_mean_latency
        cancel_risk = cancel_only_rate > max_cancel_only_clear_rate
        if latency_risk and cancel_risk:
            label = "cancel_only_and_latency_risk"
        elif cancel_risk:
            label = "cancel_only_clear_risk"
        elif latency_risk:
            label = "latency_risk"
        else:
            label = "trade_confirmed_execution_ok"
        return {
            "side": side,
            "rows": rows,
            "trade_confirmed_fill_rate": float(fill_bool.mean()),
            "cancel_only_clear_rate": cancel_only_rate,
            "mean_fill_latency": mean_latency,
            "p95_fill_latency": p95_latency,
            "mean_trade_depletion": float(trade_depletion.mean()),
            "mean_cancel_depletion": float(cancel_depletion.mean()),
            "review_label": label,
        }

    rows = [
        summarize(
            side,
            fill_values[fill_col],
            latency_values[latency_col],
            cancel_only_values[cancel_only_col],
            depletion_values[trade_depletion_col],
            depletion_values[cancel_depletion_col],
        )
        for side, fill_col, latency_col, cancel_only_col, trade_depletion_col, cancel_depletion_col in zip(
            sides,
            fill_cols,
            latency_cols,
            cancel_only_cols,
            trade_depletion_cols,
            cancel_depletion_cols,
        )
    ]
    rows.append(
        summarize(
            "all",
            pd.concat([fill_values[column] for column in fill_cols], ignore_index=True),
            pd.concat([latency_values[column] for column in latency_cols], ignore_index=True),
            pd.concat([cancel_only_values[column] for column in cancel_only_cols], ignore_index=True),
            pd.concat([depletion_values[column] for column in trade_depletion_cols], ignore_index=True),
            pd.concat([depletion_values[column] for column in cancel_depletion_cols], ignore_index=True),
        )
    )
    return pd.DataFrame(rows, columns=columns)


def queue_position_trade_confirmation_competing_risk_curve(
    frame: pd.DataFrame,
    *,
    fill_cols: tuple[str, ...] = ("bid_trade_confirmed_fill", "ask_trade_confirmed_fill"),
    latency_cols: tuple[str, ...] = ("bid_trade_confirmed_fill_latency", "ask_trade_confirmed_fill_latency"),
    cancel_only_cols: tuple[str, ...] = ("bid_queue_advance_without_trade", "ask_queue_advance_without_trade"),
    sides: tuple[str, ...] = ("bid", "ask"),
    latency_thresholds: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
    max_cancel_only_clear_rate: float = 0.05,
    max_late_trade_confirmed_rate: float = 0.10,
) -> pd.DataFrame:
    """Cumulative competing-risk curve for passive-fill confirmation latency.

    Rows produced by ``add_event_level_trade_confirmed_fill_proxy`` can end as
    prompt trade-confirmed fills, late trade-confirmed fills, cancel-only queue
    clears, or unresolved opportunities. This curve evaluates those outcomes at
    latency cutoffs so publication reviews can see whether execution-adjusted LCRI
    survives realistic reaction-time budgets rather than only an end-of-window fill
    label.
    """
    column_groups = [fill_cols, latency_cols, cancel_only_cols, sides]
    lengths = {len(group) for group in column_groups}
    if lengths != {len(sides)} or not sides:
        raise ValueError("all side column tuples must be non-empty and have the same length")
    thresholds = [float(threshold) for threshold in latency_thresholds]
    if not thresholds:
        raise ValueError("latency_thresholds must be non-empty")
    if any(not math.isfinite(threshold) or threshold < 0.0 for threshold in thresholds):
        raise ValueError("latency_thresholds must be finite non-negative values")
    if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("latency_thresholds must be strictly increasing")
    if not math.isfinite(max_cancel_only_clear_rate) or not 0.0 <= max_cancel_only_clear_rate <= 1.0:
        raise ValueError("max_cancel_only_clear_rate must be finite and in [0, 1]")
    if not math.isfinite(max_late_trade_confirmed_rate) or not 0.0 <= max_late_trade_confirmed_rate <= 1.0:
        raise ValueError("max_late_trade_confirmed_rate must be finite and in [0, 1]")

    required = set(fill_cols) | set(latency_cols) | set(cancel_only_cols)
    _require_columns(frame, required, "queue-position trade confirmation competing risk curve")
    columns = [
        "side",
        "latency_threshold",
        "rows",
        "trade_confirmed_by_threshold_rate",
        "late_trade_confirmed_rate",
        "cancel_only_clear_rate",
        "unresolved_rate",
        "competing_risk_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    fill_values = _finite_values(frame, list(fill_cols), "queue-position trade confirmation competing risk curve")
    cancel_values = _finite_values(frame, list(cancel_only_cols), "queue-position trade confirmation competing risk curve")
    if not fill_values.apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue-position trade confirmation competing risk fill flags must be in [0, 1]")
    if not cancel_values.apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue-position trade confirmation competing risk cancel-only flags must be in [0, 1]")

    latency_values = pd.DataFrame(index=frame.index)
    for latency_col in latency_cols:
        latency = pd.to_numeric(frame[latency_col], errors="coerce")
        observed = latency.dropna()
        if (observed < 0.0).any() or not np.isfinite(observed.to_numpy()).all():
            raise ValueError("queue-position trade confirmation competing risk latencies must be non-negative")
        latency_values[latency_col] = latency

    def summarize_side(side: str, fill: pd.Series, latency: pd.Series, cancel_only: pd.Series) -> list[dict[str, float | int | str]]:
        fill_bool = fill >= 0.5
        cancel_bool = cancel_only >= 0.5
        rows = int(len(fill))
        summaries: list[dict[str, float | int | str]] = []
        for threshold in thresholds:
            prompt_fill = fill_bool & latency.le(threshold).fillna(False)
            late_fill = fill_bool & ~prompt_fill
            unresolved = ~(prompt_fill | late_fill | cancel_bool)
            late_rate = float(late_fill.mean())
            cancel_rate = float(cancel_bool.mean())
            if cancel_rate > max_cancel_only_clear_rate and late_rate > max_late_trade_confirmed_rate:
                label = "cancel_only_and_late_confirmation_risk"
            elif cancel_rate > max_cancel_only_clear_rate:
                label = "cancel_only_risk"
            elif late_rate > max_late_trade_confirmed_rate:
                label = "late_confirmation_risk"
            else:
                label = "trade_confirmation_curve_ok"
            summaries.append(
                {
                    "side": side,
                    "latency_threshold": threshold,
                    "rows": rows,
                    "trade_confirmed_by_threshold_rate": float(prompt_fill.mean()),
                    "late_trade_confirmed_rate": late_rate,
                    "cancel_only_clear_rate": cancel_rate,
                    "unresolved_rate": float(unresolved.mean()),
                    "competing_risk_label": label,
                }
            )
        return summaries

    rows: list[dict[str, float | int | str]] = []
    for side, fill_col, latency_col, cancel_col in zip(sides, fill_cols, latency_cols, cancel_only_cols):
        rows.extend(summarize_side(side, fill_values[fill_col], latency_values[latency_col], cancel_values[cancel_col]))
    rows.extend(
        summarize_side(
            "all",
            pd.concat([fill_values[column] for column in fill_cols], ignore_index=True),
            pd.concat([latency_values[column] for column in latency_cols], ignore_index=True),
            pd.concat([cancel_values[column] for column in cancel_only_cols], ignore_index=True),
        )
    )
    return pd.DataFrame(rows, columns=columns)


def passive_fill_proxy_disagreement(
    frame: pd.DataFrame,
    *,
    snapshot_cols: tuple[str, ...] = ("bid_snapshot_fill", "ask_snapshot_fill"),
    event_cols: tuple[str, ...] = ("bid_event_fill", "ask_event_fill"),
    sides: tuple[str, ...] = ("bid", "ask"),
    max_disagreement_rate: float = 0.10,
) -> pd.DataFrame:
    """Audit snapshot passive-fill labels against event-level queue depletion labels.

    Snapshot depletion proxies are useful when only L2 snapshots are available, but
    they can overstate fills when cancellations move displayed size without queue
    priority reaching the child order, or understate fills that happen and refill
    between snapshots. This diagnostic compares paired snapshot/event realized-fill
    labels by side plus an aggregate row, making the proxy's censoring bias explicit
    before using it for queue-position calibration or execution-adjusted LCRI.
    """
    if len(snapshot_cols) != len(event_cols) or len(snapshot_cols) != len(sides):
        raise ValueError("snapshot_cols, event_cols, and sides must have the same length")
    if not snapshot_cols:
        raise ValueError("at least one fill-label column pair is required")
    if not math.isfinite(max_disagreement_rate) or not 0.0 <= max_disagreement_rate <= 1.0:
        raise ValueError("max_disagreement_rate must be finite and in [0, 1]")
    required = set(snapshot_cols) | set(event_cols)
    _require_columns(frame, required, "passive fill proxy disagreement")

    columns = [
        "side",
        "rows",
        "snapshot_fill_rate",
        "event_fill_rate",
        "agreement_rate",
        "disagreement_rate",
        "false_positive_rate",
        "false_negative_rate",
        "precision",
        "recall",
        "snapshot_event_fill_bias",
        "review_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    values = _finite_values(frame, sorted(required), "passive fill proxy disagreement")
    if not values.apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("passive fill proxy disagreement fill labels must be in [0, 1]")

    def summarize(side: str, snapshot: pd.Series, event: pd.Series) -> dict[str, float | int | str]:
        snapshot_bool = snapshot >= 0.5
        event_bool = event >= 0.5
        rows = int(len(snapshot_bool))
        true_positive = snapshot_bool & event_bool
        false_positive = snapshot_bool & ~event_bool
        false_negative = ~snapshot_bool & event_bool
        disagreement = false_positive | false_negative
        snapshot_positive = int(snapshot_bool.sum())
        event_positive = int(event_bool.sum())
        precision = 1.0 if snapshot_positive == 0 else float(true_positive.sum() / snapshot_positive)
        recall = 1.0 if event_positive == 0 else float(true_positive.sum() / event_positive)
        false_positive_rate = float(false_positive.mean())
        false_negative_rate = float(false_negative.mean())
        disagreement_rate = float(disagreement.mean())
        bias = float(snapshot_bool.mean() - event_bool.mean())
        if disagreement_rate <= max_disagreement_rate:
            label = "proxy_event_aligned"
        elif false_positive_rate > false_negative_rate:
            label = "proxy_event_false_positive_bias"
        elif false_negative_rate > false_positive_rate:
            label = "proxy_event_false_negative_bias"
        else:
            label = "proxy_event_disagreement"
        return {
            "side": side,
            "rows": rows,
            "snapshot_fill_rate": float(snapshot_bool.mean()),
            "event_fill_rate": float(event_bool.mean()),
            "agreement_rate": float(1.0 - disagreement_rate),
            "disagreement_rate": disagreement_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "precision": precision,
            "recall": recall,
            "snapshot_event_fill_bias": bias,
            "review_label": label,
        }

    rows = [
        summarize(side, values[snapshot_col], values[event_col])
        for side, snapshot_col, event_col in zip(sides, snapshot_cols, event_cols)
    ]
    stacked_snapshot = pd.concat([values[column] for column in snapshot_cols], ignore_index=True)
    stacked_event = pd.concat([values[column] for column in event_cols], ignore_index=True)
    rows.append(summarize("all", stacked_snapshot, stacked_event))
    return pd.DataFrame(rows, columns=columns)


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
    if {"bid_queue_clear_share", "ask_queue_clear_share"}.issubset(frame.columns):
        clear_values = _finite_values(
            frame,
            ["bid_queue_clear_share", "ask_queue_clear_share"],
            "fill probability queue clearance",
        )
        bid_queue_share = clear_values["bid_queue_clear_share"].clip(lower=0.0)
        ask_queue_share = clear_values["ask_queue_clear_share"].clip(lower=0.0)
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


def add_latency_adjusted_passive_fill_probabilities(
    frame: pd.DataFrame,
    *,
    latency_steps: float = 1.0,
    latency_col: str | None = None,
    decay_scale: float = 1.0,
    adverse_selection_scale: float = 0.50,
    config: FillProbabilityConfig | None = None,
) -> pd.DataFrame:
    """Discount passive fill odds for stale queue-position decisions.

    ``add_passive_fill_probabilities`` prices the queue state at the observed
    snapshot. In a live passive strategy the child order joins after model,
    routing, and exchange latency. This helper applies a transparent survival
    discount to fill probabilities and a matching adverse-selection uplift using
    queue-clearance share, volatility, spread, and replenishment stress. The
    resulting columns can be fed into execution-adjusted LCRI reviews without
    pretending zero-latency fills are deployable.
    """
    config = config or FillProbabilityConfig()
    if not math.isfinite(latency_steps):
        raise ValueError("latency_steps must be finite")
    if latency_steps < 0.0:
        raise ValueError("latency values must be non-negative")
    if not math.isfinite(decay_scale) or decay_scale < 0.0:
        raise ValueError("decay_scale must be finite and non-negative")
    if not math.isfinite(adverse_selection_scale) or adverse_selection_scale < 0.0:
        raise ValueError("adverse_selection_scale must be finite and non-negative")

    queue_columns = ("bid_queue_clear_share", "ask_queue_clear_share")
    if not set(queue_columns).issubset(frame.columns):
        queue_columns = ("bid_queue_share", "ask_queue_share")
    required = {
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "spread_ticks",
        "volatility",
        "replenishment_rate",
        *queue_columns,
    }
    if latency_col is not None:
        required.add(latency_col)
    _require_columns(frame, required, "latency-adjusted passive fill probability")

    numeric_columns = sorted(required)
    values = _finite_values(frame, numeric_columns, "latency-adjusted passive fill probability")
    if latency_col is None:
        latency = pd.Series(float(latency_steps), index=frame.index)
    else:
        latency = values[latency_col]
    if (latency < 0.0).any():
        raise ValueError("latency values must be non-negative")

    bid_queue = values[queue_columns[0]].clip(lower=0.0)
    ask_queue = values[queue_columns[1]].clip(lower=0.0)
    spread_stress = np.log1p(values["spread_ticks"].clip(lower=0.0))
    volatility_stress = np.log1p(values["volatility"].clip(lower=0.0))
    thin_book_stress = 1.0 / (1.0 + values["replenishment_rate"].clip(lower=0.0))
    common_stress = 0.25 * spread_stress + 0.35 * volatility_stress + 0.35 * thin_book_stress

    bid_latency_risk = latency * decay_scale * (bid_queue + common_stress)
    ask_latency_risk = latency * decay_scale * (ask_queue + common_stress)
    bid_survival = np.exp(-bid_latency_risk.clip(0.0, 40.0))
    ask_survival = np.exp(-ask_latency_risk.clip(0.0, 40.0))

    bid_fill = values["bid_fill_probability"].clip(0.0, 1.0)
    ask_fill = values["ask_fill_probability"].clip(0.0, 1.0)
    bid_adverse = values["bid_adverse_fill_probability"].clip(0.0, 1.0)
    ask_adverse = values["ask_adverse_fill_probability"].clip(0.0, 1.0)

    bid_adjusted_fill = _clip_probability(bid_fill * bid_survival, config)
    ask_adjusted_fill = _clip_probability(ask_fill * ask_survival, config)
    bid_adjusted_adverse = _clip_probability(
        bid_adverse + (1.0 - bid_adverse) * (1.0 - bid_survival) * adverse_selection_scale,
        config,
    )
    ask_adjusted_adverse = _clip_probability(
        ask_adverse + (1.0 - ask_adverse) * (1.0 - ask_survival) * adverse_selection_scale,
        config,
    )

    output = frame.copy()
    output["latency_steps"] = latency
    output["bid_latency_risk"] = bid_latency_risk
    output["ask_latency_risk"] = ask_latency_risk
    output["bid_latency_survival"] = bid_survival
    output["ask_latency_survival"] = ask_survival
    output["bid_latency_adjusted_fill_probability"] = bid_adjusted_fill
    output["ask_latency_adjusted_fill_probability"] = ask_adjusted_fill
    output["bid_latency_adjusted_adverse_fill_probability"] = bid_adjusted_adverse
    output["ask_latency_adjusted_adverse_fill_probability"] = ask_adjusted_adverse
    output["bid_latency_adjusted_fill_minus_adverse"] = bid_adjusted_fill - bid_adjusted_adverse
    output["ask_latency_adjusted_fill_minus_adverse"] = ask_adjusted_fill - ask_adjusted_adverse
    return output


def add_execution_adjusted_edge(
    frame: pd.DataFrame,
    *,
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
    long_net_col: str = "long_net_return_ticks",
    short_net_col: str = "short_net_return_ticks",
    passive_spread_capture_ticks: float = 0.0,
    maker_rebate_ticks: float = 0.0,
    adverse_slippage_ticks: float = 0.0,
) -> pd.DataFrame:
    """Convert directional LCRI edge into passive-fill-adjusted tradable edge.

    ``long_net_col`` and ``short_net_col`` encode the directional markout/label in
    ticks. Optional passive economics let reviewers stress execution-adjusted LCRI
    under venue-specific maker spread capture, rebates, and extra adverse-fill
    slippage without rebuilding labels.
    """
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

    for name, value in {
        "passive_spread_capture_ticks": passive_spread_capture_ticks,
        "maker_rebate_ticks": maker_rebate_ticks,
        "adverse_slippage_ticks": adverse_slippage_ticks,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")

    signal = values[signal_col]
    long_return = values[long_net_col]
    short_return = values[short_net_col]
    bid_fill = values["bid_fill_probability"].clip(0.0, 1.0)
    ask_fill = values["ask_fill_probability"].clip(0.0, 1.0)
    bid_adverse = values["bid_adverse_fill_probability"].clip(0.0, 1.0)
    ask_adverse = values["ask_adverse_fill_probability"].clip(0.0, 1.0)

    passive_uplift = passive_spread_capture_ticks + maker_rebate_ticks
    long_edge = bid_fill * (long_return + passive_uplift) - bid_adverse * (
        long_return.abs() + adverse_slippage_ticks
    )
    short_edge = ask_fill * (short_return + passive_uplift) - ask_adverse * (
        short_return.abs() + adverse_slippage_ticks
    )
    best_edge = np.maximum(long_edge, short_edge)
    best_side = np.select(
        [(long_edge > 0.0) & (long_edge >= short_edge), (short_edge > 0.0) & (short_edge > long_edge)],
        ["long", "short"],
        default="abstain",
    )
    raw_side = np.select(
        [signal > 0.0, signal < 0.0],
        ["long", "short"],
        default="neutral",
    )
    side_aligned = (best_side == raw_side) & (raw_side != "neutral")
    side_alignment = np.select(
        [raw_side == "neutral", best_side == "abstain", side_aligned],
        ["neutral", "abstain", "aligned"],
        default="inverted",
    )

    output = frame.copy()
    output["long_fill_adjusted_edge_ticks"] = long_edge
    output["short_fill_adjusted_edge_ticks"] = short_edge
    output["best_execution_side"] = best_side
    output["execution_lcri_side_alignment"] = side_alignment
    output["execution_adjusted_edge_ticks"] = best_edge
    output["execution_adjusted_lcri_score"] = np.where(side_aligned, signal, 0.0)
    return output


def execution_adjusted_edge_venue_economics_sensitivity(
    frame: pd.DataFrame,
    *,
    scenarios: dict[str, Any] | None = None,
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
    long_net_col: str = "long_net_return_ticks",
    short_net_col: str = "short_net_return_ticks",
    min_mean_edge_ticks: float = 0.0,
) -> pd.DataFrame:
    """Reprice execution-adjusted LCRI under venue economics scenarios.

    Each scenario is a ``(passive_spread_capture_ticks, maker_rebate_ticks,
    adverse_slippage_ticks)`` tuple passed through ``add_execution_adjusted_edge``.
    The resulting table turns venue fee/rebate and adverse-fill assumptions into a
    compact sensitivity artifact: reviewers can see whether queue-aware LCRI stays
    tradable after maker economics, or only works in a zero-cost venue abstraction.
    """
    columns = [
        "scenario",
        "passive_spread_capture_ticks",
        "maker_rebate_ticks",
        "adverse_slippage_ticks",
        "rows",
        "tradable_rows",
        "tradable_share",
        "long_share",
        "short_share",
        "abstain_share",
        "positive_edge_share",
        "mean_long_edge_ticks",
        "mean_short_edge_ticks",
        "mean_execution_adjusted_edge_ticks",
        "median_execution_adjusted_edge_ticks",
        "worst_row_edge_ticks",
        "economics_label",
    ]
    if not math.isfinite(min_mean_edge_ticks):
        raise ValueError("min_mean_edge_ticks must be finite")
    if scenarios is None:
        scenarios = {
            "base": (0.0, 0.0, 0.0),
            "maker_rebate": (0.0, 0.1, 0.0),
            "wide_spread_toxic_fill": (0.5, 0.0, 0.25),
        }
    if not scenarios:
        raise ValueError("scenarios must be a non-empty mapping")

    parsed_scenarios: list[tuple[str, float, float, float]] = []
    for scenario, settings in scenarios.items():
        if isinstance(settings, (str, bytes)):
            raise ValueError(
                "scenarios values must be (passive_spread_capture_ticks, maker_rebate_ticks, adverse_slippage_ticks) triples"
            )
        try:
            spread_raw, rebate_raw, slippage_raw = settings
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "scenarios values must be (passive_spread_capture_ticks, maker_rebate_ticks, adverse_slippage_ticks) triples"
            ) from exc
        parsed_scenarios.append((str(scenario), float(spread_raw), float(rebate_raw), float(slippage_raw)))

    rows: list[dict[str, float | int | str]] = []
    for scenario, spread_ticks, rebate_ticks, slippage_ticks in parsed_scenarios:
        adjusted = add_execution_adjusted_edge(
            frame,
            signal_col=signal_col,
            probability_col=probability_col,
            long_net_col=long_net_col,
            short_net_col=short_net_col,
            passive_spread_capture_ticks=spread_ticks,
            maker_rebate_ticks=rebate_ticks,
            adverse_slippage_ticks=slippage_ticks,
        )
        if adjusted.empty:
            rows.append(
                {
                    "scenario": scenario,
                    "passive_spread_capture_ticks": spread_ticks,
                    "maker_rebate_ticks": rebate_ticks,
                    "adverse_slippage_ticks": slippage_ticks,
                    "rows": 0,
                    "tradable_rows": 0,
                    "tradable_share": 0.0,
                    "long_share": 0.0,
                    "short_share": 0.0,
                    "abstain_share": 0.0,
                    "positive_edge_share": 0.0,
                    "mean_long_edge_ticks": 0.0,
                    "mean_short_edge_ticks": 0.0,
                    "mean_execution_adjusted_edge_ticks": 0.0,
                    "median_execution_adjusted_edge_ticks": 0.0,
                    "worst_row_edge_ticks": 0.0,
                    "economics_label": "no_evidence",
                }
            )
            continue

        side = adjusted["best_execution_side"].astype(str)
        edge = adjusted["execution_adjusted_edge_ticks"].astype(float)
        total_rows = int(len(adjusted))
        tradable_rows = int(side.isin(["long", "short"]).sum())
        mean_edge = float(edge.mean())
        positive_edge_share = float((edge > 0.0).mean())
        if mean_edge >= min_mean_edge_ticks and tradable_rows > 0:
            economics_label = "positive_after_costs"
        elif tradable_rows > 0:
            economics_label = "cost_fragile"
        else:
            economics_label = "no_tradable_edge"
        rows.append(
            {
                "scenario": scenario,
                "passive_spread_capture_ticks": spread_ticks,
                "maker_rebate_ticks": rebate_ticks,
                "adverse_slippage_ticks": slippage_ticks,
                "rows": total_rows,
                "tradable_rows": tradable_rows,
                "tradable_share": tradable_rows / total_rows,
                "long_share": float((side == "long").mean()),
                "short_share": float((side == "short").mean()),
                "abstain_share": float((side == "abstain").mean()),
                "positive_edge_share": positive_edge_share,
                "mean_long_edge_ticks": float(adjusted["long_fill_adjusted_edge_ticks"].mean()),
                "mean_short_edge_ticks": float(adjusted["short_fill_adjusted_edge_ticks"].mean()),
                "mean_execution_adjusted_edge_ticks": mean_edge,
                "median_execution_adjusted_edge_ticks": float(edge.median()),
                "worst_row_edge_ticks": float(edge.min()),
                "economics_label": economics_label,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def execution_adjusted_edge_venue_economics_release_scorecard(
    sensitivity: pd.DataFrame,
    *,
    max_fragile_scenario_share: float = 0.34,
    review_fragile_scenario_share: float = 0.10,
    min_weighted_edge_ticks: float = 0.0,
    min_worst_scenario_edge_ticks: float = -0.05,
    min_positive_edge_share: float = 0.50,
) -> dict[str, float | int | bool | str]:
    """Summarize venue-economics sensitivity into a release-facing scorecard.

    ``execution_adjusted_edge_venue_economics_sensitivity`` makes venue fee/rebate
    and adverse-slippage assumptions explicit. This reducer prevents a queue-aware
    LCRI result from being called publishable when only the base venue abstraction
    works: it scenario-weights edge by row counts, tracks fragile venue scenarios,
    and names the weakest fee/slippage assumption for review packets.
    """
    for name, value in {
        "max_fragile_scenario_share": max_fragile_scenario_share,
        "review_fragile_scenario_share": review_fragile_scenario_share,
        "min_weighted_edge_ticks": min_weighted_edge_ticks,
        "min_worst_scenario_edge_ticks": min_worst_scenario_edge_ticks,
        "min_positive_edge_share": min_positive_edge_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_fragile_scenario_share <= 1.0:
        raise ValueError("max_fragile_scenario_share must be in [0, 1]")
    if not 0.0 <= review_fragile_scenario_share <= 1.0:
        raise ValueError("review_fragile_scenario_share must be in [0, 1]")
    if not 0.0 <= min_positive_edge_share <= 1.0:
        raise ValueError("min_positive_edge_share must be in [0, 1]")

    empty: dict[str, float | int | bool | str] = {
        "scenarios": 0,
        "rows": 0,
        "tradable_rows": 0,
        "weighted_tradable_share": 0.0,
        "weighted_positive_edge_share": 0.0,
        "weighted_mean_execution_adjusted_edge_ticks": 0.0,
        "worst_scenario": "none",
        "worst_scenario_edge_ticks": 0.0,
        "worst_row_edge_ticks": 0.0,
        "fragile_scenarios": 0,
        "fragile_scenario_share": 0.0,
        "venue_economics_release_decision": "review",
        "venue_economics_release_label": "venue_economics_no_evidence",
        "publishable": False,
        "blocking_reasons": "none",
        "review_reasons": "no_venue_economics_evidence",
    }
    if sensitivity.empty:
        return empty

    required = {
        "scenario",
        "rows",
        "tradable_rows",
        "tradable_share",
        "positive_edge_share",
        "mean_execution_adjusted_edge_ticks",
        "worst_row_edge_ticks",
        "economics_label",
    }
    _require_columns(sensitivity, required, "execution-adjusted venue economics release scorecard")
    numeric_columns = [
        "rows",
        "tradable_rows",
        "tradable_share",
        "positive_edge_share",
        "mean_execution_adjusted_edge_ticks",
        "worst_row_edge_ticks",
    ]
    values = _finite_values(sensitivity, numeric_columns, "execution-adjusted venue economics release scorecard")
    if (values[["rows", "tradable_rows", "tradable_share", "positive_edge_share"]] < 0.0).any().any():
        raise ValueError("execution-adjusted venue economics scorecard counts and shares must be non-negative")
    if not values[["tradable_share", "positive_edge_share"]].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("execution-adjusted venue economics scorecard shares must be in [0, 1]")
    if (values["tradable_rows"] > values["rows"]).any():
        raise ValueError("execution-adjusted venue economics tradable rows cannot exceed rows")

    working = sensitivity.copy()
    working[numeric_columns] = values
    scenario_count = int(working["scenario"].astype(str).nunique())
    rows = int(values["rows"].sum())
    tradable_rows = int(values["tradable_rows"].sum())
    weights = values["rows"].where(values["rows"] > 0.0, 1.0)
    weighted_tradable_share = float(np.average(values["tradable_share"], weights=weights))
    weighted_positive_share = float(np.average(values["positive_edge_share"], weights=weights))
    weighted_edge = float(np.average(values["mean_execution_adjusted_edge_ticks"], weights=weights))
    worst_idx = values["mean_execution_adjusted_edge_ticks"].idxmin()
    worst_scenario = str(working.loc[worst_idx, "scenario"])
    worst_scenario_edge = float(values.loc[worst_idx, "mean_execution_adjusted_edge_ticks"])
    worst_row_edge = float(values["worst_row_edge_ticks"].min())

    labels = working["economics_label"].astype(str)
    fragile_mask = (
        labels != "positive_after_costs"
    ) | (values["positive_edge_share"] < min_positive_edge_share) | (
        values["mean_execution_adjusted_edge_ticks"] < min_weighted_edge_ticks
    )
    fragile_scenarios = int(fragile_mask.sum())
    fragile_share = fragile_scenarios / len(working)

    blocking_reasons: list[str] = []
    review_reasons: list[str] = []
    if fragile_share > max_fragile_scenario_share:
        blocking_reasons.append("fragile_scenario_share")
    elif fragile_share > review_fragile_scenario_share:
        review_reasons.append("fragile_scenario_share")
    if weighted_edge < min_weighted_edge_ticks:
        blocking_reasons.append("weighted_edge_below_floor")
    if worst_scenario_edge < min_worst_scenario_edge_ticks:
        blocking_reasons.append("worst_scenario_edge_below_floor")
    if weighted_positive_share < min_positive_edge_share:
        review_reasons.append("positive_edge_share_below_review_floor")
    if tradable_rows == 0:
        blocking_reasons.append("no_tradable_rows")

    if blocking_reasons:
        decision = "block"
        label = "venue_economics_not_publishable"
    elif review_reasons:
        decision = "review"
        label = "venue_economics_review"
    else:
        decision = "pass"
        label = "venue_economics_publishable"

    return {
        "scenarios": scenario_count,
        "rows": rows,
        "tradable_rows": tradable_rows,
        "weighted_tradable_share": weighted_tradable_share,
        "weighted_positive_edge_share": weighted_positive_share,
        "weighted_mean_execution_adjusted_edge_ticks": weighted_edge,
        "worst_scenario": worst_scenario,
        "worst_scenario_edge_ticks": worst_scenario_edge,
        "worst_row_edge_ticks": worst_row_edge,
        "fragile_scenarios": fragile_scenarios,
        "fragile_scenario_share": fragile_share,
        "venue_economics_release_decision": decision,
        "venue_economics_release_label": label,
        "publishable": decision == "pass",
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
    }


def execution_adjusted_edge_component_attribution(
    frame: pd.DataFrame,
    *,
    group_cols: str | list[str] | tuple[str, ...] | None = "best_execution_side",
    long_net_col: str = "long_net_return_ticks",
    short_net_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Decompose execution-adjusted edge into raw edge, fill capture, and toxicity drag.

    Queue-aware edge can fail for two distinct reasons: the passive order may not
    get enough queue clearance to capture the paper edge, or the fills it does get
    may be adverse-selection fills. This attribution table makes that distinction
    explicit for release reviews by aggregating the chosen execution side's raw
    directional edge, fill-captured edge, adverse-selection cost, and residual
    execution-adjusted edge across sides, regimes, or caller-provided slices.
    """
    columns = [
        *(_normalize_group_columns(frame, group_cols, "execution edge component attribution") if group_cols is not None and not frame.empty else ([group_cols] if isinstance(group_cols, str) else list(group_cols or []))),
        "rows",
        "mean_raw_edge_ticks",
        "mean_fill_captured_edge_ticks",
        "mean_adverse_selection_cost_ticks",
        "mean_execution_adjusted_edge_ticks",
        "mean_fill_shortfall_ticks",
        "fill_capture_ratio",
        "adverse_drag_ratio",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    grouping_columns = _normalize_group_columns(frame, group_cols, "execution edge component attribution")
    required = {
        "best_execution_side",
        long_net_col,
        short_net_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "long_fill_adjusted_edge_ticks",
        "short_fill_adjusted_edge_ticks",
        "execution_adjusted_edge_ticks",
    }
    _require_columns(frame, required, "execution edge component attribution")
    values = _finite_values(frame, sorted(required - {"best_execution_side"}), "execution edge component attribution")

    side = frame["best_execution_side"].astype(str)
    raw_edge = _side_probability(side, bid=values[long_net_col], ask=values[short_net_col])
    fill_captured = _side_probability(
        side,
        bid=values["bid_fill_probability"] * values[long_net_col],
        ask=values["ask_fill_probability"] * values[short_net_col],
    )
    adverse_cost = _side_probability(
        side,
        bid=values["bid_adverse_fill_probability"] * values[long_net_col].abs(),
        ask=values["ask_adverse_fill_probability"] * values[short_net_col].abs(),
    )
    execution_edge = _side_probability(
        side,
        bid=values["long_fill_adjusted_edge_ticks"],
        ask=values["short_fill_adjusted_edge_ticks"],
    )
    if "execution_adjusted_edge_ticks" in values:
        execution_edge = execution_edge.where(side != "abstain", 0.0)

    state = frame[grouping_columns].copy() if grouping_columns else pd.DataFrame(index=frame.index)
    state["raw_edge"] = raw_edge
    state["fill_captured"] = fill_captured
    state["adverse_cost"] = adverse_cost
    state["execution_edge"] = execution_edge
    state["fill_shortfall"] = raw_edge - fill_captured

    if grouping_columns:
        grouped = state.groupby(grouping_columns, sort=False, dropna=False)
    else:
        grouped = [((), state)]

    rows: list[dict[str, float | int | str]] = []
    for key, group in grouped:
        if grouping_columns:
            key_values = key if isinstance(key, tuple) else (key,)
            row: dict[str, float | int | str] = {
                column: str(value) for column, value in zip(grouping_columns, key_values)
            }
        else:
            row = {}
        mean_raw = float(group["raw_edge"].mean())
        mean_fill = float(group["fill_captured"].mean())
        mean_adverse = float(group["adverse_cost"].mean())
        row.update(
            {
                "rows": int(len(group)),
                "mean_raw_edge_ticks": mean_raw,
                "mean_fill_captured_edge_ticks": mean_fill,
                "mean_adverse_selection_cost_ticks": mean_adverse,
                "mean_execution_adjusted_edge_ticks": float(group["execution_edge"].mean()),
                "mean_fill_shortfall_ticks": float(group["fill_shortfall"].mean()),
                "fill_capture_ratio": 0.0 if mean_raw == 0.0 else float(mean_fill / mean_raw),
                "adverse_drag_ratio": 0.0 if mean_raw == 0.0 else float(mean_adverse / abs(mean_raw)),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


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


def queue_position_lcri_tail_fill_residuals(
    frame: pd.DataFrame,
    *,
    lcri_bins: int = 5,
    regime_col: str | None = "regime",
    side_col: str = "best_execution_side",
    lcri_col: str = "lcri",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    max_abs_fill_residual: float = 0.20,
) -> pd.DataFrame:
    """Audit realized passive fills in the LCRI tails used for execution claims.

    Strong raw LCRI tails can look publishable while the passive order actually
    sits behind too much queue, or fills only in toxic states. This diagnostic
    selects the side implied by ``best_execution_side`` and compares predicted
    passive fill probability with realized fill labels by regime and absolute-LCRI
    tail bin. The result isolates where tail alpha is being overstated by the
    queue-position fill model before it reaches demo/release artifacts.
    """
    if not isinstance(lcri_bins, int) or isinstance(lcri_bins, bool):
        raise ValueError("lcri_bins must be an integer")
    if lcri_bins < 1:
        raise ValueError("lcri_bins must be at least 1")
    if not math.isfinite(max_abs_fill_residual):
        raise ValueError("max_abs_fill_residual must be finite")
    if max_abs_fill_residual < 0.0:
        raise ValueError("max_abs_fill_residual must be non-negative")

    required = {
        side_col,
        lcri_col,
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
        "execution_adjusted_edge_ticks",
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position LCRI tail fill residual")

    columns = [
        "regime",
        "best_execution_side",
        "lcri_tail_bin",
        "rows",
        "mean_abs_lcri",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "fill_residual",
        "absolute_fill_residual",
        "mean_execution_adjusted_edge_ticks",
        "residual_edge_drag_ticks",
        "tail_fill_residual_label",
    ]
    if frame.empty:
        output = pd.DataFrame(columns=columns)
        return output if regime_col is None else output.rename(columns={"regime": regime_col})

    numeric_columns = [
        lcri_col,
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
        "execution_adjusted_edge_ticks",
    ]
    values = _finite_values(frame, numeric_columns, "queue position LCRI tail fill residual")
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        output = pd.DataFrame(columns=columns)
        return output if regime_col is None else output.rename(columns={"regime": regime_col})

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["regime"] = (
        frame.loc[tradable, regime_col].astype(str) if regime_col is not None else "all"
    )
    selected["best_execution_side"] = selected_side
    selected["abs_lcri"] = values.loc[tradable, lcri_col].abs()
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
    for column in ["predicted_fill_probability", "realized_fill"]:
        if not selected[column].between(0.0, 1.0).all():
            raise ValueError("queue position LCRI tail fill residual probabilities must be in [0, 1]")

    rows: list[dict[str, float | int | str]] = []
    for (regime, execution_side), side_group in selected.groupby(
        ["regime", "best_execution_side"], sort=True
    ):
        side_group = side_group.copy()
        side_group["lcri_tail_bin"] = _rank_probability_bins(side_group["abs_lcri"], lcri_bins)
        for tail_bin, group in side_group.groupby("lcri_tail_bin", sort=True):
            predicted = float(group["predicted_fill_probability"].mean())
            realized = float(group["realized_fill"].mean())
            residual = realized - predicted
            abs_residual = abs(residual)
            edge = float(group["execution_adjusted_edge_ticks"].mean())
            if abs_residual <= max_abs_fill_residual + 1e-12:
                label = "tail_fill_calibrated"
            elif residual < 0.0:
                label = "tail_fill_overstated"
            else:
                label = "tail_fill_understated"
            rows.append(
                {
                    "regime": str(regime),
                    "best_execution_side": str(execution_side),
                    "lcri_tail_bin": int(tail_bin),
                    "rows": int(len(group)),
                    "mean_abs_lcri": float(group["abs_lcri"].mean()),
                    "mean_predicted_fill_probability": predicted,
                    "realized_fill_rate": realized,
                    "fill_residual": residual,
                    "absolute_fill_residual": abs_residual,
                    "mean_execution_adjusted_edge_ticks": edge,
                    "residual_edge_drag_ticks": float(abs_residual * abs(edge)),
                    "tail_fill_residual_label": label,
                }
            )
    output = pd.DataFrame(rows, columns=columns)
    if regime_col is not None:
        output = output.rename(columns={"regime": regime_col})
    return output


def queue_position_lcri_tail_adverse_selection_surface(
    frame: pd.DataFrame,
    *,
    lcri_bins: int = 5,
    fill_probability_bins: int = 5,
    regime_col: str | None = "regime",
    side_col: str = "best_execution_side",
    lcri_col: str = "lcri",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    max_abs_fill_residual: float = 0.20,
    min_fill_minus_adverse_rate: float = 0.0,
) -> pd.DataFrame:
    """Surface high-LCRI passive-fill pockets where fills are toxic or overstated.

    Tail LCRI evidence is only tradable when selected-side passive fill odds are
    calibrated *and* not dominated by adverse-selection odds. This diagnostic
    cross-buckets tradable rows by absolute LCRI and selected-side predicted fill
    probability, then compares realized fill rates with selected-side adverse-fill
    probabilities and execution-adjusted edge. It is designed to catch seductive
    high-LCRI/high-fill cells where fills happen, but mostly because the quote is
    about to be picked off.
    """
    for name, value in {
        "lcri_bins": lcri_bins,
        "fill_probability_bins": fill_probability_bins,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    for name, value in {
        "max_abs_fill_residual": max_abs_fill_residual,
        "min_fill_minus_adverse_rate": min_fill_minus_adverse_rate,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if max_abs_fill_residual < 0.0:
        raise ValueError("max_abs_fill_residual must be non-negative")

    required = {
        side_col,
        lcri_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        bid_realized_col,
        ask_realized_col,
        "execution_adjusted_edge_ticks",
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position LCRI tail adverse selection")

    columns = [
        "regime",
        "best_execution_side",
        "lcri_tail_bin",
        "fill_probability_bin",
        "rows",
        "mean_abs_lcri",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "mean_selected_adverse_probability",
        "fill_residual",
        "absolute_fill_residual",
        "fill_minus_adverse_rate",
        "mean_execution_adjusted_edge_ticks",
        "tail_adverse_selection_label",
    ]
    if frame.empty:
        output = pd.DataFrame(columns=columns)
        return output if regime_col is None else output.rename(columns={"regime": regime_col})

    numeric_columns = [
        lcri_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        bid_realized_col,
        ask_realized_col,
        "execution_adjusted_edge_ticks",
    ]
    values = _finite_values(frame, numeric_columns, "queue position LCRI tail adverse selection")
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        output = pd.DataFrame(columns=columns)
        return output if regime_col is None else output.rename(columns={"regime": regime_col})

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["regime"] = (
        frame.loc[tradable, regime_col].astype(str) if regime_col is not None else "all"
    )
    selected["best_execution_side"] = selected_side
    selected["abs_lcri"] = values.loc[tradable, lcri_col].abs()
    selected["predicted_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["selected_adverse_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_adverse_fill_probability"],
        values.loc[tradable, "ask_adverse_fill_probability"],
    )
    selected["realized_fill"] = np.where(
        selected_side == "long",
        values.loc[tradable, bid_realized_col],
        values.loc[tradable, ask_realized_col],
    )
    selected["execution_adjusted_edge_ticks"] = values.loc[
        tradable, "execution_adjusted_edge_ticks"
    ]
    probability_columns = [
        "predicted_fill_probability",
        "selected_adverse_probability",
        "realized_fill",
    ]
    for column in probability_columns:
        if not selected[column].between(0.0, 1.0).all():
            raise ValueError("queue position LCRI tail adverse selection probabilities must be in [0, 1]")

    rows: list[dict[str, float | int | str]] = []
    for (regime, execution_side), side_group in selected.groupby(
        ["regime", "best_execution_side"], sort=True
    ):
        side_group = side_group.copy()
        side_group["lcri_tail_bin"] = _rank_probability_bins(side_group["abs_lcri"], lcri_bins)
        side_group["fill_probability_bin"] = _rank_probability_bins(
            side_group["predicted_fill_probability"], fill_probability_bins
        )
        for (tail_bin, fill_bin), group in side_group.groupby(
            ["lcri_tail_bin", "fill_probability_bin"], sort=True
        ):
            predicted = float(group["predicted_fill_probability"].mean())
            realized = float(group["realized_fill"].mean())
            adverse = float(group["selected_adverse_probability"].mean())
            residual = realized - predicted
            abs_residual = abs(residual)
            fill_minus_adverse = realized - adverse
            edge = float(group["execution_adjusted_edge_ticks"].mean())
            if fill_minus_adverse < min_fill_minus_adverse_rate - 1e-12 or edge < 0.0:
                label = "tail_adverse_toxic"
            elif abs_residual > max_abs_fill_residual + 1e-12 and residual < 0.0:
                label = "tail_fill_overstated"
            elif abs_residual > max_abs_fill_residual + 1e-12:
                label = "tail_fill_understated"
            else:
                label = "tail_adverse_publishable"
            rows.append(
                {
                    "regime": str(regime),
                    "best_execution_side": str(execution_side),
                    "lcri_tail_bin": int(tail_bin),
                    "fill_probability_bin": int(fill_bin),
                    "rows": int(len(group)),
                    "mean_abs_lcri": float(group["abs_lcri"].mean()),
                    "mean_predicted_fill_probability": predicted,
                    "realized_fill_rate": realized,
                    "mean_selected_adverse_probability": adverse,
                    "fill_residual": residual,
                    "absolute_fill_residual": abs_residual,
                    "fill_minus_adverse_rate": fill_minus_adverse,
                    "mean_execution_adjusted_edge_ticks": edge,
                    "tail_adverse_selection_label": label,
                }
            )
    output = pd.DataFrame(rows, columns=columns)
    if regime_col is not None:
        output = output.rename(columns={"regime": regime_col})
    return output


def _empty_queue_position_lcri_tail_adverse_selection_release_scorecard() -> dict[str, float | int | str]:
    return {
        "observed_tail_cells": 0,
        "eligible_tail_cells": 0,
        "total_tail_rows": 0,
        "toxic_tail_cells": 0,
        "toxic_tail_rows": 0,
        "toxic_tail_row_share": 0.0,
        "miscalibrated_tail_cells": 0,
        "miscalibrated_tail_rows": 0,
        "miscalibrated_tail_row_share": 0.0,
        "worst_tail_cell": "none",
        "worst_tail_cell_rows": 0,
        "worst_tail_cell_fill_minus_adverse_rate": 0.0,
        "worst_tail_cell_absolute_fill_residual": 0.0,
        "worst_tail_cell_mean_execution_adjusted_edge_ticks": 0.0,
        "candidate_weighted_fill_minus_adverse_rate": 0.0,
        "candidate_weighted_absolute_fill_residual": 0.0,
        "candidate_weighted_execution_adjusted_edge_ticks": 0.0,
        "tail_adverse_release_label": "pass",
        "blocking_reasons": "none",
        "review_reasons": "none",
    }


def queue_position_lcri_tail_adverse_selection_release_scorecard(
    surface: pd.DataFrame,
    *,
    min_cell_rows: int = 1,
    block_toxic_row_share: float = 0.35,
    review_toxic_row_share: float = 0.15,
    block_fill_minus_adverse_rate: float = -0.05,
    review_fill_residual: float = 0.20,
) -> dict[str, float | int | str]:
    """Gate queue-position LCRI tail cells on toxicity and fill calibration.

    The tail adverse-selection surface is reviewer-friendly, but demos and CI need
    a compact release label. This scorecard weights each LCRI/fill-probability
    cell by observed rows, blocks when toxic tail fills dominate or candidate
    weighted fill-minus-adverse turns negative, and otherwise reviews large tail
    fill-calibration misses before raw LCRI tails are presented as tradable alpha.
    """
    if not isinstance(min_cell_rows, int) or isinstance(min_cell_rows, bool):
        raise ValueError("min_cell_rows must be a positive integer")
    if min_cell_rows < 1:
        raise ValueError("min_cell_rows must be a positive integer")
    for name, value in {
        "block_toxic_row_share": block_toxic_row_share,
        "review_toxic_row_share": review_toxic_row_share,
    }.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    for name, value in {
        "block_fill_minus_adverse_rate": block_fill_minus_adverse_rate,
        "review_fill_residual": review_fill_residual,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if review_fill_residual < 0.0:
        raise ValueError("review_fill_residual must be non-negative")
    if surface.empty:
        return _empty_queue_position_lcri_tail_adverse_selection_release_scorecard()

    required = {
        "best_execution_side",
        "lcri_tail_bin",
        "fill_probability_bin",
        "rows",
        "fill_minus_adverse_rate",
        "absolute_fill_residual",
        "mean_execution_adjusted_edge_ticks",
        "tail_adverse_selection_label",
    }
    regime_col = "regime" if "regime" in surface.columns else None
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(
        surface,
        required,
        "queue position LCRI tail adverse selection release scorecard",
    )
    numeric_columns = [
        "lcri_tail_bin",
        "fill_probability_bin",
        "rows",
        "fill_minus_adverse_rate",
        "absolute_fill_residual",
        "mean_execution_adjusted_edge_ticks",
    ]
    values = _finite_values(
        surface,
        numeric_columns,
        "queue position LCRI tail adverse selection release scorecard",
    )
    if (values["rows"] < 0.0).any():
        raise ValueError("queue position LCRI tail adverse selection release scorecard rows must be non-negative")
    data = values.copy()
    data["best_execution_side"] = surface["best_execution_side"].astype(str)
    data["tail_adverse_selection_label"] = surface["tail_adverse_selection_label"].astype(str)
    data["regime"] = surface[regime_col].astype(str) if regime_col is not None else "all"
    total_rows = int(data["rows"].sum())
    if total_rows == 0:
        scorecard = _empty_queue_position_lcri_tail_adverse_selection_release_scorecard()
        scorecard["observed_tail_cells"] = int(len(surface))
        return scorecard

    eligible = data[data["rows"] >= float(min_cell_rows)].copy()
    weights = data["rows"] / float(total_rows)
    scorecard = _empty_queue_position_lcri_tail_adverse_selection_release_scorecard()
    scorecard.update(
        {
            "observed_tail_cells": int(len(surface)),
            "total_tail_rows": total_rows,
            "candidate_weighted_fill_minus_adverse_rate": float(
                (data["fill_minus_adverse_rate"] * weights).sum()
            ),
            "candidate_weighted_absolute_fill_residual": float(
                (data["absolute_fill_residual"] * weights).sum()
            ),
            "candidate_weighted_execution_adjusted_edge_ticks": float(
                (data["mean_execution_adjusted_edge_ticks"] * weights).sum()
            ),
        }
    )
    if eligible.empty:
        scorecard["tail_adverse_release_label"] = "review"
        scorecard["review_reasons"] = "insufficient_tail_cell_rows"
        return scorecard

    toxic = eligible[eligible["tail_adverse_selection_label"] == "tail_adverse_toxic"]
    miscalibrated = eligible[
        eligible["tail_adverse_selection_label"].isin(
            ["tail_fill_overstated", "tail_fill_understated"]
        )
        | (eligible["absolute_fill_residual"] >= review_fill_residual)
    ]
    toxic_rows = int(toxic["rows"].sum())
    miscalibrated_rows = int(miscalibrated["rows"].sum())
    toxic_row_share = float(toxic_rows) / float(total_rows)
    miscalibrated_row_share = float(miscalibrated_rows) / float(total_rows)

    worst_idx = eligible.sort_values(
        [
            "fill_minus_adverse_rate",
            "mean_execution_adjusted_edge_ticks",
            "absolute_fill_residual",
            "rows",
        ],
        ascending=[True, True, False, False],
    ).index[0]
    worst = data.loc[worst_idx]
    worst_cell = (
        f"{worst['regime']}:{worst['best_execution_side']}:"
        f"lcri_tail={int(worst['lcri_tail_bin'])}:fill_bin={int(worst['fill_probability_bin'])}"
    )

    blocking_reasons = []
    if toxic_row_share >= block_toxic_row_share:
        blocking_reasons.append("toxic_tail_row_share")
    if scorecard["candidate_weighted_fill_minus_adverse_rate"] <= block_fill_minus_adverse_rate:
        blocking_reasons.append("negative_tail_fill_minus_adverse")
    review_reasons = []
    if not blocking_reasons and toxic_row_share >= review_toxic_row_share:
        review_reasons.append("toxic_tail_row_share")
    if not blocking_reasons and not miscalibrated.empty:
        review_reasons.append("tail_fill_miscalibration")
    label = "block" if blocking_reasons else "review" if review_reasons else "pass"

    scorecard.update(
        {
            "eligible_tail_cells": int(len(eligible)),
            "toxic_tail_cells": int(len(toxic)),
            "toxic_tail_rows": toxic_rows,
            "toxic_tail_row_share": toxic_row_share,
            "miscalibrated_tail_cells": int(len(miscalibrated)),
            "miscalibrated_tail_rows": miscalibrated_rows,
            "miscalibrated_tail_row_share": miscalibrated_row_share,
            "worst_tail_cell": worst_cell,
            "worst_tail_cell_rows": int(worst["rows"]),
            "worst_tail_cell_fill_minus_adverse_rate": float(worst["fill_minus_adverse_rate"]),
            "worst_tail_cell_absolute_fill_residual": float(worst["absolute_fill_residual"]),
            "worst_tail_cell_mean_execution_adjusted_edge_ticks": float(
                worst["mean_execution_adjusted_edge_ticks"]
            ),
            "tail_adverse_release_label": label,
            "blocking_reasons": ",".join(blocking_reasons) if blocking_reasons else "none",
            "review_reasons": ",".join(review_reasons) if review_reasons else "none",
        }
    )
    return scorecard


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


def queue_position_order_size_sweep(
    frame: pd.DataFrame,
    *,
    order_size_fractions: list[float] | tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50),
    levels: int = 5,
    fill_config: FillProbabilityConfig | None = None,
    pressure_col: str = "lcri",
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
    long_net_col: str = "long_net_return_ticks",
    short_net_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Stress-test passive execution capacity across child order sizes.

    Queue placement alone can overstate deployability: a passive order that looks
    fillable for a token child size may lose its execution-adjusted edge once the
    strategy scales displayed quantity. This sweep keeps the existing queue-ahead
    assumption fixed, increases child order size as a fraction of best displayed
    size, and recomputes fill odds plus execution-adjusted edge for each size.
    """
    if isinstance(order_size_fractions, (str, bytes)):
        raise ValueError("order_size_fractions must be a non-empty sequence of finite values")
    order_size_fractions = list(order_size_fractions)
    if not order_size_fractions:
        raise ValueError("order_size_fractions must be a non-empty sequence")
    for fraction in order_size_fractions:
        if not math.isfinite(float(fraction)):
            raise ValueError("order_size_fraction values must be finite")
        if float(fraction) < 0.0:
            raise ValueError("order_size_fraction values must be non-negative")
    if frame.empty:
        return _empty_queue_position_order_size_sweep()

    rows: list[dict[str, float | int | str]] = []
    for fraction in order_size_fractions:
        sized = add_queue_position_order_size_features(
            frame,
            levels=levels,
            order_size_fraction=float(fraction),
        )
        filled = add_passive_fill_probabilities(
            sized,
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
                "order_size_fraction": float(fraction),
                "rows": rows_count,
                "mean_bid_child_order_size": float(executed["bid_child_order_size"].mean()),
                "mean_ask_child_order_size": float(executed["ask_child_order_size"].mean()),
                "mean_bid_queue_clear_share": float(executed["bid_queue_clear_share"].mean()),
                "mean_ask_queue_clear_share": float(executed["ask_queue_clear_share"].mean()),
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
    return pd.DataFrame(rows)[list(_empty_queue_position_order_size_sweep().columns)]


def queue_position_order_size_capacity_frontier(
    sweep: pd.DataFrame,
    *,
    min_edge_ticks: float = 0.0,
    min_tradable_share: float = 0.50,
) -> dict[str, float | int | str]:
    """Find the largest child-order size that preserves executable LCRI edge.

    ``queue_position_order_size_sweep`` isolates scale capacity at a fixed queue
    placement. This reducer converts that decay curve into a publishable frontier:
    the largest child order, as a fraction of best displayed size, that still
    satisfies execution-adjusted edge and tradability gates.
    """
    if not math.isfinite(min_edge_ticks):
        raise ValueError("min_edge_ticks must be finite")
    if not math.isfinite(min_tradable_share):
        raise ValueError("min_tradable_share must be finite")
    if not 0.0 <= min_tradable_share <= 1.0:
        raise ValueError("min_tradable_share must be in [0.0, 1.0]")

    empty = _empty_queue_position_order_size_capacity_frontier()
    if sweep.empty:
        return empty

    required = {
        "order_size_fraction",
        "rows",
        "mean_execution_adjusted_edge_ticks",
        "tradable_share",
        "dominant_execution_side",
    }
    _require_columns(sweep, required, "queue position order size capacity frontier")
    values = _finite_values(
        sweep,
        [
            "order_size_fraction",
            "rows",
            "mean_execution_adjusted_edge_ticks",
            "tradable_share",
        ],
        "queue position order size capacity frontier",
    )
    if (values["order_size_fraction"] < 0.0).any():
        raise ValueError("queue position order size capacity frontier fractions must be non-negative")
    if (values["rows"] < 0.0).any():
        raise ValueError("queue position order size capacity frontier rows must be non-negative")
    if not values["tradable_share"].between(0.0, 1.0).all():
        raise ValueError("queue position order size capacity frontier tradable shares must be in [0, 1]")

    data = values.copy()
    data["dominant_execution_side"] = sweep["dominant_execution_side"].astype(str)
    data = data.sort_values("order_size_fraction", ignore_index=True)
    minimum = data.iloc[0]
    viable = data[
        (data["mean_execution_adjusted_edge_ticks"] >= min_edge_ticks)
        & (data["tradable_share"] >= min_tradable_share)
    ]

    if viable.empty:
        result = empty.copy()
        result.update(
            {
                "rows": int(len(data)),
                "minimum_order_size_fraction": float(minimum["order_size_fraction"]),
                "minimum_size_mean_execution_adjusted_edge_ticks": float(
                    minimum["mean_execution_adjusted_edge_ticks"]
                ),
                "minimum_size_tradable_share": float(minimum["tradable_share"]),
                "order_size_capacity_label": "no_viable_child_order_capacity",
            }
        )
        return result

    capacity = viable.iloc[-1]
    edge_decay = float(
        minimum["mean_execution_adjusted_edge_ticks"]
        - capacity["mean_execution_adjusted_edge_ticks"]
    )
    tradable_decay = float(minimum["tradable_share"] - capacity["tradable_share"])
    max_size = float(capacity["order_size_fraction"])
    return {
        "rows": int(len(data)),
        "viable_rows": int(len(viable)),
        "minimum_order_size_fraction": float(minimum["order_size_fraction"]),
        "max_viable_order_size_fraction": max_size,
        "minimum_size_mean_execution_adjusted_edge_ticks": float(
            minimum["mean_execution_adjusted_edge_ticks"]
        ),
        "max_viable_mean_execution_adjusted_edge_ticks": float(
            capacity["mean_execution_adjusted_edge_ticks"]
        ),
        "edge_decay_to_capacity_ticks": edge_decay,
        "minimum_size_tradable_share": float(minimum["tradable_share"]),
        "max_viable_tradable_share": float(capacity["tradable_share"]),
        "tradable_share_decay_to_capacity": tradable_decay,
        "dominant_execution_side_at_capacity": str(capacity["dominant_execution_side"]),
        "order_size_capacity_label": _queue_order_size_capacity_label(
            max_size=max_size,
            edge_decay=edge_decay,
            tradable_decay=tradable_decay,
        ),
    }


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


def queue_position_expected_value_frontier(
    frame: pd.DataFrame,
    *,
    min_fill_probabilities: list[float] | tuple[float, ...] = (0.50, 0.60, 0.70, 0.80),
    max_queue_shares: list[float] | tuple[float, ...] = (0.25, 0.50, 0.75),
    adverse_selection_cost_ticks: float = 0.50,
    queue_drag_cost_ticks: float = 0.25,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
) -> pd.DataFrame:
    """Score queue-aware passive policies as expected-value frontiers.

    Fill-probability thresholds alone can prefer deep-in-queue orders whose expected
    edge vanishes after toxicity and opportunity-cost drag. This diagnostic crosses
    side-specific minimum fill probability with maximum visible queue share, then
    reports ex-ante edge, toxicity-adjusted EV, and coverage by regime so execution
    review can choose passive policies on tradable value rather than raw signal rank.
    """
    columns = [
        "regime",
        "min_fill_probability",
        "max_queue_share",
        "tradable_rows",
        "candidate_rows",
        "candidate_share",
        "long_rows",
        "short_rows",
        "mean_queue_share",
        "mean_fill_probability",
        "mean_adverse_fill_probability",
        "mean_execution_adjusted_edge_ticks",
        "expected_value_ticks",
        "risk_adjusted_expected_value_ticks",
        "policy_label",
    ]
    if isinstance(min_fill_probabilities, (str, bytes)) or not min_fill_probabilities:
        raise ValueError("min_fill_probabilities must be a non-empty sequence")
    if isinstance(max_queue_shares, (str, bytes)) or not max_queue_shares:
        raise ValueError("max_queue_shares must be a non-empty sequence")
    for name, sequence in {
        "min_fill_probabilities": min_fill_probabilities,
        "max_queue_shares": max_queue_shares,
    }.items():
        for value in sequence:
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
                raise ValueError(f"{name} values must be finite and in [0.0, 1.0]")
    for name, value in {
        "adverse_selection_cost_ticks": adverse_selection_cost_ticks,
        "queue_drag_cost_ticks": queue_drag_cost_ticks,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    required = {
        side_col,
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "execution_adjusted_edge_ticks",
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position expected value frontier")
    if frame.empty:
        return pd.DataFrame(columns=columns)

    numeric_columns = [
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "execution_adjusted_edge_ticks",
    ]
    values = _finite_values(frame, numeric_columns, "queue position expected value frontier")
    if (values[["bid_queue_share", "ask_queue_share"]] < 0.0).any().any():
        raise ValueError("queue position expected value frontier queue shares must be non-negative")
    probability_columns = [
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    ]
    if not values[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue position expected value frontier probabilities must be in [0, 1]")

    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return pd.DataFrame(columns=columns)

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["regime"] = frame.loc[tradable, regime_col].astype(str) if regime_col else "all"
    selected["side"] = selected_side
    selected["queue_share"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_queue_share"],
        values.loc[tradable, "ask_queue_share"],
    )
    selected["fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["adverse_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_adverse_fill_probability"],
        values.loc[tradable, "ask_adverse_fill_probability"],
    )
    selected["execution_adjusted_edge_ticks"] = values.loc[
        tradable, "execution_adjusted_edge_ticks"
    ]
    selected["expected_value_ticks"] = (
        selected["execution_adjusted_edge_ticks"] * selected["fill_probability"]
    )
    selected["risk_adjusted_expected_value_ticks"] = (
        selected["expected_value_ticks"]
        - selected["adverse_fill_probability"] * adverse_selection_cost_ticks
        - selected["queue_share"] * queue_drag_cost_ticks
    )

    rows: list[dict[str, float | int | str]] = []
    for regime, group in selected.groupby("regime", sort=True):
        tradable_rows = len(group)
        for min_fill in sorted(float(value) for value in min_fill_probabilities):
            for max_queue in sorted(float(value) for value in max_queue_shares):
                candidate = group[
                    (group["fill_probability"] >= min_fill)
                    & (group["queue_share"] <= max_queue)
                ]
                candidate_rows = len(candidate)
                if candidate_rows:
                    mean_queue = float(candidate["queue_share"].mean())
                    mean_fill = float(candidate["fill_probability"].mean())
                    mean_adverse = float(candidate["adverse_fill_probability"].mean())
                    mean_edge = float(candidate["execution_adjusted_edge_ticks"].mean())
                    expected_value = float(candidate["expected_value_ticks"].mean())
                    risk_adjusted_ev = float(candidate["risk_adjusted_expected_value_ticks"].mean())
                    long_rows = int((candidate["side"] == "long").sum())
                    short_rows = int((candidate["side"] == "short").sum())
                else:
                    mean_queue = 0.0
                    mean_fill = 0.0
                    mean_adverse = 0.0
                    mean_edge = 0.0
                    expected_value = 0.0
                    risk_adjusted_ev = 0.0
                    long_rows = 0
                    short_rows = 0
                candidate_share = float(candidate_rows / tradable_rows) if tradable_rows else 0.0
                rows.append(
                    {
                        "regime": str(regime),
                        "min_fill_probability": min_fill,
                        "max_queue_share": max_queue,
                        "tradable_rows": int(tradable_rows),
                        "candidate_rows": int(candidate_rows),
                        "candidate_share": candidate_share,
                        "long_rows": long_rows,
                        "short_rows": short_rows,
                        "mean_queue_share": mean_queue,
                        "mean_fill_probability": mean_fill,
                        "mean_adverse_fill_probability": mean_adverse,
                        "mean_execution_adjusted_edge_ticks": mean_edge,
                        "expected_value_ticks": expected_value,
                        "risk_adjusted_expected_value_ticks": risk_adjusted_ev,
                        "policy_label": _queue_position_expected_value_policy_label(
                            candidate_share=candidate_share,
                            risk_adjusted_ev=risk_adjusted_ev,
                            mean_adverse_fill_probability=mean_adverse,
                        ),
                    }
                )
    return pd.DataFrame(rows)[columns]


def queue_position_expected_value_policy_selection(
    frontier: pd.DataFrame,
    *,
    min_candidate_share: float = 0.10,
    require_positive_ev: bool = True,
) -> pd.DataFrame:
    """Select deployable queue-position policies from an expected-value frontier.

    The frontier enumerates many queue/fill cutoffs; this reducer picks one
    execution policy per regime after enforcing capacity and EV constraints. It
    still returns the best blocked row so review can separate negative EV from
    insufficient passive capacity.
    """
    columns = [
        "regime",
        "selected_min_fill_probability",
        "selected_max_queue_share",
        "tradable_rows",
        "candidate_rows",
        "candidate_share",
        "risk_adjusted_expected_value_ticks",
        "expected_value_ticks",
        "mean_fill_probability",
        "mean_queue_share",
        "mean_adverse_fill_probability",
        "policy_rank",
        "selection_label",
    ]
    if not math.isfinite(min_candidate_share) or not 0.0 <= min_candidate_share <= 1.0:
        raise ValueError("min_candidate_share must be finite and in [0.0, 1.0]")
    if not isinstance(require_positive_ev, bool):
        raise ValueError("require_positive_ev must be boolean")

    required = {
        "regime",
        "min_fill_probability",
        "max_queue_share",
        "tradable_rows",
        "candidate_rows",
        "candidate_share",
        "risk_adjusted_expected_value_ticks",
        "expected_value_ticks",
        "mean_fill_probability",
        "mean_queue_share",
        "mean_adverse_fill_probability",
    }
    _require_columns(frontier, required, "queue position expected value policy selection")
    if frontier.empty:
        return pd.DataFrame(columns=columns)

    numeric_columns = sorted(required - {"regime"})
    values = _finite_values(frontier, numeric_columns, "queue position expected value policy selection")
    share_columns = ["min_fill_probability", "max_queue_share", "candidate_share"]
    if not values[share_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue position expected value policy selection shares must be in [0, 1]")
    if (values[["tradable_rows", "candidate_rows"]] < 0.0).any().any():
        raise ValueError("queue position expected value policy selection row counts must be non-negative")

    working = frontier.copy()
    working[numeric_columns] = values
    rows: list[dict[str, float | int | str]] = []
    for regime, group in working.groupby("regime", sort=True):
        viable = group[group["candidate_share"] >= min_candidate_share]
        if require_positive_ev:
            viable = viable[viable["risk_adjusted_expected_value_ticks"] > 0.0]
        if viable.empty:
            ranked = group.sort_values(
                ["candidate_share", "risk_adjusted_expected_value_ticks", "expected_value_ticks"],
                ascending=[False, False, False],
            )
            selected = ranked.iloc[0]
            if float(selected["candidate_rows"]) <= 0.0:
                label = "no_candidates"
            elif float(selected["candidate_share"]) < min_candidate_share:
                label = "capacity_constrained"
            elif require_positive_ev and float(selected["risk_adjusted_expected_value_ticks"]) <= 0.0:
                label = "negative_expected_value"
            else:
                label = "review"
        else:
            ranked = viable.sort_values(
                [
                    "risk_adjusted_expected_value_ticks",
                    "expected_value_ticks",
                    "candidate_share",
                    "mean_fill_probability",
                    "mean_queue_share",
                ],
                ascending=[False, False, False, False, True],
            )
            selected = ranked.iloc[0]
            label = "deployable"
        rows.append(
            {
                "regime": str(regime),
                "selected_min_fill_probability": float(selected["min_fill_probability"]),
                "selected_max_queue_share": float(selected["max_queue_share"]),
                "tradable_rows": int(selected["tradable_rows"]),
                "candidate_rows": int(selected["candidate_rows"]),
                "candidate_share": float(selected["candidate_share"]),
                "risk_adjusted_expected_value_ticks": float(
                    selected["risk_adjusted_expected_value_ticks"]
                ),
                "expected_value_ticks": float(selected["expected_value_ticks"]),
                "mean_fill_probability": float(selected["mean_fill_probability"]),
                "mean_queue_share": float(selected["mean_queue_share"]),
                "mean_adverse_fill_probability": float(selected["mean_adverse_fill_probability"]),
                "policy_rank": len(rows) + 1,
                "selection_label": label,
            }
        )
    return pd.DataFrame(rows)[columns]


def queue_position_expected_value_policy_scorecard(selection: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected EV policies into an execution-readiness scorecard.

    ``queue_position_expected_value_policy_selection`` gives one policy per regime.
    This scorecard compresses those policies into a publishable deployment artifact:
    how many regimes are deployable, why blocked regimes fail, and the candidate-
    capacity-weighted EV/toxicity profile of the selected policy set.
    """
    columns = [
        "regimes",
        "deployable_regimes",
        "blocked_regimes",
        "deployable_share",
        "capacity_constrained_regimes",
        "negative_expected_value_regimes",
        "no_candidate_regimes",
        "candidate_weighted_share",
        "candidate_weighted_risk_adjusted_expected_value_ticks",
        "worst_risk_adjusted_expected_value_ticks",
        "candidate_weighted_adverse_fill_probability",
        "readiness_label",
    ]
    required = {
        "regime",
        "candidate_rows",
        "candidate_share",
        "risk_adjusted_expected_value_ticks",
        "mean_adverse_fill_probability",
        "selection_label",
    }
    _require_columns(selection, required, "queue position expected value policy scorecard")
    if selection.empty:
        return pd.DataFrame(columns=columns)

    numeric_columns = [
        "candidate_rows",
        "candidate_share",
        "risk_adjusted_expected_value_ticks",
        "mean_adverse_fill_probability",
    ]
    values = _finite_values(
        selection, numeric_columns, "queue position expected value policy scorecard"
    )
    if (values["candidate_rows"] < 0.0).any():
        raise ValueError("queue position expected value policy scorecard candidate_rows must be non-negative")
    share_columns = ["candidate_share", "mean_adverse_fill_probability"]
    if not values[share_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue position expected value policy scorecard shares must be in [0, 1]")

    labels = selection["selection_label"].astype(str)
    deployable = labels == "deployable"
    regimes = int(len(selection))
    deployable_regimes = int(deployable.sum())
    blocked_regimes = regimes - deployable_regimes
    weights = values["candidate_rows"]
    weight_total = float(weights.sum())
    if weight_total > 0.0:
        candidate_weighted_share = float((values["candidate_share"] * weights).sum() / weight_total)
        weighted_ev = float(
            (values["risk_adjusted_expected_value_ticks"] * weights).sum() / weight_total
        )
        weighted_adverse = float(
            (values["mean_adverse_fill_probability"] * weights).sum() / weight_total
        )
    else:
        candidate_weighted_share = 0.0
        weighted_ev = 0.0
        weighted_adverse = 0.0
    worst_ev = float(values["risk_adjusted_expected_value_ticks"].min())

    if deployable_regimes == regimes and weighted_ev > 0.0:
        readiness_label = "execution_ready"
    elif deployable_regimes > 0:
        readiness_label = "mixed_review"
    else:
        readiness_label = "not_deployable"

    return pd.DataFrame(
        [
            {
                "regimes": regimes,
                "deployable_regimes": deployable_regimes,
                "blocked_regimes": blocked_regimes,
                "deployable_share": deployable_regimes / regimes,
                "capacity_constrained_regimes": int((labels == "capacity_constrained").sum()),
                "negative_expected_value_regimes": int((labels == "negative_expected_value").sum()),
                "no_candidate_regimes": int((labels == "no_candidates").sum()),
                "candidate_weighted_share": candidate_weighted_share,
                "candidate_weighted_risk_adjusted_expected_value_ticks": weighted_ev,
                "worst_risk_adjusted_expected_value_ticks": worst_ev,
                "candidate_weighted_adverse_fill_probability": weighted_adverse,
                "readiness_label": readiness_label,
            }
        ],
        columns=columns,
    )


def queue_position_expected_value_policy_drift(
    train_selection: pd.DataFrame,
    holdout_selection: pd.DataFrame,
    *,
    max_threshold_drift: float = 0.10,
    max_ev_decay_ratio: float = 0.50,
    min_holdout_candidate_share: float = 0.10,
) -> pd.DataFrame:
    """Compare train vs holdout EV policy cutoffs for recalibration risk.

    OOS replay checks whether the *same* selected policy still works. This companion
    diagnostic asks a different deployment question: if the policy is re-selected on
    holdout data, did the required fill/queue cutoffs materially move? Large cutoff
    drift implies the passive policy may be path- or regime-fragile even when average
    EV remains positive.
    """
    columns = [
        "regime",
        "train_min_fill_probability",
        "holdout_min_fill_probability",
        "min_fill_probability_delta",
        "train_max_queue_share",
        "holdout_max_queue_share",
        "max_queue_share_delta",
        "threshold_l1_drift",
        "train_candidate_share",
        "holdout_candidate_share",
        "candidate_share_delta",
        "train_risk_adjusted_expected_value_ticks",
        "holdout_risk_adjusted_expected_value_ticks",
        "ev_decay_ticks",
        "ev_decay_ratio",
        "train_selection_label",
        "holdout_selection_label",
        "policy_drift_label",
        "review_reasons",
    ]
    if not math.isfinite(max_threshold_drift) or max_threshold_drift < 0.0:
        raise ValueError("max_threshold_drift must be finite and non-negative")
    if not math.isfinite(max_ev_decay_ratio) or max_ev_decay_ratio < 0.0:
        raise ValueError("max_ev_decay_ratio must be finite and non-negative")
    if not math.isfinite(min_holdout_candidate_share) or not 0.0 <= min_holdout_candidate_share <= 1.0:
        raise ValueError("min_holdout_candidate_share must be finite and in [0.0, 1.0]")

    required = {
        "regime",
        "selected_min_fill_probability",
        "selected_max_queue_share",
        "candidate_share",
        "risk_adjusted_expected_value_ticks",
        "selection_label",
    }
    _require_columns(train_selection, required, "queue position expected value policy drift")
    _require_columns(holdout_selection, required, "queue position expected value policy drift")
    if train_selection.empty:
        return pd.DataFrame(columns=columns)

    numeric_columns = sorted(required - {"regime", "selection_label"})
    train_values = _finite_values(
        train_selection, numeric_columns, "queue position expected value policy drift"
    )
    if not holdout_selection.empty:
        holdout_values = _finite_values(
            holdout_selection, numeric_columns, "queue position expected value policy drift"
        )
        share_columns = [
            "selected_min_fill_probability",
            "selected_max_queue_share",
            "candidate_share",
        ]
        if not holdout_values[share_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
            raise ValueError("queue position expected value policy drift shares must be in [0, 1]")
    else:
        holdout_values = pd.DataFrame(index=holdout_selection.index, columns=numeric_columns)
    share_columns = ["selected_min_fill_probability", "selected_max_queue_share", "candidate_share"]
    if not train_values[share_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue position expected value policy drift shares must be in [0, 1]")

    train = train_selection.copy()
    holdout = holdout_selection.copy()
    train[numeric_columns] = train_values
    if not holdout.empty:
        holdout[numeric_columns] = holdout_values
    holdout_by_regime = {
        str(row["regime"]): row for _, row in holdout.drop_duplicates("regime", keep="first").iterrows()
    }

    rows: list[dict[str, float | str]] = []
    for _, train_row in train.iterrows():
        regime = str(train_row["regime"])
        train_min_fill = float(train_row["selected_min_fill_probability"])
        train_queue_share = float(train_row["selected_max_queue_share"])
        train_candidate_share = float(train_row["candidate_share"])
        train_ev = float(train_row["risk_adjusted_expected_value_ticks"])
        train_label = str(train_row["selection_label"])
        holdout_row = holdout_by_regime.get(regime)
        if holdout_row is None:
            rows.append(
                {
                    "regime": regime,
                    "train_min_fill_probability": train_min_fill,
                    "holdout_min_fill_probability": 0.0,
                    "min_fill_probability_delta": 0.0,
                    "train_max_queue_share": train_queue_share,
                    "holdout_max_queue_share": 0.0,
                    "max_queue_share_delta": 0.0,
                    "threshold_l1_drift": 0.0,
                    "train_candidate_share": train_candidate_share,
                    "holdout_candidate_share": 0.0,
                    "candidate_share_delta": -train_candidate_share,
                    "train_risk_adjusted_expected_value_ticks": train_ev,
                    "holdout_risk_adjusted_expected_value_ticks": 0.0,
                    "ev_decay_ticks": max(train_ev, 0.0),
                    "ev_decay_ratio": 1.0 if train_ev > 0.0 else 0.0,
                    "train_selection_label": train_label,
                    "holdout_selection_label": "missing",
                    "policy_drift_label": "holdout_missing_regime",
                    "review_reasons": "missing_holdout_regime",
                }
            )
            continue

        holdout_min_fill = float(holdout_row["selected_min_fill_probability"])
        holdout_queue_share = float(holdout_row["selected_max_queue_share"])
        holdout_candidate_share = float(holdout_row["candidate_share"])
        holdout_ev = float(holdout_row["risk_adjusted_expected_value_ticks"])
        holdout_label = str(holdout_row["selection_label"])
        min_fill_delta = holdout_min_fill - train_min_fill
        queue_share_delta = holdout_queue_share - train_queue_share
        threshold_l1_drift = abs(min_fill_delta) + abs(queue_share_delta)
        candidate_share_delta = holdout_candidate_share - train_candidate_share
        ev_decay_ticks = train_ev - holdout_ev
        if train_ev > 0.0:
            ev_decay_ratio = max(0.0, ev_decay_ticks) / train_ev
        else:
            ev_decay_ratio = 1.0 if holdout_ev < train_ev else 0.0

        reasons: list[str] = []
        if train_label != "deployable" or holdout_label != "deployable":
            reasons.append("not_deployable")
        if threshold_l1_drift > max_threshold_drift:
            reasons.append("threshold_drift")
        if holdout_candidate_share < min_holdout_candidate_share:
            reasons.append("holdout_capacity")
        if ev_decay_ratio > max_ev_decay_ratio:
            reasons.append("ev_decay")

        if "not_deployable" in reasons:
            policy_label = "not_deployable"
        elif reasons:
            policy_label = "policy_recalibration_required"
        else:
            policy_label = "policy_stable"

        rows.append(
            {
                "regime": regime,
                "train_min_fill_probability": train_min_fill,
                "holdout_min_fill_probability": holdout_min_fill,
                "min_fill_probability_delta": min_fill_delta,
                "train_max_queue_share": train_queue_share,
                "holdout_max_queue_share": holdout_queue_share,
                "max_queue_share_delta": queue_share_delta,
                "threshold_l1_drift": threshold_l1_drift,
                "train_candidate_share": train_candidate_share,
                "holdout_candidate_share": holdout_candidate_share,
                "candidate_share_delta": candidate_share_delta,
                "train_risk_adjusted_expected_value_ticks": train_ev,
                "holdout_risk_adjusted_expected_value_ticks": holdout_ev,
                "ev_decay_ticks": ev_decay_ticks,
                "ev_decay_ratio": ev_decay_ratio,
                "train_selection_label": train_label,
                "holdout_selection_label": holdout_label,
                "policy_drift_label": policy_label,
                "review_reasons": ";".join(reasons) if reasons else "none",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def queue_position_expected_value_oos_validation(
    selection: pd.DataFrame,
    holdout_frontier: pd.DataFrame,
    *,
    min_holdout_candidate_share: float = 0.10,
    max_ev_decay_ratio: float = 0.50,
    min_holdout_expected_value_ticks: float = 0.0,
) -> pd.DataFrame:
    """Replay selected queue-position EV policies on a holdout frontier.

    ``queue_position_expected_value_policy_selection`` chooses the best in-sample
    fill/queue cutoff per regime. This validator joins those exact policy cutoffs
    onto an independently computed holdout frontier so execution claims are not
    promoted when the chosen passive policy loses capacity, flips negative EV, or
    suffers severe out-of-sample expected-value decay.
    """
    columns = [
        "regime",
        "selected_min_fill_probability",
        "selected_max_queue_share",
        "train_candidate_rows",
        "train_candidate_share",
        "train_risk_adjusted_expected_value_ticks",
        "holdout_tradable_rows",
        "holdout_candidate_rows",
        "holdout_candidate_share",
        "holdout_risk_adjusted_expected_value_ticks",
        "holdout_expected_value_ticks",
        "holdout_mean_fill_probability",
        "holdout_mean_queue_share",
        "holdout_mean_adverse_fill_probability",
        "ev_decay_ticks",
        "ev_decay_ratio",
        "oos_validation_label",
        "review_reasons",
    ]
    if not math.isfinite(min_holdout_candidate_share) or not 0.0 <= min_holdout_candidate_share <= 1.0:
        raise ValueError("min_holdout_candidate_share must be finite and in [0.0, 1.0]")
    if not math.isfinite(max_ev_decay_ratio) or max_ev_decay_ratio < 0.0:
        raise ValueError("max_ev_decay_ratio must be finite and non-negative")
    if not math.isfinite(min_holdout_expected_value_ticks):
        raise ValueError("min_holdout_expected_value_ticks must be finite")

    selection_required = {
        "regime",
        "selected_min_fill_probability",
        "selected_max_queue_share",
        "candidate_rows",
        "candidate_share",
        "risk_adjusted_expected_value_ticks",
    }
    holdout_required = {
        "regime",
        "min_fill_probability",
        "max_queue_share",
        "tradable_rows",
        "candidate_rows",
        "candidate_share",
        "risk_adjusted_expected_value_ticks",
        "expected_value_ticks",
        "mean_fill_probability",
        "mean_queue_share",
        "mean_adverse_fill_probability",
    }
    _require_columns(selection, selection_required, "queue position expected value OOS validation")
    _require_columns(holdout_frontier, holdout_required, "queue position expected value OOS validation")
    if selection.empty:
        return pd.DataFrame(columns=columns)

    selection_numeric = [
        "selected_min_fill_probability",
        "selected_max_queue_share",
        "candidate_rows",
        "candidate_share",
        "risk_adjusted_expected_value_ticks",
    ]
    holdout_numeric = sorted(holdout_required - {"regime"})
    selected_values = _finite_values(
        selection, selection_numeric, "queue position expected value OOS validation"
    )
    holdout_values = _finite_values(
        holdout_frontier, holdout_numeric, "queue position expected value OOS validation"
    )
    selected_share_columns = [
        "selected_min_fill_probability",
        "selected_max_queue_share",
        "candidate_share",
    ]
    holdout_share_columns = [
        "min_fill_probability",
        "max_queue_share",
        "candidate_share",
        "mean_fill_probability",
        "mean_queue_share",
        "mean_adverse_fill_probability",
    ]
    if not selected_values[selected_share_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue position expected value OOS validation selection shares must be in [0, 1]")
    if not holdout_values[holdout_share_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue position expected value OOS validation holdout shares must be in [0, 1]")
    if (selected_values["candidate_rows"] < 0.0).any() or (
        holdout_values[["tradable_rows", "candidate_rows"]] < 0.0
    ).any().any():
        raise ValueError("queue position expected value OOS validation row counts must be non-negative")

    selected = selection.copy()
    selected[selection_numeric] = selected_values
    holdout = holdout_frontier.copy()
    holdout[holdout_numeric] = holdout_values
    holdout["_regime_key"] = holdout["regime"].astype(str)
    rows: list[dict[str, float | int | str]] = []
    for _, policy in selected.iterrows():
        regime = str(policy["regime"])
        min_fill = float(policy["selected_min_fill_probability"])
        max_queue = float(policy["selected_max_queue_share"])
        match = holdout[
            (holdout["_regime_key"] == regime)
            & np.isclose(holdout["min_fill_probability"], min_fill)
            & np.isclose(holdout["max_queue_share"], max_queue)
        ]
        train_ev = float(policy["risk_adjusted_expected_value_ticks"])
        if match.empty:
            holdout_tradable = 0
            holdout_candidates = 0
            holdout_share = 0.0
            holdout_ev = 0.0
            holdout_expected_value = 0.0
            holdout_fill = 0.0
            holdout_queue = 0.0
            holdout_adverse = 0.0
            ev_decay = train_ev
            ev_decay_ratio = 1.0 if train_ev > 0.0 else 0.0
            label = "oos_missing_policy"
            review_reasons = "missing_holdout_policy"
        else:
            selected_holdout = match.sort_values(
                ["candidate_share", "risk_adjusted_expected_value_ticks"], ascending=[False, False]
            ).iloc[0]
            holdout_tradable = int(selected_holdout["tradable_rows"])
            holdout_candidates = int(selected_holdout["candidate_rows"])
            holdout_share = float(selected_holdout["candidate_share"])
            holdout_ev = float(selected_holdout["risk_adjusted_expected_value_ticks"])
            holdout_expected_value = float(selected_holdout["expected_value_ticks"])
            holdout_fill = float(selected_holdout["mean_fill_probability"])
            holdout_queue = float(selected_holdout["mean_queue_share"])
            holdout_adverse = float(selected_holdout["mean_adverse_fill_probability"])
            ev_decay = train_ev - holdout_ev
            ev_decay_ratio = ev_decay / abs(train_ev) if train_ev != 0.0 else 0.0
            reasons: list[str] = []
            if holdout_share < min_holdout_candidate_share:
                reasons.append("capacity")
            if holdout_ev < min_holdout_expected_value_ticks:
                reasons.append("negative_ev")
            if ev_decay_ratio > max_ev_decay_ratio:
                reasons.append("ev_decay")
            if not reasons:
                label = "oos_stable"
                review_reasons = "none"
            elif "negative_ev" in reasons or "capacity" in reasons:
                label = "oos_broken"
                review_reasons = ";".join(reasons)
            else:
                label = "oos_degraded"
                review_reasons = ";".join(reasons)
        rows.append(
            {
                "regime": regime,
                "selected_min_fill_probability": min_fill,
                "selected_max_queue_share": max_queue,
                "train_candidate_rows": int(policy["candidate_rows"]),
                "train_candidate_share": float(policy["candidate_share"]),
                "train_risk_adjusted_expected_value_ticks": train_ev,
                "holdout_tradable_rows": holdout_tradable,
                "holdout_candidate_rows": holdout_candidates,
                "holdout_candidate_share": holdout_share,
                "holdout_risk_adjusted_expected_value_ticks": holdout_ev,
                "holdout_expected_value_ticks": holdout_expected_value,
                "holdout_mean_fill_probability": holdout_fill,
                "holdout_mean_queue_share": holdout_queue,
                "holdout_mean_adverse_fill_probability": holdout_adverse,
                "ev_decay_ticks": ev_decay,
                "ev_decay_ratio": ev_decay_ratio,
                "oos_validation_label": label,
                "review_reasons": review_reasons,
            }
        )
    return pd.DataFrame(rows)[columns]


def queue_position_expected_value_stress_table(
    selection: pd.DataFrame,
    *,
    stress_scenarios: dict[str, Any] | None = None,
    adverse_selection_cost_ticks: float = 0.50,
    queue_drag_cost_ticks: float = 0.25,
    min_candidate_share: float = 0.10,
    min_stressed_expected_value_ticks: float = 0.0,
) -> pd.DataFrame:
    """Stress selected queue-position EV policies for latency/toxicity haircuts.

    ``queue_position_expected_value_policy_selection`` chooses the best policy under
    observed fill, adverse-fill, and queue-share estimates. This table asks whether
    those choices survive realistic degradation: lower passive fill rates after
    latency or queue-priority loss, and higher adverse-selection probability when
    quotes are filled by toxic flow. It derives the selected policy's implied gross
    edge from ``expected_value_ticks / mean_fill_probability`` and re-prices it
    under each scenario so a publishability packet can distinguish robust policies
    from ones that only work at optimistic queue assumptions.
    """
    columns = [
        "scenario",
        "regime",
        "fill_probability_haircut",
        "adverse_fill_probability_uplift",
        "candidate_rows",
        "candidate_share",
        "stressed_fill_probability",
        "stressed_adverse_fill_probability",
        "implied_edge_ticks",
        "stressed_expected_value_ticks",
        "expected_value_decay_ticks",
        "stress_label",
    ]
    if stress_scenarios is None:
        stress_scenarios = {
            "base": (0.0, 0.0),
            "latency_haircut": (0.15, 0.05),
            "toxicity_haircut": (0.25, 0.10),
        }
    if not stress_scenarios:
        raise ValueError("stress_scenarios must be a non-empty mapping")
    for name, value in {
        "adverse_selection_cost_ticks": adverse_selection_cost_ticks,
        "queue_drag_cost_ticks": queue_drag_cost_ticks,
        "min_candidate_share": min_candidate_share,
        "min_stressed_expected_value_ticks": min_stressed_expected_value_ticks,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if adverse_selection_cost_ticks < 0.0:
        raise ValueError("adverse_selection_cost_ticks must be non-negative")
    if queue_drag_cost_ticks < 0.0:
        raise ValueError("queue_drag_cost_ticks must be non-negative")
    if not 0.0 <= min_candidate_share <= 1.0:
        raise ValueError("min_candidate_share must be in [0.0, 1.0]")

    required = {
        "regime",
        "candidate_rows",
        "candidate_share",
        "expected_value_ticks",
        "mean_fill_probability",
        "mean_queue_share",
        "mean_adverse_fill_probability",
    }
    _require_columns(selection, required, "queue position expected value stress table")
    if selection.empty:
        return pd.DataFrame(columns=columns)

    numeric_columns = sorted(required - {"regime"})
    values = _finite_values(selection, numeric_columns, "queue position expected value stress table")
    if (values[["candidate_rows", "mean_queue_share"]] < 0.0).any().any():
        raise ValueError("queue position expected value stress table counts and queue shares must be non-negative")
    probability_columns = ["candidate_share", "mean_fill_probability", "mean_adverse_fill_probability"]
    if not values[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue position expected value stress table probabilities must be in [0, 1]")

    parsed_scenarios: list[tuple[str, float, float]] = []
    for scenario, settings in stress_scenarios.items():
        if isinstance(settings, (str, bytes)):
            raise ValueError("stress_scenarios values must be (fill_haircut, adverse_uplift) pairs")
        try:
            fill_haircut_raw, adverse_uplift_raw = settings
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "stress_scenarios values must be (fill_haircut, adverse_uplift) pairs"
            ) from exc
        fill_haircut = float(fill_haircut_raw)
        adverse_uplift = float(adverse_uplift_raw)
        if (
            not math.isfinite(fill_haircut)
            or not math.isfinite(adverse_uplift)
            or not 0.0 <= fill_haircut <= 1.0
            or not 0.0 <= adverse_uplift <= 1.0
        ):
            raise ValueError("stress_scenarios values must be finite probabilities in [0.0, 1.0]")
        parsed_scenarios.append((str(scenario), fill_haircut, adverse_uplift))

    working = selection.copy()
    working[numeric_columns] = values
    rows: list[dict[str, float | int | str]] = []
    for scenario, fill_haircut, adverse_uplift in parsed_scenarios:
        for _, row in working.iterrows():
            mean_fill = float(row["mean_fill_probability"])
            implied_edge = float(row["expected_value_ticks"]) / mean_fill if mean_fill > 0.0 else 0.0
            stressed_fill = mean_fill * (1.0 - fill_haircut)
            stressed_adverse = min(1.0, float(row["mean_adverse_fill_probability"]) + adverse_uplift)
            stressed_ev = (
                implied_edge * stressed_fill
                - stressed_adverse * adverse_selection_cost_ticks
                - float(row["mean_queue_share"]) * queue_drag_cost_ticks
            )
            expected_value_decay = float(row["expected_value_ticks"]) - stressed_ev
            capacity_ok = float(row["candidate_share"]) >= min_candidate_share
            ev_ok = stressed_ev >= min_stressed_expected_value_ticks
            if capacity_ok and ev_ok:
                stress_label = "stress_robust"
            elif not capacity_ok and ev_ok:
                stress_label = "capacity_fragile"
            elif capacity_ok and not ev_ok:
                stress_label = "expected_value_fragile"
            else:
                stress_label = "capacity_or_ev_fragile"
            rows.append(
                {
                    "scenario": scenario,
                    "regime": str(row["regime"]),
                    "fill_probability_haircut": fill_haircut,
                    "adverse_fill_probability_uplift": adverse_uplift,
                    "candidate_rows": int(row["candidate_rows"]),
                    "candidate_share": float(row["candidate_share"]),
                    "stressed_fill_probability": stressed_fill,
                    "stressed_adverse_fill_probability": stressed_adverse,
                    "implied_edge_ticks": implied_edge,
                    "stressed_expected_value_ticks": stressed_ev,
                    "expected_value_decay_ticks": expected_value_decay,
                    "stress_label": stress_label,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def queue_position_expected_value_stress_summary(
    stress: pd.DataFrame,
    *,
    max_fragile_candidate_share: float = 0.25,
    review_fragile_candidate_share: float = 0.10,
    min_candidate_weighted_ev_ticks: float = 0.0,
    min_worst_scenario_ev_ticks: float = -0.05,
) -> dict[str, float | int | str]:
    """Summarize queue-position EV stress survival across regimes and scenarios.

    The stress table is intentionally granular: each selected regime/policy is
    repriced under latency and toxicity haircuts. This reducer converts that grid
    into one release-facing decision by candidate-weighting stressed EV, tracking
    how much selected capacity sits in fragile rows, and naming the weakest
    scenario/regime pockets. It is designed to make optimistic queue EV policies
    non-publishable when they only survive in the base scenario.
    """
    for name, value in {
        "max_fragile_candidate_share": max_fragile_candidate_share,
        "review_fragile_candidate_share": review_fragile_candidate_share,
        "min_candidate_weighted_ev_ticks": min_candidate_weighted_ev_ticks,
        "min_worst_scenario_ev_ticks": min_worst_scenario_ev_ticks,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_fragile_candidate_share <= 1.0:
        raise ValueError("max_fragile_candidate_share must be in [0, 1]")
    if not 0.0 <= review_fragile_candidate_share <= 1.0:
        raise ValueError("review_fragile_candidate_share must be in [0, 1]")

    empty: dict[str, float | int | str] = {
        "stress_rows": 0,
        "scenarios": 0,
        "regimes": 0,
        "candidate_rows": 0,
        "fragile_candidate_rows": 0,
        "fragile_candidate_share": 0.0,
        "candidate_weighted_expected_value_ticks": 0.0,
        "candidate_weighted_decay_ticks": 0.0,
        "worst_scenario": "none",
        "worst_scenario_expected_value_ticks": 0.0,
        "worst_regime": "none",
        "worst_regime_expected_value_ticks": 0.0,
        "stress_release_decision": "review",
        "stress_release_label": "queue_expected_value_stress_no_evidence",
        "blocking_reasons": "none",
        "review_reasons": "no_stress_evidence",
    }
    if stress.empty:
        return empty

    required = {
        "scenario",
        "regime",
        "candidate_rows",
        "candidate_share",
        "stressed_expected_value_ticks",
        "expected_value_decay_ticks",
        "stress_label",
    }
    _require_columns(stress, required, "queue position expected value stress summary")
    values = _finite_values(
        stress,
        [
            "candidate_rows",
            "candidate_share",
            "stressed_expected_value_ticks",
            "expected_value_decay_ticks",
        ],
        "queue position expected value stress summary",
    )
    if (values[["candidate_rows", "candidate_share"]] < 0.0).any().any():
        raise ValueError("queue position expected value stress summary counts must be non-negative")
    if not values["candidate_share"].between(0.0, 1.0).all():
        raise ValueError("queue position expected value stress summary shares must be in [0, 1]")

    candidate_rows = int(values["candidate_rows"].sum())
    weights = values["candidate_rows"]
    if candidate_rows == 0:
        return empty | {
            "stress_rows": int(len(stress)),
            "scenarios": int(stress["scenario"].astype(str).nunique()),
            "regimes": int(stress["regime"].astype(str).nunique()),
            "review_reasons": "no_candidate_stress_evidence",
        }

    labels = stress["stress_label"].astype(str)
    fragile_mask = labels != "stress_robust"
    fragile_candidate_rows = int(values.loc[fragile_mask, "candidate_rows"].sum())
    fragile_candidate_share = fragile_candidate_rows / candidate_rows
    weighted_ev = float(np.average(values["stressed_expected_value_ticks"], weights=weights))
    weighted_decay = float(np.average(values["expected_value_decay_ticks"], weights=weights))

    working = stress.copy()
    working[["candidate_rows", "stressed_expected_value_ticks"]] = values[
        ["candidate_rows", "stressed_expected_value_ticks"]
    ]

    def weighted_ev_by(group_col: str) -> pd.Series:
        grouped: dict[str, float] = {}
        for key, group in working.groupby(working[group_col].astype(str), sort=False):
            group_weights = group["candidate_rows"].astype(float)
            total_weight = float(group_weights.sum())
            grouped[str(key)] = (
                0.0
                if total_weight == 0.0
                else float(np.average(group["stressed_expected_value_ticks"], weights=group_weights))
            )
        return pd.Series(grouped, dtype=float)

    scenario_ev = weighted_ev_by("scenario")
    regime_ev = weighted_ev_by("regime")
    worst_scenario = str(scenario_ev.idxmin()) if not scenario_ev.empty else "none"
    worst_regime = str(regime_ev.idxmin()) if not regime_ev.empty else "none"
    worst_scenario_ev = float(scenario_ev.min()) if not scenario_ev.empty else 0.0
    worst_regime_ev = float(regime_ev.min()) if not regime_ev.empty else 0.0

    blocking_reasons: list[str] = []
    review_reasons: list[str] = []
    if fragile_candidate_share > max_fragile_candidate_share:
        blocking_reasons.append("fragile_candidate_share")
    elif fragile_candidate_share > review_fragile_candidate_share:
        review_reasons.append("fragile_candidate_share")
    if weighted_ev < min_candidate_weighted_ev_ticks:
        blocking_reasons.append("candidate_weighted_expected_value")
    if worst_scenario_ev < min_worst_scenario_ev_ticks:
        blocking_reasons.append("worst_scenario_expected_value")

    if blocking_reasons:
        decision = "block"
        label = "queue_expected_value_stress_blocked"
    elif review_reasons:
        decision = "review"
        label = "queue_expected_value_stress_review"
    else:
        decision = "pass"
        label = "queue_expected_value_stress_pass"

    return {
        "stress_rows": int(len(stress)),
        "scenarios": int(stress["scenario"].astype(str).nunique()),
        "regimes": int(stress["regime"].astype(str).nunique()),
        "candidate_rows": candidate_rows,
        "fragile_candidate_rows": fragile_candidate_rows,
        "fragile_candidate_share": float(fragile_candidate_share),
        "candidate_weighted_expected_value_ticks": weighted_ev,
        "candidate_weighted_decay_ticks": weighted_decay,
        "worst_scenario": worst_scenario,
        "worst_scenario_expected_value_ticks": worst_scenario_ev,
        "worst_regime": worst_regime,
        "worst_regime_expected_value_ticks": worst_regime_ev,
        "stress_release_decision": decision,
        "stress_release_label": label,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
    }


def queue_position_adverse_selection_policy_frontier(
    frame: pd.DataFrame,
    *,
    fill_thresholds: list[float] | tuple[float, ...] = (0.50, 0.60, 0.70, 0.80),
    adverse_thresholds: list[float] | tuple[float, ...] = (0.20, 0.30, 0.40),
    side_col: str = "best_execution_side",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    long_return_col: str = "long_net_return_ticks",
    short_return_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Evaluate passive policies jointly on fill probability and adverse-selection risk.

    A fill cutoff alone can accidentally select toxic queues where execution is
    likely precisely because informed flow is about to trade through the quote.
    This frontier sweeps fill-probability lower bounds against adverse-fill upper
    bounds, reporting capacity, realized fill, realized edge, and how many otherwise
    executable rows were filtered by toxicity controls.
    """
    _validate_probability_thresholds(fill_thresholds, "fill_thresholds", "fill")
    _validate_probability_thresholds(adverse_thresholds, "adverse_thresholds", "adverse")

    required = {
        side_col,
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
    _require_columns(frame, required, "queue position adverse-selection policy frontier")
    if frame.empty:
        return _empty_queue_position_adverse_selection_policy_frontier()

    values = _finite_values(
        frame,
        [
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
            bid_realized_col,
            ask_realized_col,
            long_return_col,
            short_return_col,
            "execution_adjusted_edge_ticks",
        ],
        "queue position adverse-selection policy frontier",
    )
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_queue_position_adverse_selection_policy_frontier()

    selected_side = side.loc[tradable]
    selected = pd.DataFrame(index=frame.index[tradable])
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
    tradable_rows = len(selected)
    for fill_threshold in sorted(float(value) for value in fill_thresholds):
        fill_eligible = selected["predicted_fill_probability"] >= fill_threshold
        for adverse_threshold in sorted(float(value) for value in adverse_thresholds):
            toxicity_ok = selected["adverse_fill_probability"] <= adverse_threshold
            candidate = selected[fill_eligible & toxicity_ok]
            candidate_rows = len(candidate)
            toxicity_filtered_rows = int((fill_eligible & ~toxicity_ok).sum())
            if candidate_rows:
                mean_predicted = float(candidate["predicted_fill_probability"].mean())
                mean_adverse = float(candidate["adverse_fill_probability"].mean())
                realized_fill_rate = float(candidate["realized_fill"].mean())
                mean_realized_edge = float(candidate["realized_edge_ticks"].mean())
                positive_edge_rate = float((candidate["realized_edge_ticks"] > 0.0).mean())
                mean_adjusted_edge = float(candidate["execution_adjusted_edge_ticks"].mean())
                long_rows = int((candidate["side"] == "long").sum())
                short_rows = int((candidate["side"] == "short").sum())
            else:
                mean_predicted = 0.0
                mean_adverse = 0.0
                realized_fill_rate = 0.0
                mean_realized_edge = 0.0
                positive_edge_rate = 0.0
                mean_adjusted_edge = 0.0
                long_rows = 0
                short_rows = 0
            trade_share = float(candidate_rows / total_rows) if total_rows else 0.0
            toxicity_filtered_share = (
                float(toxicity_filtered_rows / tradable_rows) if tradable_rows else 0.0
            )
            rows.append(
                {
                    "fill_threshold": fill_threshold,
                    "adverse_threshold": adverse_threshold,
                    "candidate_rows": int(candidate_rows),
                    "trade_share": trade_share,
                    "long_rows": long_rows,
                    "short_rows": short_rows,
                    "mean_predicted_fill_probability": mean_predicted,
                    "mean_adverse_fill_probability": mean_adverse,
                    "realized_fill_rate": realized_fill_rate,
                    "mean_realized_edge_ticks": mean_realized_edge,
                    "positive_edge_rate": positive_edge_rate,
                    "mean_execution_adjusted_edge_ticks": mean_adjusted_edge,
                    "toxicity_filtered_rows": toxicity_filtered_rows,
                    "toxicity_filtered_share": toxicity_filtered_share,
                    "policy_label": _queue_position_policy_frontier_label(
                        trade_share=trade_share,
                        realized_fill_rate=realized_fill_rate,
                        mean_realized_edge=mean_realized_edge,
                        mean_adverse_fill_probability=mean_adverse,
                    ),
                }
            )
    return pd.DataFrame(rows)[list(_empty_queue_position_adverse_selection_policy_frontier().columns)]


def queue_position_adverse_selection_policy_summary(
    frontier: pd.DataFrame,
    *,
    min_trade_share: float = 0.10,
    min_realized_fill_rate: float = 0.70,
    min_mean_realized_edge_ticks: float = 0.0,
    max_mean_adverse_fill_probability: float = 0.30,
    max_toxicity_filtered_share: float = 1.0,
) -> dict[str, float | int | str]:
    """Select a publishable passive policy from fill/toxicity threshold frontiers.

    ``queue_position_adverse_selection_policy_frontier`` intentionally emits the
    whole threshold grid. This reducer turns that grid into a release-facing policy
    decision by requiring minimum executable capacity, realized fill quality, edge,
    and adverse-selection controls, then selecting the highest-edge policy that
    survives. It prevents demos from cherry-picking a fill-only threshold while
    hiding that the quote set is either too small or toxicity-filtered away.
    """
    for name, value in {
        "min_trade_share": min_trade_share,
        "min_realized_fill_rate": min_realized_fill_rate,
        "min_mean_realized_edge_ticks": min_mean_realized_edge_ticks,
        "max_mean_adverse_fill_probability": max_mean_adverse_fill_probability,
        "max_toxicity_filtered_share": max_toxicity_filtered_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name, value in {
        "min_trade_share": min_trade_share,
        "min_realized_fill_rate": min_realized_fill_rate,
        "max_mean_adverse_fill_probability": max_mean_adverse_fill_probability,
        "max_toxicity_filtered_share": max_toxicity_filtered_share,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0.0, 1.0]")

    empty: dict[str, float | int | str] = {
        "policies": 0,
        "publishable_policies": 0,
        "best_fill_threshold": 0.0,
        "best_adverse_threshold": 0.0,
        "best_candidate_rows": 0,
        "best_trade_share": 0.0,
        "best_realized_fill_rate": 0.0,
        "best_mean_predicted_fill_probability": 0.0,
        "best_mean_adverse_fill_probability": 0.0,
        "best_mean_realized_edge_ticks": 0.0,
        "best_positive_edge_rate": 0.0,
        "best_mean_execution_adjusted_edge_ticks": 0.0,
        "best_toxicity_filtered_share": 0.0,
        "dominant_side": "none",
        "best_policy_label": "none",
        "policy_summary_label": "no_policy_frontier_data",
    }
    if frontier.empty:
        return empty

    required = {
        "fill_threshold",
        "adverse_threshold",
        "candidate_rows",
        "trade_share",
        "long_rows",
        "short_rows",
        "mean_predicted_fill_probability",
        "mean_adverse_fill_probability",
        "realized_fill_rate",
        "mean_realized_edge_ticks",
        "positive_edge_rate",
        "mean_execution_adjusted_edge_ticks",
        "toxicity_filtered_share",
        "policy_label",
    }
    _require_columns(frontier, required, "queue position adverse-selection policy summary")
    numeric_columns = list(required - {"policy_label"})
    values = _finite_values(frontier, numeric_columns, "queue position adverse-selection policy summary")
    if not values[["candidate_rows", "long_rows", "short_rows"]].ge(0.0).all().all():
        raise ValueError("queue position adverse-selection policy summary counts must be non-negative")
    probability_columns = [
        "fill_threshold",
        "adverse_threshold",
        "trade_share",
        "mean_predicted_fill_probability",
        "mean_adverse_fill_probability",
        "realized_fill_rate",
        "positive_edge_rate",
        "toxicity_filtered_share",
    ]
    if not values[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue position adverse-selection policy summary probabilities must be in [0, 1]")
    if (values["long_rows"] + values["short_rows"] > values["candidate_rows"]).any():
        raise ValueError("queue position adverse-selection policy summary side counts exceed candidates")

    data = values.copy()
    data["policy_label"] = frontier["policy_label"].astype(str)
    viable = data[
        (data["candidate_rows"] > 0.0)
        & (data["trade_share"] >= min_trade_share)
        & (data["realized_fill_rate"] >= min_realized_fill_rate)
        & (data["mean_realized_edge_ticks"] >= min_mean_realized_edge_ticks)
        & (data["mean_adverse_fill_probability"] <= max_mean_adverse_fill_probability)
        & (data["toxicity_filtered_share"] <= max_toxicity_filtered_share)
    ]
    result = empty.copy()
    result.update(
        {
            "policies": int(len(data)),
            "publishable_policies": int(len(viable)),
        }
    )
    if viable.empty:
        result["policy_summary_label"] = "no_publishable_toxicity_control_policy"
        return result

    best = viable.sort_values(
        [
            "mean_realized_edge_ticks",
            "realized_fill_rate",
            "trade_share",
            "mean_adverse_fill_probability",
            "toxicity_filtered_share",
        ],
        ascending=[False, False, False, True, True],
    ).iloc[0]
    dominant_side = (
        "long"
        if float(best["long_rows"]) > float(best["short_rows"])
        else "short"
        if float(best["short_rows"]) > float(best["long_rows"])
        else "balanced"
    )
    result.update(
        {
            "best_fill_threshold": float(best["fill_threshold"]),
            "best_adverse_threshold": float(best["adverse_threshold"]),
            "best_candidate_rows": int(best["candidate_rows"]),
            "best_trade_share": float(best["trade_share"]),
            "best_realized_fill_rate": float(best["realized_fill_rate"]),
            "best_mean_predicted_fill_probability": float(best["mean_predicted_fill_probability"]),
            "best_mean_adverse_fill_probability": float(best["mean_adverse_fill_probability"]),
            "best_mean_realized_edge_ticks": float(best["mean_realized_edge_ticks"]),
            "best_positive_edge_rate": float(best["positive_edge_rate"]),
            "best_mean_execution_adjusted_edge_ticks": float(best["mean_execution_adjusted_edge_ticks"]),
            "best_toxicity_filtered_share": float(best["toxicity_filtered_share"]),
            "dominant_side": dominant_side,
            "best_policy_label": str(best["policy_label"]),
            "policy_summary_label": _queue_position_policy_summary_label(
                trade_share=float(best["trade_share"]),
                realized_fill_rate=float(best["realized_fill_rate"]),
                mean_realized_edge=float(best["mean_realized_edge_ticks"]),
                mean_adverse_fill_probability=float(best["mean_adverse_fill_probability"]),
            ),
        }
    )
    return result


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


def queue_position_fill_monotonicity_scorecard(
    surface: pd.DataFrame,
    *,
    regime_col: str = "regime",
    inversion_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Score whether deeper visible queues have worse passive-fill odds.

    Queue-position-aware fill probabilities should not improve simply because a
    passive child order sits farther back in the visible queue. This reducer takes
    the side-aware fill calibration surface, collapses fill-probability bins into a
    queue-depth ladder per regime/side, and flags predicted or realized fill-rate
    inversions where a deeper queue bin exceeds the previous shallower bin by more
    than ``inversion_tolerance``.
    """
    if not regime_col:
        raise ValueError("regime_col must be non-empty")
    if not math.isfinite(inversion_tolerance) or inversion_tolerance < 0.0:
        raise ValueError("inversion_tolerance must be a finite non-negative value")

    columns = list(_empty_queue_position_fill_monotonicity_scorecard().columns)
    if surface.empty:
        return _empty_queue_position_fill_monotonicity_scorecard()

    required = {
        regime_col,
        "best_execution_side",
        "queue_share_bin",
        "rows",
        "mean_queue_share",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
    }
    _require_columns(surface, required, "queue position fill monotonicity scorecard")
    numeric_columns = [
        "queue_share_bin",
        "rows",
        "mean_queue_share",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
    ]
    values = _finite_values(surface, numeric_columns, "queue position fill monotonicity scorecard")
    if not values["rows"].ge(0.0).all():
        raise ValueError("queue position fill monotonicity rows must be non-negative")
    if not values["mean_queue_share"].ge(0.0).all():
        raise ValueError("queue position fill monotonicity queue shares must be non-negative")
    for column in ["mean_predicted_fill_probability", "realized_fill_rate"]:
        if not values[column].between(0.0, 1.0).all():
            raise ValueError(f"queue position fill monotonicity {column} must be in [0, 1]")

    working = surface.copy()
    working[regime_col] = working[regime_col].astype(str)
    working["best_execution_side"] = working["best_execution_side"].astype(str)
    for column in numeric_columns:
        working[column] = values[column]
    working = working[working["rows"] > 0.0]
    if working.empty:
        return _empty_queue_position_fill_monotonicity_scorecard()

    rows: list[dict[str, float | int | str]] = []
    for (regime, execution_side), group in working.groupby(
        [regime_col, "best_execution_side"], sort=True
    ):
        ladder_rows: list[dict[str, float | int]] = []
        for queue_bin, queue_group in group.groupby("queue_share_bin", sort=True):
            weights = queue_group["rows"].to_numpy(dtype=float)
            ladder_rows.append(
                {
                    "queue_share_bin": int(queue_bin),
                    "rows": int(queue_group["rows"].sum()),
                    "mean_queue_share": float(
                        np.average(queue_group["mean_queue_share"], weights=weights)
                    ),
                    "mean_predicted_fill_probability": float(
                        np.average(
                            queue_group["mean_predicted_fill_probability"], weights=weights
                        )
                    ),
                    "realized_fill_rate": float(
                        np.average(queue_group["realized_fill_rate"], weights=weights)
                    ),
                }
            )
        ladder = pd.DataFrame(ladder_rows).sort_values("mean_queue_share")
        predicted_inversions = 0
        realized_inversions = 0
        max_predicted_inversion = 0.0
        max_realized_inversion = 0.0
        previous_predicted: float | None = None
        previous_realized: float | None = None
        for _, row in ladder.iterrows():
            predicted = float(row["mean_predicted_fill_probability"])
            realized = float(row["realized_fill_rate"])
            if previous_predicted is not None:
                predicted_gap = predicted - previous_predicted
                realized_gap = realized - previous_realized if previous_realized is not None else 0.0
                if predicted_gap > inversion_tolerance:
                    predicted_inversions += 1
                    max_predicted_inversion = max(max_predicted_inversion, float(predicted_gap))
                if realized_gap > inversion_tolerance:
                    realized_inversions += 1
                    max_realized_inversion = max(max_realized_inversion, float(realized_gap))
            previous_predicted = predicted
            previous_realized = realized
        queue_steps = max(len(ladder) - 1, 0)
        if queue_steps == 0:
            label = "queue_fill_monotonicity_review"
        elif realized_inversions:
            label = "queue_fill_monotonicity_block"
        elif predicted_inversions:
            label = "queue_fill_monotonicity_review"
        else:
            label = "queue_fill_monotonicity_pass"
        rows.append(
            {
                "regime": str(regime),
                "best_execution_side": str(execution_side),
                "queue_bins": int(len(ladder)),
                "queue_steps": int(queue_steps),
                "rows": int(ladder["rows"].sum()),
                "predicted_fill_inversions": int(predicted_inversions),
                "realized_fill_inversions": int(realized_inversions),
                "max_predicted_fill_inversion": max_predicted_inversion,
                "max_realized_fill_inversion": max_realized_inversion,
                "monotonicity_label": label,
            }
        )
    output = pd.DataFrame(rows, columns=columns)
    if regime_col != "regime":
        output = output.rename(columns={"regime": regime_col})
    return output.sort_values(
        ["realized_fill_inversions", "predicted_fill_inversions", "rows", regime_col],
        ascending=[False, False, False, True],
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


def queue_position_calibration_stability(
    research_surface: pd.DataFrame,
    heldout_surface: pd.DataFrame,
    *,
    regime_col: str = "regime",
    max_error_gap: float = 0.10,
    max_fill_rate_gap: float = 0.20,
) -> pd.DataFrame:
    """Compare queue-position fill calibration cells across research/holdout samples.

    A queue-aware fill model is publishable only if its calibration cells replicate
    outside the sample used to tune queue depth and probability cutoffs. This join
    keeps regime, execution side, queue-depth bin, and predicted-fill bin fixed,
    then measures heldout gaps in fill rate, calibration error, Brier score, and
    execution-adjusted edge so fragile passive-fill cells can be gated explicitly.
    """
    if not regime_col:
        raise ValueError("regime_col must be non-empty")
    if not math.isfinite(max_error_gap) or max_error_gap < 0.0:
        raise ValueError("max_error_gap must be a finite non-negative value")
    if not math.isfinite(max_fill_rate_gap) or max_fill_rate_gap < 0.0:
        raise ValueError("max_fill_rate_gap must be a finite non-negative value")
    if research_surface.empty and heldout_surface.empty:
        return _empty_queue_position_calibration_stability(regime_col=regime_col)

    required = {
        regime_col,
        "best_execution_side",
        "queue_share_bin",
        "fill_probability_bin",
        "rows",
        "realized_fill_rate",
        "calibration_error",
        "absolute_calibration_error",
        "brier_score",
        "mean_execution_adjusted_edge_ticks",
    }
    _require_columns(research_surface, required, "research queue position calibration stability")
    _require_columns(heldout_surface, required, "heldout queue position calibration stability")

    def normalize(surface: pd.DataFrame, label: str) -> pd.DataFrame:
        values = _finite_values(
            surface,
            [
                "queue_share_bin",
                "fill_probability_bin",
                "rows",
                "realized_fill_rate",
                "calibration_error",
                "absolute_calibration_error",
                "brier_score",
                "mean_execution_adjusted_edge_ticks",
            ],
            f"{label} queue position calibration stability",
        )
        if not values["rows"].ge(0.0).all():
            raise ValueError(f"{label} queue position calibration stability rows must be non-negative")
        for column in ["realized_fill_rate", "absolute_calibration_error", "brier_score"]:
            if not values[column].between(0.0, 1.0).all():
                raise ValueError(
                    f"{label} queue position calibration stability {column} must be in [0, 1]"
                )
        if not values["calibration_error"].between(-1.0, 1.0).all():
            raise ValueError(
                f"{label} queue position calibration stability calibration_error must be in [-1, 1]"
            )
        output = pd.DataFrame(
            {
                regime_col: surface[regime_col].astype(str),
                "best_execution_side": surface["best_execution_side"].astype(str),
            }
        )
        for column in values.columns:
            output[column] = values[column]
        return output

    research = normalize(research_surface, "research")
    heldout = normalize(heldout_surface, "heldout")
    key_cols = [regime_col, "best_execution_side", "queue_share_bin", "fill_probability_bin"]
    joined = research.merge(
        heldout,
        on=key_cols,
        how="outer",
        suffixes=("_research", "_heldout"),
        indicator=True,
    )
    if joined.empty:
        return _empty_queue_position_calibration_stability(regime_col=regime_col)

    rows: list[dict[str, float | int | str]] = []
    for _, row in joined.sort_values(key_cols).iterrows():
        merge_state = str(row["_merge"])
        research_missing = merge_state == "right_only"
        heldout_missing = merge_state == "left_only"
        research_rows = 0 if research_missing else int(row["rows_research"])
        heldout_rows = 0 if heldout_missing else int(row["rows_heldout"])
        fill_gap = 0.0 if research_missing or heldout_missing else float(
            row["realized_fill_rate_heldout"] - row["realized_fill_rate_research"]
        )
        calibration_gap = 0.0 if research_missing or heldout_missing else float(
            row["calibration_error_heldout"] - row["calibration_error_research"]
        )
        abs_error_gap = float(
            (0.0 if heldout_missing else row["absolute_calibration_error_heldout"])
            - (0.0 if research_missing else row["absolute_calibration_error_research"])
        )
        brier_gap = float(
            (0.0 if heldout_missing else row["brier_score_heldout"])
            - (0.0 if research_missing else row["brier_score_research"])
        )
        edge_gap = 0.0 if research_missing or heldout_missing else float(
            row["mean_execution_adjusted_edge_ticks_heldout"]
            - row["mean_execution_adjusted_edge_ticks_research"]
        )
        label = _queue_calibration_stability_label(
            research_missing=research_missing,
            heldout_missing=heldout_missing,
            fill_gap=fill_gap,
            abs_error_gap=abs_error_gap,
            edge_gap=edge_gap,
            max_error_gap=max_error_gap,
            max_fill_rate_gap=max_fill_rate_gap,
        )
        rows.append(
            {
                "regime": str(row[regime_col]),
                "best_execution_side": str(row["best_execution_side"]),
                "queue_share_bin": int(row["queue_share_bin"]),
                "fill_probability_bin": int(row["fill_probability_bin"]),
                "research_rows": research_rows,
                "heldout_rows": heldout_rows,
                "research_realized_fill_rate": 0.0 if research_missing else float(row["realized_fill_rate_research"]),
                "heldout_realized_fill_rate": 0.0 if heldout_missing else float(row["realized_fill_rate_heldout"]),
                "realized_fill_rate_gap": fill_gap,
                "research_calibration_error": 0.0 if research_missing else float(row["calibration_error_research"]),
                "heldout_calibration_error": 0.0 if heldout_missing else float(row["calibration_error_heldout"]),
                "calibration_error_gap": calibration_gap,
                "research_absolute_calibration_error": 0.0 if research_missing else float(row["absolute_calibration_error_research"]),
                "heldout_absolute_calibration_error": 0.0 if heldout_missing else float(row["absolute_calibration_error_heldout"]),
                "absolute_calibration_error_gap": abs_error_gap,
                "brier_score_gap": brier_gap,
                "research_mean_execution_adjusted_edge_ticks": 0.0 if research_missing else float(row["mean_execution_adjusted_edge_ticks_research"]),
                "heldout_mean_execution_adjusted_edge_ticks": 0.0 if heldout_missing else float(row["mean_execution_adjusted_edge_ticks_heldout"]),
                "execution_adjusted_edge_gap_ticks": edge_gap,
                "calibration_stability_label": label,
            }
        )
    output = pd.DataFrame(rows, columns=_empty_queue_position_calibration_stability().columns)
    if regime_col != "regime":
        output = output.rename(columns={"regime": regime_col})
    return output


def queue_position_calibration_stability_summary(
    stability: pd.DataFrame,
    *,
    regime_col: str = "regime",
) -> dict[str, float | int | str]:
    """Reduce queue-position calibration stability into a release-gate summary."""
    if not regime_col:
        raise ValueError("regime_col must be non-empty")
    empty: dict[str, float | int | str] = {
        "cells": 0,
        "common_cells": 0,
        "replicated_cells": 0,
        "degraded_cells": 0,
        "lost_cells": 0,
        "gained_cells": 0,
        "degraded_cell_share": 0.0,
        "mean_absolute_calibration_error_gap": 0.0,
        "worst_regime": "none",
        "worst_best_execution_side": "none",
        "worst_queue_share_bin": 0,
        "worst_fill_probability_bin": 0,
        "worst_calibration_stability_label": "none",
        "queue_calibration_stability_label": "no_queue_calibration_stability_data",
    }
    if stability.empty:
        return empty
    required = {
        regime_col,
        "best_execution_side",
        "queue_share_bin",
        "fill_probability_bin",
        "research_rows",
        "heldout_rows",
        "absolute_calibration_error_gap",
        "calibration_stability_label",
    }
    _require_columns(stability, required, "queue position calibration stability summary")
    values = _finite_values(
        stability,
        [
            "queue_share_bin",
            "fill_probability_bin",
            "research_rows",
            "heldout_rows",
            "absolute_calibration_error_gap",
        ],
        "queue position calibration stability summary",
    )
    if not values[["research_rows", "heldout_rows"]].ge(0.0).all().all():
        raise ValueError("queue position calibration stability summary rows must be non-negative")

    labels = stability["calibration_stability_label"].astype(str)
    cells = int(len(stability))
    common = (values["research_rows"] > 0.0) & (values["heldout_rows"] > 0.0)
    degraded = labels == "calibration_degraded"
    lost = labels == "calibration_cell_lost"
    gained = labels == "calibration_cell_gained"
    replicated = labels == "calibration_replicated"
    if bool(lost.any()):
        label = "queue_calibration_cells_lost"
    elif bool(degraded.any()):
        label = "queue_calibration_degraded"
    elif bool(replicated.any()) and int(replicated.sum()) == int(common.sum()) and not bool(gained.any()):
        label = "queue_calibration_replicated"
    else:
        label = "queue_calibration_mixed"
    worst = pd.DataFrame(
        {
            regime_col: stability[regime_col].astype(str),
            "best_execution_side": stability["best_execution_side"].astype(str),
            "queue_share_bin": values["queue_share_bin"],
            "fill_probability_bin": values["fill_probability_bin"],
            "label": labels,
            "lost": lost.astype(int),
            "degraded": degraded.astype(int),
            "abs_error_gap": values["absolute_calibration_error_gap"],
        }
    ).sort_values(
        ["lost", "degraded", "abs_error_gap", regime_col, "best_execution_side"],
        ascending=[False, False, False, True, True],
    ).iloc[0]
    return {
        "cells": cells,
        "common_cells": int(common.sum()),
        "replicated_cells": int(replicated.sum()),
        "degraded_cells": int(degraded.sum()),
        "lost_cells": int(lost.sum()),
        "gained_cells": int(gained.sum()),
        "degraded_cell_share": float(degraded.sum() / cells) if cells else 0.0,
        "mean_absolute_calibration_error_gap": float(
            values.loc[common, "absolute_calibration_error_gap"].mean()
        ) if bool(common.any()) else 0.0,
        "worst_regime": str(worst[regime_col]),
        "worst_best_execution_side": str(worst["best_execution_side"]),
        "worst_queue_share_bin": int(worst["queue_share_bin"]),
        "worst_fill_probability_bin": int(worst["fill_probability_bin"]),
        "worst_calibration_stability_label": str(worst["label"]),
        "queue_calibration_stability_label": label,
    }


def queue_position_calibration_reliability_scorecard(
    residual_summary: pd.DataFrame,
    drift: pd.DataFrame,
    *,
    stability_summary: dict[str, float | int | str] | None = None,
    regime_col: str = "regime",
    max_weighted_abs_error: float = 0.20,
    max_unstable_drift_share: float = 0.25,
) -> dict[str, float | int | str]:
    """Combine queue-fill residual, drift, and holdout stability into a release scorecard.

    Queue-position fill probabilities should not be treated as publishable simply
    because their aggregate calibration looks acceptable. This scorecard focuses
    review on the fragile execution modes that matter for passive LCRI: systematic
    underfills that already drag execution edge, queue/probability cells that drift
    across regimes, and calibration cells that degrade or disappear in holdout.
    """
    if not regime_col:
        raise ValueError("regime_col must be non-empty")
    if not math.isfinite(max_weighted_abs_error) or max_weighted_abs_error < 0.0:
        raise ValueError("max_weighted_abs_error must be a finite non-negative value")
    if not math.isfinite(max_unstable_drift_share) or not 0.0 <= max_unstable_drift_share <= 1.0:
        raise ValueError("max_unstable_drift_share must be in [0, 1]")

    empty: dict[str, float | int | str] = {
        "residual_slices": 0,
        "underfilled_execution_drag_slices": 0,
        "residual_watch_slices": 0,
        "max_weighted_absolute_calibration_error": 0.0,
        "worst_residual_regime": "none",
        "worst_residual_best_execution_side": "none",
        "worst_residual_label": "none",
        "drift_bins": 0,
        "unstable_drift_bins": 0,
        "watch_drift_bins": 0,
        "unstable_drift_share": 0.0,
        "max_fill_rate_range": 0.0,
        "max_calibration_error_range": 0.0,
        "worst_drift_regime": "none",
        "worst_drift_best_execution_side": "none",
        "worst_drift_label": "none",
        "stability_cells": 0,
        "degraded_stability_cells": 0,
        "lost_stability_cells": 0,
        "stability_label": "no_queue_calibration_stability_data",
        "queue_calibration_reliability_label": "no_queue_calibration_reliability_data",
    }
    if residual_summary.empty and drift.empty and not stability_summary:
        return empty

    result = dict(empty)
    if not residual_summary.empty:
        residual_required = {
            regime_col,
            "best_execution_side",
            "rows",
            "underfilled_bins",
            "weighted_calibration_error",
            "weighted_absolute_calibration_error",
            "weighted_mean_execution_adjusted_edge_ticks",
            "residual_label",
        }
        _require_columns(
            residual_summary,
            residual_required,
            "queue position calibration reliability residual",
        )
        residual_values = _finite_values(
            residual_summary,
            [
                "rows",
                "underfilled_bins",
                "weighted_calibration_error",
                "weighted_absolute_calibration_error",
                "weighted_mean_execution_adjusted_edge_ticks",
            ],
            "queue position calibration reliability residual",
        )
        if not residual_values[["rows", "underfilled_bins"]].ge(0.0).all().all():
            raise ValueError("queue position calibration reliability residual counts must be non-negative")
        if not residual_values["weighted_absolute_calibration_error"].between(0.0, 1.0).all():
            raise ValueError("queue position calibration reliability residual errors must be in [0, 1]")
        labels = residual_summary["residual_label"].astype(str)
        worst = pd.DataFrame(
            {
                regime_col: residual_summary[regime_col].astype(str),
                "best_execution_side": residual_summary["best_execution_side"].astype(str),
                "label": labels,
                "weighted_abs_error": residual_values["weighted_absolute_calibration_error"],
                "underfill_drag": (labels == "underfilled_execution_drag").astype(int),
                "rows": residual_values["rows"],
            }
        ).sort_values(
            ["underfill_drag", "weighted_abs_error", "rows", regime_col, "best_execution_side"],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        result.update(
            {
                "residual_slices": int(len(residual_summary)),
                "underfilled_execution_drag_slices": int((labels == "underfilled_execution_drag").sum()),
                "residual_watch_slices": int(labels.isin(["calibration_residual_watch", "underfilled_but_edge_positive"]).sum()),
                "max_weighted_absolute_calibration_error": float(
                    residual_values["weighted_absolute_calibration_error"].max()
                ),
                "worst_residual_regime": str(worst[regime_col]),
                "worst_residual_best_execution_side": str(worst["best_execution_side"]),
                "worst_residual_label": str(worst["label"]),
            }
        )

    if not drift.empty:
        drift_required = {
            "best_execution_side",
            "rows",
            "fill_rate_range",
            "calibration_error_range",
            "weighted_mean_absolute_calibration_error",
            "worst_regime",
            "drift_label",
        }
        _require_columns(drift, drift_required, "queue position calibration reliability drift")
        drift_values = _finite_values(
            drift,
            [
                "rows",
                "fill_rate_range",
                "calibration_error_range",
                "weighted_mean_absolute_calibration_error",
            ],
            "queue position calibration reliability drift",
        )
        if not drift_values["rows"].ge(0.0).all():
            raise ValueError("queue position calibration reliability drift rows must be non-negative")
        for column in ["fill_rate_range", "calibration_error_range", "weighted_mean_absolute_calibration_error"]:
            if not drift_values[column].between(0.0, 1.0).all():
                raise ValueError(f"queue position calibration reliability drift {column} must be in [0, 1]")
        drift_labels = drift["drift_label"].astype(str)
        worst_drift = pd.DataFrame(
            {
                "best_execution_side": drift["best_execution_side"].astype(str),
                "worst_regime": drift["worst_regime"].astype(str),
                "label": drift_labels,
                "unstable": (drift_labels == "calibration_unstable").astype(int),
                "weighted_abs_error": drift_values["weighted_mean_absolute_calibration_error"],
                "fill_range": drift_values["fill_rate_range"],
                "calibration_range": drift_values["calibration_error_range"],
                "rows": drift_values["rows"],
            }
        ).sort_values(
            ["unstable", "weighted_abs_error", "fill_range", "calibration_range", "rows", "worst_regime"],
            ascending=[False, False, False, False, False, True],
        ).iloc[0]
        drift_bins = int(len(drift))
        unstable_bins = int((drift_labels == "calibration_unstable").sum())
        result.update(
            {
                "drift_bins": drift_bins,
                "unstable_drift_bins": unstable_bins,
                "watch_drift_bins": int((drift_labels == "calibration_watch").sum()),
                "unstable_drift_share": float(unstable_bins / drift_bins) if drift_bins else 0.0,
                "max_fill_rate_range": float(drift_values["fill_rate_range"].max()),
                "max_calibration_error_range": float(drift_values["calibration_error_range"].max()),
                "worst_drift_regime": str(worst_drift["worst_regime"]),
                "worst_drift_best_execution_side": str(worst_drift["best_execution_side"]),
                "worst_drift_label": str(worst_drift["label"]),
            }
        )

    stability_summary = stability_summary or {}
    stability_label = str(
        stability_summary.get("queue_calibration_stability_label", "no_queue_calibration_stability_data")
    )
    result.update(
        {
            "stability_cells": int(stability_summary.get("cells", 0) or 0),
            "degraded_stability_cells": int(stability_summary.get("degraded_cells", 0) or 0),
            "lost_stability_cells": int(stability_summary.get("lost_cells", 0) or 0),
            "stability_label": stability_label,
        }
    )

    underfilled_drag_slices = int(result["underfilled_execution_drag_slices"])
    lost_stability_cells = int(result["lost_stability_cells"])
    max_abs_error = float(result["max_weighted_absolute_calibration_error"])
    unstable_drift_share = float(result["unstable_drift_share"])
    degraded_stability_cells = int(result["degraded_stability_cells"])
    evidence_items = int(result["residual_slices"]) + int(result["drift_bins"]) + int(result["stability_cells"])

    if underfilled_drag_slices > 0:
        reliability_label = "queue_calibration_underfill_block"
    elif lost_stability_cells > 0 or stability_label == "queue_calibration_cells_lost":
        reliability_label = "queue_calibration_stability_block"
    elif max_abs_error > max_weighted_abs_error:
        reliability_label = "queue_calibration_residual_review"
    elif unstable_drift_share > max_unstable_drift_share:
        reliability_label = "queue_calibration_drift_review"
    elif degraded_stability_cells > 0 or stability_label == "queue_calibration_degraded":
        reliability_label = "queue_calibration_stability_review"
    elif evidence_items > 0:
        reliability_label = "queue_calibration_release_ready"
    else:
        reliability_label = "no_queue_calibration_reliability_data"
    result["queue_calibration_reliability_label"] = reliability_label
    return result


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
    *,
    toxicity_surface: pd.DataFrame | None = None,
    max_toxic_queue_row_share: float = 0.10,
) -> dict[str, float | int | str | bool]:
    """Combine queue execution quality, stability, toxicity, and capacity into one release gate.

    Execution-adjusted LCRI should not be published solely because a fill model is
    calibrated on average. This scorecard joins three orthogonal blockers:
    side-aware queue calibration quality, out-of-sample capacity stability,
    adverse-selection toxicity in queue-depth cells, and regime concentration of
    viable passive capacity. The result is a compact artifact for demos/review
    packets that states whether execution evidence is publishable or still needs
    queue-model/capacity remediation.
    """
    if not math.isfinite(max_toxic_queue_row_share) or not 0.0 <= max_toxic_queue_row_share <= 1.0:
        raise ValueError("max_toxic_queue_row_share must be in [0, 1]")
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

    toxicity_label = "not_supplied"
    toxic_queue_row_share = 0.0
    toxic_queue_regimes = 0
    worst_toxicity_regime = "none"
    worst_toxicity_adverse_to_fill_ratio = 0.0
    worst_toxicity_realized_loss_rate = 0.0
    toxic_queue_mean_edge_ticks = 0.0
    toxicity_blocked = False
    if toxicity_surface is not None:
        toxicity_required = {
            "regime",
            "rows",
            "adverse_to_fill_ratio",
            "realized_loss_rate",
            "mean_execution_adjusted_edge_ticks",
            "queue_toxicity_label",
        }
        _require_columns(toxicity_surface, toxicity_required, "queue execution readiness toxicity")
        toxicity_values = _finite_values(
            toxicity_surface,
            [
                "rows",
                "adverse_to_fill_ratio",
                "realized_loss_rate",
                "mean_execution_adjusted_edge_ticks",
            ],
            "queue execution readiness toxicity",
        )
        if (toxicity_values["rows"] < 0.0).any():
            raise ValueError("queue execution readiness toxicity rows must be non-negative")
        if (toxicity_values[["adverse_to_fill_ratio", "realized_loss_rate"]] < 0.0).any().any():
            raise ValueError("queue execution readiness toxicity rates must be non-negative")
        total_toxicity_rows = float(toxicity_values["rows"].sum())
        toxicity_labels = toxicity_surface["queue_toxicity_label"].astype(str)
        toxic_mask = toxicity_labels == "toxic_queue_fill"
        toxic_rows = float(toxicity_values.loc[toxic_mask, "rows"].sum())
        toxic_queue_row_share = toxic_rows / total_toxicity_rows if total_toxicity_rows > 0.0 else 0.0
        if toxic_rows > 0.0:
            toxic_values = toxicity_values.loc[toxic_mask]
            toxic_weights = toxic_values["rows"] / toxic_rows
            worst_toxicity_adverse_to_fill_ratio = float(
                toxic_values["adverse_to_fill_ratio"].max()
            )
            worst_toxicity_realized_loss_rate = float(toxic_values["realized_loss_rate"].max())
            toxic_queue_mean_edge_ticks = float(
                (toxic_values["mean_execution_adjusted_edge_ticks"] * toxic_weights).sum()
            )
        toxic_regime_rows = (
            toxicity_values.assign(regime=toxicity_surface["regime"].astype(str), toxic_rows=0.0)
            .assign(toxic_rows=lambda data: data["rows"].where(toxic_mask, 0.0))
            .groupby("regime", sort=True)["toxic_rows"]
            .sum()
        )
        toxic_regime_rows = toxic_regime_rows[toxic_regime_rows > 0.0]
        toxic_queue_regimes = int(len(toxic_regime_rows))
        if toxic_queue_regimes:
            worst_toxicity_regime = str(toxic_regime_rows.idxmax())
        toxicity_blocked = toxic_queue_row_share > max_toxic_queue_row_share
        if toxicity_blocked:
            toxicity_label = "toxic_queue_blocked"
        elif toxic_queue_regimes:
            toxicity_label = "toxic_queue_review"
        else:
            toxicity_label = "benign_queue_toxicity"

    quality_blocked = str(quality_gate["quality_gate_label"]) == "queue_execution_blocked"
    stability_blocked = str(capacity_stability["capacity_stability_label"]) == "capacity_fragile"
    blocker_count = (
        int(quality_blocked)
        + int(stability_blocked)
        + int(concentration_blocked)
        + int(toxicity_blocked)
    )
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
        "queue_toxicity_label": toxicity_label,
        "blocked_regimes": int(quality_numeric["blocked_regimes"]),
        "eligible_regimes": int(quality_numeric["eligible_regimes"]),
        "execution_blocker_count": blocker_count,
        "worst_calibration_regime": str(quality_gate["worst_calibration_regime"]),
        "worst_decay_regime": str(quality_gate["worst_decay_regime"]),
        "worst_capacity_regime": worst_capacity_regime,
        "worst_toxicity_regime": worst_toxicity_regime,
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
        "toxic_queue_row_share": toxic_queue_row_share,
        "toxic_queue_regimes": toxic_queue_regimes,
        "worst_toxicity_adverse_to_fill_ratio": worst_toxicity_adverse_to_fill_ratio,
        "worst_toxicity_realized_loss_rate": worst_toxicity_realized_loss_rate,
        "toxic_queue_mean_edge_ticks": toxic_queue_mean_edge_ticks,
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


def passive_fill_brier_decomposition(curve: pd.DataFrame) -> dict[str, float | int | str]:
    """Decompose passive-fill Brier loss into calibration and discrimination terms.

    ``passive_fill_calibration_curve`` already reports per-bin Brier loss. This
    summary adds the Murphy decomposition reviewers expect for publishable
    probabilistic forecasts: reliability (calibration penalty), resolution
    (separation of queue states from the base fill rate), uncertainty (base-rate
    variance), and Brier skill versus a constant base-rate fill model.
    """
    if curve.empty:
        return _empty_passive_fill_brier_decomposition()
    required = {
        "rows",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "brier_score",
    }
    _require_columns(curve, required, "passive fill brier decomposition")
    values = _finite_values(
        curve,
        ["rows", "mean_predicted_fill_probability", "realized_fill_rate", "brier_score"],
        "passive fill brier decomposition",
    )
    if (values["rows"] < 0.0).any():
        raise ValueError("passive fill brier decomposition rows must be non-negative")
    if not values["mean_predicted_fill_probability"].between(0.0, 1.0).all():
        raise ValueError("passive fill brier decomposition probabilities must be in [0, 1]")
    if not values["realized_fill_rate"].between(0.0, 1.0).all():
        raise ValueError("passive fill brier decomposition fill rates must be in [0, 1]")
    if (values["brier_score"] < 0.0).any():
        raise ValueError("passive fill brier decomposition brier scores must be non-negative")

    total_rows = int(values["rows"].sum())
    if total_rows == 0:
        return _empty_passive_fill_brier_decomposition()
    weights = values["rows"] / total_rows
    predicted = values["mean_predicted_fill_probability"]
    realized = values["realized_fill_rate"]
    base_fill_rate = float((realized * weights).sum())
    weighted_brier = float((values["brier_score"] * weights).sum())
    uncertainty = float(base_fill_rate * (1.0 - base_fill_rate))
    reliability = float((((predicted - realized) ** 2) * weights).sum())
    resolution = float((((realized - base_fill_rate) ** 2) * weights).sum())
    brier_skill = 0.0 if uncertainty == 0.0 else float(1.0 - weighted_brier / uncertainty)
    decomposition_error = float(weighted_brier - (reliability - resolution + uncertainty))
    return {
        "rows": total_rows,
        "bins": int(len(curve)),
        "base_fill_rate": base_fill_rate,
        "weighted_brier_score": weighted_brier,
        "uncertainty": uncertainty,
        "reliability": reliability,
        "resolution": resolution,
        "brier_skill_score": brier_skill,
        "brier_decomposition_error": decomposition_error,
        "calibration_quality_label": _passive_fill_calibration_quality_label(
            reliability=reliability,
            resolution=resolution,
            brier_skill_score=brier_skill,
        ),
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


def _passive_fill_hazard_label(slippage: float, tolerance: float) -> str:
    if slippage > tolerance:
        return "over_realized"
    if slippage < -tolerance:
        return "under_realized"
    return "near_prediction"


def passive_fill_realization_hazard_curve(
    frame: pd.DataFrame,
    *,
    horizons: list[int] | tuple[int, ...] = (1, 2, 3, 5),
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Measure when queue-position passive fills materialize across horizons.

    The calibration sweep reports cumulative quality at each horizon. This curve
    decomposes that cumulative rate into incremental realized fills and a
    conditional fill hazard by optional regime, making delayed-fill dependence
    explicit before LCRI edges are called executable.
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
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    required = {side_col, "bid_fill_probability", "ask_fill_probability"}
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "passive fill realization hazard curve")
    values = _finite_values(
        frame,
        ["bid_fill_probability", "ask_fill_probability"],
        "passive fill realization hazard curve",
    )
    side = frame[side_col].astype(str)
    eligible = side.isin(["long", "short"])
    selected_probability = pd.Series(
        np.select(
            [side == "long", side == "short"],
            [values["bid_fill_probability"], values["ask_fill_probability"]],
            default=0.0,
        ),
        index=frame.index,
        dtype=float,
    )
    regime_name = regime_col or "regime"
    regime_values = frame[regime_col].astype(str) if regime_col is not None else pd.Series("all", index=frame.index)

    cumulative_by_horizon: dict[int, pd.Series] = {}
    for horizon in sorted(horizons):
        realized = add_queue_position_realized_fill_proxy(
            frame,
            horizon=horizon,
            group_cols=group_cols,
            bid_realized_col="_hazard_bid_realized_fill",
            ask_realized_col="_hazard_ask_realized_fill",
        )
        realized_values = _finite_values(
            realized,
            ["_hazard_bid_realized_fill", "_hazard_ask_realized_fill"],
            "passive fill realization hazard curve",
        )
        cumulative_by_horizon[horizon] = pd.Series(
            np.select(
                [side == "long", side == "short"],
                [
                    realized_values["_hazard_bid_realized_fill"],
                    realized_values["_hazard_ask_realized_fill"],
                ],
                default=0.0,
            ),
            index=frame.index,
            dtype=float,
        )

    rows: list[dict[str, float | int | str]] = []
    working = pd.DataFrame(
        {
            regime_name: regime_values,
            "eligible": eligible,
            "selected_probability": selected_probability,
        },
        index=frame.index,
    )
    regimes = working.loc[eligible, regime_name].drop_duplicates().tolist()
    for regime in regimes:
        mask = eligible & (working[regime_name] == regime)
        eligible_rows = int(mask.sum())
        previous_rate = 0.0
        for horizon in sorted(horizons):
            cumulative_rate = float(cumulative_by_horizon[horizon].loc[mask].mean()) if eligible_rows else 0.0
            incremental = max(0.0, cumulative_rate - previous_rate)
            survival = max(0.0, 1.0 - previous_rate)
            hazard = incremental / survival if survival > 0.0 else 0.0
            predicted = float(working.loc[mask, "selected_probability"].mean()) if eligible_rows else 0.0
            slippage = cumulative_rate - predicted
            rows.append(
                {
                    regime_name: str(regime),
                    "horizon": int(horizon),
                    "eligible_rows": eligible_rows,
                    "mean_selected_fill_probability": predicted,
                    "cumulative_realized_fill_rate": cumulative_rate,
                    "incremental_realized_fill_rate": incremental,
                    "conditional_fill_hazard": hazard,
                    "survival_rate_entering_horizon": survival,
                    "timing_slippage_vs_prediction": slippage,
                    "horizon_timing_label": _passive_fill_hazard_label(slippage, tolerance),
                }
            )
            previous_rate = cumulative_rate
    columns = [
        regime_name,
        "horizon",
        "eligible_rows",
        "mean_selected_fill_probability",
        "cumulative_realized_fill_rate",
        "incremental_realized_fill_rate",
        "conditional_fill_hazard",
        "survival_rate_entering_horizon",
        "timing_slippage_vs_prediction",
        "horizon_timing_label",
    ]
    return pd.DataFrame(rows, columns=columns)


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


def _empty_passive_fill_event_window_transition_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "from_passive_fill_event_window_regime",
            "to_passive_fill_event_window_regime",
            "rows",
            "transition_share",
            "mean_from_execution_adjusted_edge_ticks",
            "mean_to_execution_adjusted_edge_ticks",
            "mean_edge_delta_ticks",
            "to_negative_edge_share",
            "mean_to_passive_fill_event_toxicity_probability",
            "dominant_to_passive_fill_event_side",
        ]
    )


def _empty_passive_fill_event_window_regime_scorecard() -> dict[str, float | int | str]:
    return {
        "regimes": 0,
        "total_rows": 0,
        "event_rows": 0,
        "event_row_share": 0.0,
        "post_event_rows": 0,
        "post_event_negative_edge_share": 0.0,
        "post_event_mean_toxicity_probability": 0.0,
        "event_mean_execution_adjusted_edge_ticks": 0.0,
        "worst_regime_by_toxicity": "none",
        "worst_regime_toxicity_probability": 0.0,
        "worst_regime_negative_edge_share": 0.0,
        "event_window_release_label": "insufficient_event_window_evidence",
    }


def passive_fill_event_window_regime_scorecard(
    summary: pd.DataFrame,
    *,
    min_event_rows: int = 1,
    max_post_event_negative_edge_share: float = 0.50,
    max_post_event_toxicity_probability: float = 0.50,
    min_event_mean_edge_ticks: float = 0.0,
) -> dict[str, float | int | str]:
    """Gate passive-fill event-window regimes for execution publishability.

    ``passive_fill_event_window_regime_summary`` identifies where high-probability
    passive fills occur and what happens immediately around them. This scorecard
    compresses that surface into release-ready evidence: enough executable event
    rows, positive event-time edge, and no toxic post-event reversal where fills
    are available but quickly followed by adverse execution economics.
    """
    if not isinstance(min_event_rows, int) or isinstance(min_event_rows, bool):
        raise ValueError("min_event_rows must be an integer")
    if min_event_rows < 0:
        raise ValueError("min_event_rows must be non-negative")
    for name, value in {
        "max_post_event_negative_edge_share": max_post_event_negative_edge_share,
        "max_post_event_toxicity_probability": max_post_event_toxicity_probability,
        "min_event_mean_edge_ticks": min_event_mean_edge_ticks,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_post_event_negative_edge_share <= 1.0:
        raise ValueError("max_post_event_negative_edge_share must be in [0.0, 1.0]")
    if not 0.0 <= max_post_event_toxicity_probability <= 1.0:
        raise ValueError("max_post_event_toxicity_probability must be in [0.0, 1.0]")

    required = {
        "passive_fill_event_window_regime",
        "rows",
        "event_rows",
        "row_share",
        "mean_passive_fill_event_toxicity_probability",
        "mean_execution_adjusted_edge_ticks",
        "negative_edge_share",
    }
    _require_columns(summary, required, "passive fill event window regime scorecard")
    if summary.empty:
        return _empty_passive_fill_event_window_regime_scorecard()

    data = summary[list(required)].copy()
    numeric_columns = [
        "rows",
        "event_rows",
        "row_share",
        "mean_passive_fill_event_toxicity_probability",
        "mean_execution_adjusted_edge_ticks",
        "negative_edge_share",
    ]
    values = _finite_values(data, numeric_columns, "passive fill event window regime scorecard")
    if (values[["rows", "event_rows", "row_share", "negative_edge_share"]] < 0.0).any().any():
        raise ValueError("passive fill event window regime scorecard values must be non-negative")
    if (values[["row_share", "negative_edge_share"]] > 1.0).any().any():
        raise ValueError("passive fill event window regime scorecard shares must be in [0.0, 1.0]")
    data[numeric_columns] = values[numeric_columns]
    data["passive_fill_event_window_regime"] = data[
        "passive_fill_event_window_regime"
    ].astype(str)

    total_rows = int(round(float(data["rows"].sum())))
    event_rows = int(round(float(data["event_rows"].sum())))
    event_row_share = float(event_rows) / float(total_rows) if total_rows else 0.0
    post = data[data["passive_fill_event_window_regime"] == "post_event"]
    event = data[data["passive_fill_event_window_regime"] == "event"]
    post_event_rows = int(round(float(post["rows"].sum()))) if not post.empty else 0
    post_event_negative_edge_share = (
        float(np.average(post["negative_edge_share"], weights=post["rows"])) if post_event_rows else 0.0
    )
    post_event_toxicity = (
        float(
            np.average(
                post["mean_passive_fill_event_toxicity_probability"],
                weights=post["rows"],
            )
        )
        if post_event_rows
        else 0.0
    )
    event_mean_edge = (
        float(np.average(event["mean_execution_adjusted_edge_ticks"], weights=event["rows"]))
        if not event.empty and float(event["rows"].sum()) > 0.0
        else 0.0
    )
    worst_idx = data["mean_passive_fill_event_toxicity_probability"].idxmax()
    worst = data.loc[worst_idx]

    if event_rows < min_event_rows:
        label = "insufficient_event_window_evidence"
    elif post_event_rows and (
        post_event_negative_edge_share > max_post_event_negative_edge_share
        or post_event_toxicity > max_post_event_toxicity_probability
    ):
        label = "toxic_post_event_reversal"
    elif event_mean_edge <= min_event_mean_edge_ticks:
        label = "nonpositive_event_edge"
    else:
        label = "event_window_execution_ready"

    return {
        "regimes": int(len(data)),
        "total_rows": total_rows,
        "event_rows": event_rows,
        "event_row_share": event_row_share,
        "post_event_rows": post_event_rows,
        "post_event_negative_edge_share": post_event_negative_edge_share,
        "post_event_mean_toxicity_probability": post_event_toxicity,
        "event_mean_execution_adjusted_edge_ticks": event_mean_edge,
        "worst_regime_by_toxicity": str(worst["passive_fill_event_window_regime"]),
        "worst_regime_toxicity_probability": float(
            worst["mean_passive_fill_event_toxicity_probability"]
        ),
        "worst_regime_negative_edge_share": float(worst["negative_edge_share"]),
        "event_window_release_label": label,
    }


def passive_fill_event_window_transition_matrix(
    frame: pd.DataFrame,
    *,
    regime_col: str = "passive_fill_event_window_regime",
    event_side_col: str = "passive_fill_event_side",
    toxicity_probability_col: str = "passive_fill_event_toxicity_probability",
    edge_col: str = "execution_adjusted_edge_ticks",
    group_cols: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Summarize one-step passive-fill event-window regime transitions.

    Event-window aggregates can hide the path that produced fragility: pre-event
    buildup may decay smoothly into events, while event-to-post-event paths often
    reveal adverse selection after fills become available. This transition matrix
    keeps the chronology intact by shifting regimes only within optional symbol,
    venue, or session groups, then reports edge decay, next-state toxicity, and
    dominant destination side for each observed path.
    """
    columns = list(_empty_passive_fill_event_window_transition_matrix().columns)
    if frame.empty:
        return _empty_passive_fill_event_window_transition_matrix()
    grouping_columns = _normalize_group_columns(
        frame, group_cols, "passive fill event window transition matrix group"
    )
    required = {regime_col, event_side_col, toxicity_probability_col, edge_col}
    _require_columns(frame, required, "passive fill event window transition matrix")
    values = _finite_values(
        frame,
        [toxicity_probability_col, edge_col],
        "passive fill event window transition matrix",
    )
    data = frame[[regime_col, event_side_col]].copy()
    data[toxicity_probability_col] = values[toxicity_probability_col]
    data[edge_col] = values[edge_col]

    if grouping_columns:
        keys = [frame[group_col] for group_col in grouping_columns]
        data["to_regime"] = data[regime_col].groupby(keys, sort=False, dropna=False).shift(-1)
        data["to_side"] = data[event_side_col].groupby(keys, sort=False, dropna=False).shift(-1)
        data["to_toxicity"] = data[toxicity_probability_col].groupby(
            keys, sort=False, dropna=False
        ).shift(-1)
        data["to_edge"] = data[edge_col].groupby(keys, sort=False, dropna=False).shift(-1)
    else:
        data["to_regime"] = data[regime_col].shift(-1)
        data["to_side"] = data[event_side_col].shift(-1)
        data["to_toxicity"] = data[toxicity_probability_col].shift(-1)
        data["to_edge"] = data[edge_col].shift(-1)

    transitions = data.dropna(subset=["to_regime", "to_side", "to_toxicity", "to_edge"]).copy()
    if transitions.empty:
        return _empty_passive_fill_event_window_transition_matrix()
    transitions["edge_delta"] = transitions["to_edge"] - transitions[edge_col]
    total_transitions = float(len(transitions))
    rows: list[dict[str, float | int | str]] = []
    for (from_regime, to_regime), group in transitions.groupby([regime_col, "to_regime"], sort=True):
        side_counts = group["to_side"].astype(str).value_counts()
        dominant_side = "none" if side_counts.empty else str(side_counts.idxmax())
        rows.append(
            {
                "from_passive_fill_event_window_regime": str(from_regime),
                "to_passive_fill_event_window_regime": str(to_regime),
                "rows": int(len(group)),
                "transition_share": float(len(group)) / total_transitions if total_transitions else 0.0,
                "mean_from_execution_adjusted_edge_ticks": float(group[edge_col].mean()),
                "mean_to_execution_adjusted_edge_ticks": float(group["to_edge"].mean()),
                "mean_edge_delta_ticks": float(group["edge_delta"].mean()),
                "to_negative_edge_share": float((group["to_edge"] < 0.0).sum()) / float(len(group)),
                "mean_to_passive_fill_event_toxicity_probability": float(group["to_toxicity"].mean()),
                "dominant_to_passive_fill_event_side": dominant_side,
            }
        )
    return (
        pd.DataFrame(rows)[columns]
        .sort_values(
            [
                "to_negative_edge_share",
                "mean_to_passive_fill_event_toxicity_probability",
                "mean_edge_delta_ticks",
                "rows",
            ],
            ascending=[False, False, True, False],
        )
        .reset_index(drop=True)
    )


def _empty_passive_fill_event_window_transition_scorecard() -> dict[str, float | int | str]:
    return {
        "observed_transition_paths": 0,
        "total_transition_rows": 0,
        "eligible_transition_paths": 0,
        "blocked_transition_paths": 0,
        "review_transition_paths": 0,
        "worst_transition_path": "none",
        "worst_path_rows": 0,
        "worst_path_transition_share": 0.0,
        "worst_path_mean_edge_delta_ticks": 0.0,
        "worst_path_to_negative_edge_share": 0.0,
        "worst_path_to_toxicity_probability": 0.0,
        "candidate_weighted_mean_edge_delta_ticks": 0.0,
        "candidate_weighted_to_negative_edge_share": 0.0,
        "candidate_weighted_to_toxicity_probability": 0.0,
        "transition_release_label": "pass",
        "blocking_reasons": "none",
        "review_reasons": "none",
    }


def passive_fill_event_window_transition_scorecard(
    transition_matrix: pd.DataFrame,
    *,
    min_transition_rows: int = 1,
    block_event_post_delta_ticks: float = -0.50,
    block_negative_edge_share: float = 0.60,
    block_toxicity_probability: float = 0.75,
    review_edge_delta_ticks: float = -0.25,
    review_negative_edge_share: float = 0.40,
    review_toxicity_probability: float = 0.60,
) -> dict[str, float | int | str]:
    """Gate event-window transition decay before publishing passive-fill evidence.

    Static event-window buckets can look acceptable while the chronological path
    from executable rows into post-event rows loses edge immediately. This compact
    scorecard turns the transition matrix into a release label, emphasizing the
    economically important event→post_event path while still surfacing broader
    transition-level edge decay for reviewer triage.
    """
    if not isinstance(min_transition_rows, int) or isinstance(min_transition_rows, bool):
        raise ValueError("min_transition_rows must be a positive integer")
    if min_transition_rows < 1:
        raise ValueError("min_transition_rows must be a positive integer")
    for name, value in {
        "block_event_post_delta_ticks": block_event_post_delta_ticks,
        "review_edge_delta_ticks": review_edge_delta_ticks,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name, value in {
        "block_negative_edge_share": block_negative_edge_share,
        "block_toxicity_probability": block_toxicity_probability,
        "review_negative_edge_share": review_negative_edge_share,
        "review_toxicity_probability": review_toxicity_probability,
    }.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if transition_matrix.empty:
        return _empty_passive_fill_event_window_transition_scorecard()

    required = {
        "from_passive_fill_event_window_regime",
        "to_passive_fill_event_window_regime",
        "rows",
        "transition_share",
        "mean_edge_delta_ticks",
        "to_negative_edge_share",
        "mean_to_passive_fill_event_toxicity_probability",
    }
    _require_columns(
        transition_matrix,
        required,
        "passive fill event window transition scorecard",
    )
    numeric_columns = [
        "rows",
        "transition_share",
        "mean_edge_delta_ticks",
        "to_negative_edge_share",
        "mean_to_passive_fill_event_toxicity_probability",
    ]
    values = _finite_values(
        transition_matrix,
        numeric_columns,
        "passive fill event window transition scorecard",
    )
    if (values["rows"] < 0.0).any():
        raise ValueError("passive fill event window transition scorecard rows must be non-negative")
    data = values.copy()
    data["from_regime"] = transition_matrix["from_passive_fill_event_window_regime"].astype(str)
    data["to_regime"] = transition_matrix["to_passive_fill_event_window_regime"].astype(str)
    data["transition_path"] = data["from_regime"] + "->" + data["to_regime"]
    total_rows = int(data["rows"].sum())
    if total_rows == 0:
        scorecard = _empty_passive_fill_event_window_transition_scorecard()
        scorecard["observed_transition_paths"] = int(len(transition_matrix))
        return scorecard

    eligible = data[data["rows"] >= float(min_transition_rows)].copy()
    weights = data["rows"] / float(total_rows)
    if eligible.empty:
        scorecard = _empty_passive_fill_event_window_transition_scorecard()
        scorecard.update(
            {
                "observed_transition_paths": int(len(transition_matrix)),
                "total_transition_rows": total_rows,
                "candidate_weighted_mean_edge_delta_ticks": float(
                    (data["mean_edge_delta_ticks"] * weights).sum()
                ),
                "candidate_weighted_to_negative_edge_share": float(
                    (data["to_negative_edge_share"] * weights).sum()
                ),
                "candidate_weighted_to_toxicity_probability": float(
                    (
                        data["mean_to_passive_fill_event_toxicity_probability"]
                        * weights
                    ).sum()
                ),
                "transition_release_label": "review",
                "review_reasons": "insufficient_transition_rows",
            }
        )
        return scorecard

    event_post = eligible[
        (eligible["from_regime"] == "event") & (eligible["to_regime"] == "post_event")
    ]
    toxic_event_post = event_post[
        (event_post["mean_edge_delta_ticks"] <= block_event_post_delta_ticks)
        & (event_post["to_negative_edge_share"] >= block_negative_edge_share)
        & (
            event_post["mean_to_passive_fill_event_toxicity_probability"]
            >= block_toxicity_probability
        )
    ]
    review_paths = eligible[
        (eligible["mean_edge_delta_ticks"] <= review_edge_delta_ticks)
        | (eligible["to_negative_edge_share"] >= review_negative_edge_share)
        | (
            eligible["mean_to_passive_fill_event_toxicity_probability"]
            >= review_toxicity_probability
        )
    ]
    if not toxic_event_post.empty:
        worst_idx = toxic_event_post.sort_values(
            [
                "mean_edge_delta_ticks",
                "to_negative_edge_share",
                "mean_to_passive_fill_event_toxicity_probability",
                "rows",
            ],
            ascending=[True, False, False, False],
        ).index[0]
    else:
        worst_idx = eligible.sort_values(
            [
                "to_negative_edge_share",
                "mean_to_passive_fill_event_toxicity_probability",
                "mean_edge_delta_ticks",
                "rows",
            ],
            ascending=[False, False, True, False],
        ).index[0]
    blocking_reasons = []
    if not toxic_event_post.empty:
        blocking_reasons.append("toxic_event_post_event_decay")
    review_reasons = []
    if toxic_event_post.empty and not review_paths.empty:
        review_reasons.append("meaningful_transition_edge_decay")
    label = "block" if blocking_reasons else "review" if review_reasons else "pass"

    return {
        "observed_transition_paths": int(len(transition_matrix)),
        "total_transition_rows": total_rows,
        "eligible_transition_paths": int(len(eligible)),
        "blocked_transition_paths": int(len(toxic_event_post)),
        "review_transition_paths": int(len(review_paths)),
        "worst_transition_path": str(data.loc[worst_idx, "transition_path"]),
        "worst_path_rows": int(data.loc[worst_idx, "rows"]),
        "worst_path_transition_share": float(data.loc[worst_idx, "transition_share"]),
        "worst_path_mean_edge_delta_ticks": float(data.loc[worst_idx, "mean_edge_delta_ticks"]),
        "worst_path_to_negative_edge_share": float(data.loc[worst_idx, "to_negative_edge_share"]),
        "worst_path_to_toxicity_probability": float(
            data.loc[worst_idx, "mean_to_passive_fill_event_toxicity_probability"]
        ),
        "candidate_weighted_mean_edge_delta_ticks": float(
            (data["mean_edge_delta_ticks"] * weights).sum()
        ),
        "candidate_weighted_to_negative_edge_share": float(
            (data["to_negative_edge_share"] * weights).sum()
        ),
        "candidate_weighted_to_toxicity_probability": float(
            (data["mean_to_passive_fill_event_toxicity_probability"] * weights).sum()
        ),
        "transition_release_label": label,
        "blocking_reasons": ",".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ",".join(review_reasons) if review_reasons else "none",
    }


def _empty_passive_fill_event_window_transition_stability() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "transition_path",
            "from_passive_fill_event_window_regime",
            "to_passive_fill_event_window_regime",
            "rows_train",
            "rows_heldout",
            "transition_share_train",
            "transition_share_heldout",
            "transition_share_drift",
            "mean_edge_delta_ticks_train",
            "mean_edge_delta_ticks_heldout",
            "edge_delta_drift_ticks",
            "to_negative_edge_share_train",
            "to_negative_edge_share_heldout",
            "negative_edge_share_drift",
            "mean_to_passive_fill_event_toxicity_probability_train",
            "mean_to_passive_fill_event_toxicity_probability_heldout",
            "toxicity_probability_drift",
            "transition_stability_label",
        ]
    )


def passive_fill_event_window_transition_stability(
    train_transition_matrix: pd.DataFrame,
    heldout_transition_matrix: pd.DataFrame,
    *,
    min_train_rows: int = 1,
    min_heldout_rows: int = 1,
    block_edge_delta_drift_ticks: float = -0.50,
    block_negative_edge_share_drift: float = 0.25,
    block_toxicity_probability_drift: float = 0.25,
    review_edge_delta_drift_ticks: float = -0.25,
    review_negative_edge_share_drift: float = 0.15,
    review_toxicity_probability_drift: float = 0.15,
) -> pd.DataFrame:
    """Compare in-sample and heldout event-window transition fragility.

    A publishable execution demo should not only pass the transition scorecard on
    a single split: the same event-window paths should remain economically stable
    out of sample. This helper joins train/heldout transition matrices by path and
    flags paths where heldout edge decay, negative-edge share, or toxicity worsens.
    """
    for name, value in {"min_train_rows": min_train_rows, "min_heldout_rows": min_heldout_rows}.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in {
        "block_edge_delta_drift_ticks": block_edge_delta_drift_ticks,
        "block_negative_edge_share_drift": block_negative_edge_share_drift,
        "block_toxicity_probability_drift": block_toxicity_probability_drift,
        "review_edge_delta_drift_ticks": review_edge_delta_drift_ticks,
        "review_negative_edge_share_drift": review_negative_edge_share_drift,
        "review_toxicity_probability_drift": review_toxicity_probability_drift,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if train_transition_matrix.empty or heldout_transition_matrix.empty:
        return _empty_passive_fill_event_window_transition_stability()

    required = {
        "from_passive_fill_event_window_regime",
        "to_passive_fill_event_window_regime",
        "rows",
        "transition_share",
        "mean_edge_delta_ticks",
        "to_negative_edge_share",
        "mean_to_passive_fill_event_toxicity_probability",
    }
    _require_columns(
        train_transition_matrix,
        required,
        "train passive fill event window transition stability",
    )
    _require_columns(
        heldout_transition_matrix,
        required,
        "heldout passive fill event window transition stability",
    )
    numeric_columns = sorted(required - {"from_passive_fill_event_window_regime", "to_passive_fill_event_window_regime"})

    def _prepared(matrix: pd.DataFrame, label: str) -> pd.DataFrame:
        values = _finite_values(matrix, numeric_columns, label)
        if (values["rows"] < 0.0).any():
            raise ValueError(f"{label} rows must be non-negative")
        data = values.copy()
        data["from_passive_fill_event_window_regime"] = matrix[
            "from_passive_fill_event_window_regime"
        ].astype(str)
        data["to_passive_fill_event_window_regime"] = matrix[
            "to_passive_fill_event_window_regime"
        ].astype(str)
        data["transition_path"] = (
            data["from_passive_fill_event_window_regime"]
            + "->"
            + data["to_passive_fill_event_window_regime"]
        )
        return data

    train = _prepared(train_transition_matrix, "train passive fill event window transition stability")
    heldout = _prepared(heldout_transition_matrix, "heldout passive fill event window transition stability")
    joined = train.merge(heldout, on="transition_path", suffixes=("_train", "_heldout"))
    if joined.empty:
        return _empty_passive_fill_event_window_transition_stability()
    joined = joined[
        (joined["rows_train"] >= float(min_train_rows))
        & (joined["rows_heldout"] >= float(min_heldout_rows))
    ].copy()
    if joined.empty:
        return _empty_passive_fill_event_window_transition_stability()

    joined["from_passive_fill_event_window_regime"] = joined[
        "from_passive_fill_event_window_regime_train"
    ]
    joined["to_passive_fill_event_window_regime"] = joined[
        "to_passive_fill_event_window_regime_train"
    ]
    joined["transition_share_drift"] = (
        joined["transition_share_heldout"] - joined["transition_share_train"]
    )
    joined["edge_delta_drift_ticks"] = (
        joined["mean_edge_delta_ticks_heldout"] - joined["mean_edge_delta_ticks_train"]
    )
    joined["negative_edge_share_drift"] = (
        joined["to_negative_edge_share_heldout"] - joined["to_negative_edge_share_train"]
    )
    joined["toxicity_probability_drift"] = (
        joined["mean_to_passive_fill_event_toxicity_probability_heldout"]
        - joined["mean_to_passive_fill_event_toxicity_probability_train"]
    )

    block_mask = (
        (joined["edge_delta_drift_ticks"] <= block_edge_delta_drift_ticks)
        | (joined["negative_edge_share_drift"] >= block_negative_edge_share_drift)
        | (joined["toxicity_probability_drift"] >= block_toxicity_probability_drift)
    )
    review_mask = (
        (joined["edge_delta_drift_ticks"] <= review_edge_delta_drift_ticks)
        | (joined["negative_edge_share_drift"] >= review_negative_edge_share_drift)
        | (joined["toxicity_probability_drift"] >= review_toxicity_probability_drift)
    )
    joined["transition_stability_label"] = np.select(
        [block_mask, review_mask],
        ["transition_stability_block", "transition_stability_review"],
        default="transition_stability_pass",
    )
    columns = list(_empty_passive_fill_event_window_transition_stability().columns)
    return (
        joined[columns]
        .sort_values(
            ["transition_stability_label", "edge_delta_drift_ticks", "negative_edge_share_drift", "rows_heldout"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )


def _empty_passive_fill_event_window_transition_stability_scorecard() -> dict[str, float | int | str | bool]:
    return {
        "evaluated_transition_paths": 0,
        "blocked_transition_paths": 0,
        "review_transition_paths": 0,
        "worst_transition_path": "none",
        "worst_edge_delta_drift_ticks": 0.0,
        "worst_negative_edge_share_drift": 0.0,
        "worst_toxicity_probability_drift": 0.0,
        "transition_stability_release_label": "pass",
        "publishable": True,
        "blocking_reasons": "none",
        "review_reasons": "none",
    }


def passive_fill_event_window_transition_stability_scorecard(
    stability: pd.DataFrame,
) -> dict[str, float | int | str | bool]:
    """Gate train-vs-heldout stability for passive-fill transition artifacts."""
    if stability.empty:
        return _empty_passive_fill_event_window_transition_stability_scorecard()
    required = {
        "transition_path",
        "rows_train",
        "rows_heldout",
        "edge_delta_drift_ticks",
        "negative_edge_share_drift",
        "toxicity_probability_drift",
        "transition_stability_label",
    }
    _require_columns(stability, required, "passive fill event window transition stability scorecard")
    values = _finite_values(
        stability,
        [
            "rows_train",
            "rows_heldout",
            "edge_delta_drift_ticks",
            "negative_edge_share_drift",
            "toxicity_probability_drift",
        ],
        "passive fill event window transition stability scorecard",
    )
    data = values.copy()
    data["transition_path"] = stability["transition_path"].astype(str)
    data["transition_stability_label"] = stability["transition_stability_label"].astype(str)
    blocked = data[data["transition_stability_label"] == "transition_stability_block"]
    review = data[data["transition_stability_label"] == "transition_stability_review"]
    worst_source = blocked if not blocked.empty else review if not review.empty else data
    worst_idx = worst_source.sort_values(
        ["edge_delta_drift_ticks", "negative_edge_share_drift", "toxicity_probability_drift", "rows_heldout"],
        ascending=[True, False, False, False],
    ).index[0]
    event_post_block = (
        not blocked[blocked["transition_path"] == "event->post_event"].empty
    )
    blocking_reasons: list[str] = []
    if event_post_block:
        blocking_reasons.append("event_post_holdout_decay")
    if not blocked.empty:
        blocking_reasons.append("unstable_transition_paths")
    review_reasons = ["transition_stability_review_paths"] if blocked.empty and not review.empty else []
    label = "block" if blocking_reasons else "review" if review_reasons else "pass"
    return {
        "evaluated_transition_paths": int(len(stability)),
        "blocked_transition_paths": int(len(blocked)),
        "review_transition_paths": int(len(review)),
        "worst_transition_path": str(data.loc[worst_idx, "transition_path"]),
        "worst_edge_delta_drift_ticks": float(data.loc[worst_idx, "edge_delta_drift_ticks"]),
        "worst_negative_edge_share_drift": float(data.loc[worst_idx, "negative_edge_share_drift"]),
        "worst_toxicity_probability_drift": float(data.loc[worst_idx, "toxicity_probability_drift"]),
        "transition_stability_release_label": label,
        "publishable": label == "pass",
        "blocking_reasons": ",".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ",".join(review_reasons) if review_reasons else "none",
    }


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


def passive_fill_event_window_sensitivity(
    frame: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90),
    windows: tuple[int, ...] = (1, 3, 5),
    side_col: str = "best_execution_side",
    long_return_col: str = "long_net_return_ticks",
    short_return_col: str = "short_net_return_ticks",
    regime_col: str | None = None,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    max_adverse_post_edge_share: float = 0.60,
    min_mean_post_minus_pre_edge: float = -0.25,
    min_events: int = 1,
) -> pd.DataFrame:
    """Stress-test passive-fill event toxicity across threshold/window choices.

    A single event-window threshold can make execution-adjusted LCRI look either
    publishable or fragile by accident. This sensitivity surface reruns the full
    event-window diagnostic, regime summary, and toxicity scorecard for each
    threshold/window pair so review artifacts can distinguish robust execution
    toxicity from a hyperparameter artifact.
    """
    columns = [
        "threshold",
        "window",
        "event_rows",
        "summary_rows",
        "regimes",
        "total_events",
        "eligible_regimes",
        "blocked_regimes",
        "worst_regime",
        "worst_adverse_post_edge_share",
        "worst_mean_post_minus_pre_realized_edge",
        "worst_post_minus_pre_realized_edge",
        "weighted_mean_event_fill_probability",
        "weighted_mean_event_adverse_fill_probability",
        "weighted_mean_post_minus_pre_realized_edge",
        "event_toxicity_label",
        "sensitivity_label",
    ]
    if not thresholds:
        raise ValueError("thresholds must be a non-empty sequence")
    if not windows:
        raise ValueError("windows must be a non-empty sequence")
    clean_thresholds = [float(threshold) for threshold in thresholds]
    if any(not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0 for threshold in clean_thresholds):
        raise ValueError("threshold values must be in [0.0, 1.0]")
    clean_windows: list[int] = []
    for window in windows:
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError("window values must be positive integers")
        clean_windows.append(int(window))

    rows: list[dict[str, float | int | str]] = []
    for threshold in sorted(clean_thresholds):
        for window in sorted(clean_windows):
            events = passive_fill_event_window_diagnostics(
                frame,
                threshold=threshold,
                window=window,
                side_col=side_col,
                long_return_col=long_return_col,
                short_return_col=short_return_col,
                regime_col=regime_col,
                group_cols=group_cols,
            )
            summary = passive_fill_event_regime_summary(events)
            scorecard = passive_fill_event_toxicity_scorecard(
                summary,
                max_adverse_post_edge_share=max_adverse_post_edge_share,
                min_mean_post_minus_pre_edge=min_mean_post_minus_pre_edge,
                min_events=min_events,
            )
            label = str(scorecard["event_toxicity_label"])
            if label == "event_window_pass":
                sensitivity_label = "event_window_threshold_pass"
            elif label == "event_window_blocker":
                sensitivity_label = "event_window_threshold_blocker"
            else:
                sensitivity_label = "event_window_threshold_insufficient_sample"
            rows.append(
                {
                    "threshold": float(threshold),
                    "window": int(window),
                    "event_rows": int(len(events)),
                    "summary_rows": int(len(summary)),
                    "regimes": int(scorecard["regimes"]),
                    "total_events": int(scorecard["total_events"]),
                    "eligible_regimes": int(scorecard["eligible_regimes"]),
                    "blocked_regimes": int(scorecard["blocked_regimes"]),
                    "worst_regime": str(scorecard["worst_regime"]),
                    "worst_adverse_post_edge_share": float(
                        scorecard["worst_adverse_post_edge_share"]
                    ),
                    "worst_mean_post_minus_pre_realized_edge": float(
                        scorecard["worst_mean_post_minus_pre_realized_edge"]
                    ),
                    "worst_post_minus_pre_realized_edge": float(
                        scorecard["worst_post_minus_pre_realized_edge"]
                    ),
                    "weighted_mean_event_fill_probability": float(
                        scorecard["weighted_mean_event_fill_probability"]
                    ),
                    "weighted_mean_event_adverse_fill_probability": float(
                        scorecard["weighted_mean_event_adverse_fill_probability"]
                    ),
                    "weighted_mean_post_minus_pre_realized_edge": float(
                        scorecard["weighted_mean_post_minus_pre_realized_edge"]
                    ),
                    "event_toxicity_label": label,
                    "sensitivity_label": sensitivity_label,
                }
            )
    return pd.DataFrame(rows)[columns]



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


def _empty_queue_position_order_size_sweep() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "order_size_fraction",
            "rows",
            "mean_bid_child_order_size",
            "mean_ask_child_order_size",
            "mean_bid_queue_clear_share",
            "mean_ask_queue_clear_share",
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


def _empty_queue_position_order_size_capacity_frontier() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "viable_rows": 0,
        "minimum_order_size_fraction": 0.0,
        "max_viable_order_size_fraction": 0.0,
        "minimum_size_mean_execution_adjusted_edge_ticks": 0.0,
        "max_viable_mean_execution_adjusted_edge_ticks": 0.0,
        "edge_decay_to_capacity_ticks": 0.0,
        "minimum_size_tradable_share": 0.0,
        "max_viable_tradable_share": 0.0,
        "tradable_share_decay_to_capacity": 0.0,
        "dominant_execution_side_at_capacity": "none",
        "order_size_capacity_label": "empty_sweep",
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


def _queue_order_size_capacity_label(*, max_size: float, edge_decay: float, tradable_decay: float) -> str:
    if max_size >= 0.50 and edge_decay <= 0.25 and tradable_decay <= 0.15:
        return "large_child_order_resilient_capacity"
    if max_size >= 0.20:
        return "child_order_capacity_constrained"
    return "small_child_order_only_capacity"


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


def _empty_queue_position_adverse_selection_policy_frontier() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fill_threshold",
            "adverse_threshold",
            "candidate_rows",
            "trade_share",
            "long_rows",
            "short_rows",
            "mean_predicted_fill_probability",
            "mean_adverse_fill_probability",
            "realized_fill_rate",
            "mean_realized_edge_ticks",
            "positive_edge_rate",
            "mean_execution_adjusted_edge_ticks",
            "toxicity_filtered_rows",
            "toxicity_filtered_share",
            "policy_label",
        ]
    )


def _validate_probability_thresholds(
    thresholds: list[float] | tuple[float, ...], argument_name: str, label: str
) -> None:
    if isinstance(thresholds, (str, bytes)):
        raise ValueError(f"{argument_name} must be a non-empty sequence of finite values")
    values = list(thresholds)
    if not values:
        raise ValueError(f"{argument_name} must be a non-empty sequence")
    for threshold in values:
        if not math.isfinite(float(threshold)):
            raise ValueError(f"{label} threshold values must be finite")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"{label} threshold values must be in [0.0, 1.0]")


def _queue_position_policy_frontier_label(
    *,
    trade_share: float,
    realized_fill_rate: float,
    mean_realized_edge: float,
    mean_adverse_fill_probability: float,
) -> str:
    if trade_share <= 0.0:
        return "no_executable_policy"
    if mean_realized_edge <= 0.0:
        return "execution_policy_rejected"
    if trade_share >= 0.30 and realized_fill_rate >= 0.70 and mean_adverse_fill_probability <= 0.25:
        return "balanced_execution_policy"
    if realized_fill_rate >= 0.70 and mean_adverse_fill_probability <= 0.30:
        return "selective_toxicity_control_policy"
    if realized_fill_rate >= 0.70:
        return "high_quality_capacity_constrained_policy"
    return "edge_positive_fill_uncertain_policy"


def _queue_position_policy_summary_label(
    *,
    trade_share: float,
    realized_fill_rate: float,
    mean_realized_edge: float,
    mean_adverse_fill_probability: float,
) -> str:
    if trade_share <= 0.0:
        return "no_publishable_toxicity_control_policy"
    if mean_realized_edge <= 0.0:
        return "rejected_toxicity_control_policy"
    if trade_share >= 0.30 and realized_fill_rate >= 0.70 and mean_adverse_fill_probability <= 0.25:
        return "broad_toxicity_control_policy"
    if realized_fill_rate >= 0.70 and mean_adverse_fill_probability <= 0.30:
        return "publishable_toxicity_control_policy"
    return "fragile_toxicity_control_policy"


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


def _queue_position_expected_value_policy_label(
    *, candidate_share: float, risk_adjusted_ev: float, mean_adverse_fill_probability: float
) -> str:
    if candidate_share <= 0.0:
        return "no_queue_ev_candidates"
    if risk_adjusted_ev <= 0.0:
        return "queue_policy_rejected"
    if mean_adverse_fill_probability >= 0.30:
        return "queue_policy_toxicity_review"
    if candidate_share >= 0.50:
        return "broad_positive_ev_queue_policy"
    return "selective_positive_ev_queue_policy"


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


def _empty_queue_position_fill_monotonicity_scorecard() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "best_execution_side",
            "queue_bins",
            "queue_steps",
            "rows",
            "predicted_fill_inversions",
            "realized_fill_inversions",
            "max_predicted_fill_inversion",
            "max_realized_fill_inversion",
            "monotonicity_label",
        ]
    )


def _empty_queue_position_calibration_stability(regime_col: str = "regime") -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            regime_col,
            "best_execution_side",
            "queue_share_bin",
            "fill_probability_bin",
            "research_rows",
            "heldout_rows",
            "research_realized_fill_rate",
            "heldout_realized_fill_rate",
            "realized_fill_rate_gap",
            "research_calibration_error",
            "heldout_calibration_error",
            "calibration_error_gap",
            "research_absolute_calibration_error",
            "heldout_absolute_calibration_error",
            "absolute_calibration_error_gap",
            "brier_score_gap",
            "research_mean_execution_adjusted_edge_ticks",
            "heldout_mean_execution_adjusted_edge_ticks",
            "execution_adjusted_edge_gap_ticks",
            "calibration_stability_label",
        ]
    )


def _queue_calibration_stability_label(
    *,
    research_missing: bool,
    heldout_missing: bool,
    fill_gap: float,
    abs_error_gap: float,
    edge_gap: float,
    max_error_gap: float,
    max_fill_rate_gap: float,
) -> str:
    if heldout_missing:
        return "calibration_cell_lost"
    if research_missing:
        return "calibration_cell_gained"
    if (
        abs_error_gap > max_error_gap
        or fill_gap < -max_fill_rate_gap
        or edge_gap < -max_error_gap
    ):
        return "calibration_degraded"
    return "calibration_replicated"


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


def _empty_passive_fill_brier_decomposition() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "bins": 0,
        "base_fill_rate": 0.0,
        "weighted_brier_score": 0.0,
        "uncertainty": 0.0,
        "reliability": 0.0,
        "resolution": 0.0,
        "brier_skill_score": 0.0,
        "brier_decomposition_error": 0.0,
        "calibration_quality_label": "empty",
    }


def _passive_fill_calibration_quality_label(
    *, reliability: float, resolution: float, brier_skill_score: float
) -> str:
    if brier_skill_score >= 0.20 and reliability <= 0.02 and resolution >= 0.05:
        return "resolved_calibrated_skill"
    if brier_skill_score >= 0.05 and resolution > reliability:
        return "resolved_but_needs_calibration"
    if reliability >= 0.05:
        return "miscalibrated_low_skill"
    if brier_skill_score < 0.0:
        return "worse_than_base_rate"
    return "low_resolution"


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



def queue_position_unfilled_opportunity_curve(
    frame: pd.DataFrame,
    *,
    lcri_bins: int = 5,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    lcri_col: str = "lcri",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    long_edge_col: str = "long_net_return_ticks",
    short_edge_col: str = "short_net_return_ticks",
    min_mean_unfilled_opportunity_ticks: float = 0.50,
    min_edge_capture_rate: float = 0.50,
) -> pd.DataFrame:
    """Quantify LCRI signal edge stranded by queue-position non-fills.

    Passive-alpha diagnostics can look publishable when the filled subset has edge,
    while the strongest LCRI rows are simply not fillable. This curve keeps the
    selected-side LCRI chronology in the denominator: rows are binned by absolute
    LCRI, then selected-side signal edge is split into realized/captured edge and
    unfilled opportunity. Optional grouping columns (for example event-window
    regimes or liquidity regimes) expose where the queue prevents LCRI tails from
    becoming executable rather than just measuring adverse selection after fills.
    """
    if not isinstance(lcri_bins, int) or isinstance(lcri_bins, bool):
        raise ValueError("lcri_bins must be an integer")
    if lcri_bins < 1:
        raise ValueError("lcri_bins must be at least 1")
    for name, value in {
        "min_mean_unfilled_opportunity_ticks": min_mean_unfilled_opportunity_ticks,
        "min_edge_capture_rate": min_edge_capture_rate,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if min_mean_unfilled_opportunity_ticks < 0.0:
        raise ValueError("min_mean_unfilled_opportunity_ticks must be non-negative")
    if not 0.0 <= min_edge_capture_rate <= 1.0:
        raise ValueError("min_edge_capture_rate must be in [0, 1]")

    grouping_columns = _normalize_group_columns(
        frame,
        group_cols,
        "queue position unfilled opportunity",
    )
    required = {
        side_col,
        lcri_col,
        bid_realized_col,
        ask_realized_col,
        "bid_fill_probability",
        "ask_fill_probability",
        long_edge_col,
        short_edge_col,
    }
    required.update(grouping_columns)
    _require_columns(frame, required, "queue position unfilled opportunity")

    columns = [
        *grouping_columns,
        "lcri_tail_bin",
        "rows",
        "mean_abs_lcri",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "mean_signal_edge_ticks",
        "mean_captured_edge_ticks",
        "mean_unfilled_opportunity_ticks",
        "edge_capture_rate",
        "unfilled_opportunity_share",
        "unfilled_opportunity_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    numeric_columns = [
        lcri_col,
        bid_realized_col,
        ask_realized_col,
        "bid_fill_probability",
        "ask_fill_probability",
        long_edge_col,
        short_edge_col,
    ]
    values = _finite_values(frame, numeric_columns, "queue position unfilled opportunity")
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return pd.DataFrame(columns=columns)

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    for column in grouping_columns:
        selected[column] = frame.loc[tradable, column].astype(str)
    selected["abs_lcri"] = values.loc[tradable, lcri_col].abs()
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
    selected["signal_edge_ticks"] = np.where(
        selected_side == "long",
        values.loc[tradable, long_edge_col],
        values.loc[tradable, short_edge_col],
    )
    probability_columns = ["predicted_fill_probability", "realized_fill"]
    for column in probability_columns:
        if not selected[column].between(0.0, 1.0).all():
            raise ValueError("queue position unfilled opportunity probabilities must be in [0, 1]")
    selected["captured_edge_ticks"] = selected["signal_edge_ticks"] * selected["realized_fill"]
    selected["unfilled_opportunity_ticks"] = selected["signal_edge_ticks"] * (1.0 - selected["realized_fill"])

    rows: list[dict[str, float | int | str]] = []
    groupers = grouping_columns.copy()
    if not groupers:
        selected["_all"] = "all"
        groupers = ["_all"]
    for keys, group in selected.groupby(groupers, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        grouped = group.copy()
        grouped["lcri_tail_bin"] = _rank_probability_bins(grouped["abs_lcri"], lcri_bins)
        for tail_bin, tail_group in grouped.groupby("lcri_tail_bin", sort=True):
            total_signal_edge = float(tail_group["signal_edge_ticks"].sum())
            total_captured_edge = float(tail_group["captured_edge_ticks"].sum())
            total_unfilled_opportunity = float(tail_group["unfilled_opportunity_ticks"].sum())
            edge_capture_rate = (
                total_captured_edge / total_signal_edge if abs(total_signal_edge) > 1e-12 else 0.0
            )
            unfilled_share = (
                total_unfilled_opportunity / total_signal_edge if abs(total_signal_edge) > 1e-12 else 0.0
            )
            mean_unfilled = float(tail_group["unfilled_opportunity_ticks"].mean())
            if (
                mean_unfilled >= min_mean_unfilled_opportunity_ticks - 1e-12
                and edge_capture_rate < min_edge_capture_rate - 1e-12
            ):
                label = "unfilled_tail_opportunity"
            elif mean_unfilled >= min_mean_unfilled_opportunity_ticks - 1e-12:
                label = "partial_opportunity_capture"
            else:
                label = "opportunity_captured"
            row: dict[str, float | int | str] = {}
            for column, key in zip(groupers, keys):
                if column != "_all":
                    row[column] = str(key)
            row.update(
                {
                    "lcri_tail_bin": int(tail_bin),
                    "rows": int(len(tail_group)),
                    "mean_abs_lcri": float(tail_group["abs_lcri"].mean()),
                    "mean_predicted_fill_probability": float(
                        tail_group["predicted_fill_probability"].mean()
                    ),
                    "realized_fill_rate": float(tail_group["realized_fill"].mean()),
                    "mean_signal_edge_ticks": float(tail_group["signal_edge_ticks"].mean()),
                    "mean_captured_edge_ticks": float(tail_group["captured_edge_ticks"].mean()),
                    "mean_unfilled_opportunity_ticks": mean_unfilled,
                    "edge_capture_rate": float(edge_capture_rate),
                    "unfilled_opportunity_share": float(unfilled_share),
                    "unfilled_opportunity_label": label,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def queue_position_unfilled_opportunity_scorecard(
    curve: pd.DataFrame,
    *,
    min_tail_bin: int | None = None,
    max_tail_unfilled_opportunity_share: float = 0.50,
    min_tail_edge_capture_rate: float = 0.50,
    min_tail_rows: int = 20,
) -> dict[str, bool | float | int | str]:
    """Gate execution-adjusted LCRI on high-signal opportunity that never fills.

    A passive policy can pass filled-trade toxicity checks while leaving the best
    LCRI tail edge stranded behind the queue. This scorecard reduces
    ``queue_position_unfilled_opportunity_curve`` into a release decision focused
    on the highest LCRI tail bins: it blocks when tail opportunity share is too
    large or edge capture is too low, and reviews when tail evidence is too thin.
    """
    if min_tail_bin is not None and (
        not isinstance(min_tail_bin, int) or isinstance(min_tail_bin, bool) or min_tail_bin < 1
    ):
        raise ValueError("min_tail_bin must be a positive integer when provided")
    if not isinstance(min_tail_rows, int) or isinstance(min_tail_rows, bool) or min_tail_rows < 1:
        raise ValueError("min_tail_rows must be a positive integer")
    for name, value in {
        "max_tail_unfilled_opportunity_share": max_tail_unfilled_opportunity_share,
        "min_tail_edge_capture_rate": min_tail_edge_capture_rate,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_tail_unfilled_opportunity_share <= 1.0:
        raise ValueError("max_tail_unfilled_opportunity_share must be in [0, 1]")
    if not 0.0 <= min_tail_edge_capture_rate <= 1.0:
        raise ValueError("min_tail_edge_capture_rate must be in [0, 1]")

    required = {
        "lcri_tail_bin",
        "rows",
        "mean_abs_lcri",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "mean_signal_edge_ticks",
        "mean_captured_edge_ticks",
        "mean_unfilled_opportunity_ticks",
        "edge_capture_rate",
        "unfilled_opportunity_share",
        "unfilled_opportunity_label",
    }
    missing = sorted(required - set(curve.columns))
    if missing:
        raise ValueError(f"missing queue position unfilled opportunity scorecard columns: {missing}")
    if curve.empty:
        return {
            "evaluated_cells": 0,
            "tail_cells": 0,
            "tail_rows": 0,
            "max_tail_unfilled_opportunity_share": 0.0,
            "min_tail_edge_capture_rate": 0.0,
            "weighted_tail_unfilled_opportunity_share": 0.0,
            "weighted_tail_edge_capture_rate": 0.0,
            "worst_tail_cell": "none",
            "unfilled_opportunity_release_label": "review",
            "publishable": False,
            "blocking_reasons": "none",
            "review_reasons": "empty_unfilled_opportunity_curve",
        }

    numeric_columns = list(required - {"unfilled_opportunity_label"})
    numeric = curve[numeric_columns].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("queue position unfilled opportunity scorecard metrics must be finite")
    if (numeric["rows"] < 0.0).any() or not np.isclose(
        numeric["rows"], np.round(numeric["rows"]), atol=1e-9
    ).all():
        raise ValueError("queue position unfilled opportunity scorecard rows must be non-negative integers")
    bounded_columns = [
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "edge_capture_rate",
        "unfilled_opportunity_share",
    ]
    if not numeric[bounded_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("queue position unfilled opportunity scorecard rates must be in [0, 1]")

    tail_start = int(min_tail_bin) if min_tail_bin is not None else int(numeric["lcri_tail_bin"].max())
    tail = curve.loc[numeric["lcri_tail_bin"] >= tail_start].copy()
    if tail.empty:
        tail = curve.loc[numeric["lcri_tail_bin"] == numeric["lcri_tail_bin"].max()].copy()
    tail_numeric = tail[numeric_columns].astype(float)
    tail_rows = int(tail_numeric["rows"].sum())
    weights = tail_numeric["rows"] / float(tail_rows) if tail_rows > 0 else pd.Series(0.0, index=tail.index)
    weighted_share = float((tail_numeric["unfilled_opportunity_share"] * weights).sum())
    weighted_capture = float((tail_numeric["edge_capture_rate"] * weights).sum())
    max_unfilled_share = float(tail_numeric["unfilled_opportunity_share"].max())
    min_capture = float(tail_numeric["edge_capture_rate"].min())

    risk = tail_numeric["unfilled_opportunity_share"] - tail_numeric["edge_capture_rate"]
    worst_index = risk.sort_values(ascending=False).index[0]
    group_columns = [
        column
        for column in curve.columns
        if column not in required and not column.startswith("mean_") and column not in {"rows"}
    ]
    if group_columns:
        prefix = ":".join(str(curve.loc[worst_index, column]) for column in group_columns)
        worst_tail_cell = f"{prefix}:tail_{int(curve.loc[worst_index, 'lcri_tail_bin'])}"
    else:
        worst_tail_cell = f"tail_{int(curve.loc[worst_index, 'lcri_tail_bin'])}"

    blocking_reasons: list[str] = []
    review_reasons: list[str] = []
    if max_unfilled_share > max_tail_unfilled_opportunity_share + 1e-12:
        blocking_reasons.append("tail_opportunity_share")
    if min_capture < min_tail_edge_capture_rate - 1e-12:
        blocking_reasons.append("tail_capture_shortfall")
    if tail_rows < min_tail_rows:
        review_reasons.append("thin_tail_opportunity_evidence")

    if blocking_reasons:
        label = "block"
    elif review_reasons:
        label = "review"
    else:
        label = "pass"

    return {
        "evaluated_cells": int(len(curve)),
        "tail_cells": int(len(tail)),
        "tail_rows": tail_rows,
        "max_tail_unfilled_opportunity_share": max_unfilled_share,
        "min_tail_edge_capture_rate": min_capture,
        "weighted_tail_unfilled_opportunity_share": weighted_share,
        "weighted_tail_edge_capture_rate": weighted_capture,
        "worst_tail_cell": worst_tail_cell,
        "unfilled_opportunity_release_label": label,
        "publishable": bool(label == "pass"),
        "blocking_reasons": ",".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ",".join(review_reasons) if review_reasons else "none",
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
    lcri_regime_attribution: pd.DataFrame | None = None,
    lcri_event_window_scorecard: dict[str, float | int | str] | None = None,
    latency_sensitivity: pd.DataFrame | None = None,
    fill_brier_decomposition: dict[str, float | int | str] | None = None,
    path_risk_scorecard: pd.DataFrame | None = None,
    max_conflict_share: float = 0.25,
    max_high_priority_conflict_share: float = 0.10,
    min_lcri_regime_survival_share: float = 0.50,
    max_lcri_regime_conflict_share: float = 0.40,
    max_latency_fill_decay: float = 0.10,
    min_latency_candidate_retention_share: float = 0.50,
    max_fragile_path_share: float = 0.25,
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
        "min_lcri_regime_survival_share": min_lcri_regime_survival_share,
        "max_lcri_regime_conflict_share": max_lcri_regime_conflict_share,
        "max_latency_fill_decay": max_latency_fill_decay,
        "min_latency_candidate_retention_share": min_latency_candidate_retention_share,
        "max_fragile_path_share": max_fragile_path_share,
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

    lcri_regime_label = "lcri_regime_execution_not_evaluated"
    weak_lcri_regime_sides = 0
    worst_lcri_regime = "none"
    worst_lcri_side = "none"
    min_lcri_survival = 0.0
    max_lcri_conflict = 0.0
    if lcri_regime_attribution is not None and not lcri_regime_attribution.empty:
        required_lcri_regime_columns = {
            "regime",
            "lcri_side",
            "rows",
            "execution_survival_share",
            "execution_conflict_share",
        }
        _require_columns(
            lcri_regime_attribution,
            required_lcri_regime_columns,
            "execution publishability LCRI regime survival",
        )
        lcri_values = _finite_values(
            lcri_regime_attribution,
            ["rows", "execution_survival_share", "execution_conflict_share"],
            "execution publishability LCRI regime survival",
        )
        if (lcri_values[["rows", "execution_survival_share", "execution_conflict_share"]] < 0.0).any().any():
            raise ValueError("execution publishability LCRI regime survival metrics must be non-negative")
        if (lcri_values[["execution_survival_share", "execution_conflict_share"]] > 1.0).any().any():
            raise ValueError("execution publishability LCRI regime shares must be in [0, 1]")
        lcri_sides = lcri_regime_attribution["lcri_side"].astype(str)
        lcri_signal = lcri_regime_attribution.loc[lcri_sides != "neutral"].copy()
        if not lcri_signal.empty:
            signal_values = lcri_values.loc[lcri_signal.index]
            weak_mask = (
                (signal_values["execution_survival_share"] < min_lcri_regime_survival_share)
                | (signal_values["execution_conflict_share"] > max_lcri_regime_conflict_share)
            )
            weak_lcri_regime_sides = int(weak_mask.sum())
            min_lcri_survival = float(signal_values["execution_survival_share"].min())
            max_lcri_conflict = float(signal_values["execution_conflict_share"].max())
            weakness_score = (
                (min_lcri_regime_survival_share - signal_values["execution_survival_share"])
                + (signal_values["execution_conflict_share"] - max_lcri_regime_conflict_share)
            )
            worst_index = weakness_score.idxmax()
            worst_lcri_regime = str(lcri_regime_attribution.loc[worst_index, "regime"])
            worst_lcri_side = str(lcri_regime_attribution.loc[worst_index, "lcri_side"])
            lcri_regime_label = (
                "lcri_regime_execution_not_preserved"
                if weak_lcri_regime_sides
                else "lcri_regime_execution_preserved"
            )

    lcri_event_window_label = "lcri_event_window_not_evaluated"
    lcri_event_window_decision = "pass"
    lcri_event_window_high_lcri_rows = 0
    lcri_event_window_toxic_high_lcri_row_share = 0.0
    lcri_event_window_event_toxic_high_lcri_row_share = 0.0
    lcri_event_window_signal_survival_ratio = 0.0
    lcri_event_window_fill_adverse_spread = 0.0
    worst_lcri_event_window_regime = "none"
    worst_lcri_event_window_bucket = "none"
    worst_lcri_event_window_label = "none"
    lcri_event_window_blocking_reasons = "none"
    lcri_event_window_review_reasons = "none"
    if lcri_event_window_scorecard is not None:
        required_event_window_keys = {
            "high_lcri_rows",
            "toxic_high_lcri_row_share",
            "event_toxic_high_lcri_row_share",
            "weighted_high_lcri_signal_survival_ratio",
            "weighted_high_lcri_fill_adverse_spread",
            "worst_event_window_regime",
            "worst_event_window_bucket",
            "worst_event_window_label",
            "release_decision",
            "release_label",
            "blocking_reasons",
            "review_reasons",
        }
        missing_event_window_keys = sorted(required_event_window_keys - set(lcri_event_window_scorecard))
        if missing_event_window_keys:
            raise ValueError(
                "missing execution publishability LCRI event-window scorecard keys: "
                f"{missing_event_window_keys}"
            )
        lcri_event_window_label = str(lcri_event_window_scorecard["release_label"])
        lcri_event_window_decision = str(lcri_event_window_scorecard["release_decision"])
        if lcri_event_window_decision not in {"pass", "review", "block"}:
            raise ValueError("execution publishability LCRI event-window decision is invalid")
        lcri_event_window_high_lcri_rows = int(lcri_event_window_scorecard["high_lcri_rows"])
        lcri_event_window_toxic_high_lcri_row_share = float(
            lcri_event_window_scorecard["toxic_high_lcri_row_share"]
        )
        lcri_event_window_event_toxic_high_lcri_row_share = float(
            lcri_event_window_scorecard["event_toxic_high_lcri_row_share"]
        )
        lcri_event_window_signal_survival_ratio = float(
            lcri_event_window_scorecard["weighted_high_lcri_signal_survival_ratio"]
        )
        lcri_event_window_fill_adverse_spread = float(
            lcri_event_window_scorecard["weighted_high_lcri_fill_adverse_spread"]
        )
        for metric_name, metric_value in {
            "lcri_event_window_high_lcri_rows": float(lcri_event_window_high_lcri_rows),
            "lcri_event_window_toxic_high_lcri_row_share": lcri_event_window_toxic_high_lcri_row_share,
            "lcri_event_window_event_toxic_high_lcri_row_share": lcri_event_window_event_toxic_high_lcri_row_share,
            "lcri_event_window_signal_survival_ratio": lcri_event_window_signal_survival_ratio,
            "lcri_event_window_fill_adverse_spread": lcri_event_window_fill_adverse_spread,
        }.items():
            if not math.isfinite(metric_value):
                raise ValueError(f"{metric_name} must be finite")
        if lcri_event_window_high_lcri_rows < 0:
            raise ValueError("execution publishability LCRI event-window counts must be non-negative")
        for metric_name, metric_value in {
            "lcri_event_window_toxic_high_lcri_row_share": lcri_event_window_toxic_high_lcri_row_share,
            "lcri_event_window_event_toxic_high_lcri_row_share": lcri_event_window_event_toxic_high_lcri_row_share,
            "lcri_event_window_signal_survival_ratio": lcri_event_window_signal_survival_ratio,
        }.items():
            if not 0.0 <= metric_value <= 1.0:
                raise ValueError(f"{metric_name} must be in [0, 1]")
        worst_lcri_event_window_regime = str(lcri_event_window_scorecard["worst_event_window_regime"])
        worst_lcri_event_window_bucket = str(lcri_event_window_scorecard["worst_event_window_bucket"])
        worst_lcri_event_window_label = str(lcri_event_window_scorecard["worst_event_window_label"])
        lcri_event_window_blocking_reasons = str(lcri_event_window_scorecard["blocking_reasons"])
        lcri_event_window_review_reasons = str(lcri_event_window_scorecard["review_reasons"])

    latency_label = "queue_latency_not_evaluated"
    worst_latency_steps = 0
    worst_latency_fill_gap = 0.0
    min_latency_retention = 0.0
    if latency_sensitivity is not None and not latency_sensitivity.empty:
        required_latency_columns = {
            "latency_steps",
            "candidates",
            "realized_fill_gap_vs_immediate",
        }
        _require_columns(
            latency_sensitivity,
            required_latency_columns,
            "execution publishability latency sensitivity",
        )
        latency_values = _finite_values(
            latency_sensitivity,
            ["latency_steps", "candidates", "realized_fill_gap_vs_immediate"],
            "execution publishability latency sensitivity",
        )
        if (latency_values[["latency_steps", "candidates"]] < 0.0).any().any():
            raise ValueError("execution publishability latency sensitivity counts must be non-negative")
        anchor_rows = latency_values[latency_values["latency_steps"] == 0.0]
        anchor_candidates = float(anchor_rows["candidates"].max()) if not anchor_rows.empty else 0.0
        delayed = latency_values[latency_values["latency_steps"] > 0.0]
        if delayed.empty or anchor_candidates <= 0.0:
            latency_label = "queue_latency_insufficient_evidence"
        else:
            worst_index = delayed["realized_fill_gap_vs_immediate"].idxmin()
            worst_latency_steps = int(latency_values.loc[worst_index, "latency_steps"])
            worst_latency_fill_gap = float(latency_values.loc[worst_index, "realized_fill_gap_vs_immediate"])
            min_latency_retention = float((delayed["candidates"] / anchor_candidates).min())
            latency_label = (
                "queue_latency_fragile"
                if worst_latency_fill_gap < -max_latency_fill_decay
                or min_latency_retention < min_latency_candidate_retention_share
                else "queue_latency_robust"
            )

    fill_calibration_label = "fill_calibration_not_evaluated"
    fill_brier_skill_score = 0.0
    fill_calibration_reliability = 0.0
    fill_calibration_resolution = 0.0
    if fill_brier_decomposition is not None:
        fill_calibration_label = str(
            fill_brier_decomposition.get("calibration_quality_label", "fill_calibration_missing_label")
        )
        fill_brier_skill_score = float(fill_brier_decomposition.get("brier_skill_score", 0.0))
        fill_calibration_reliability = float(fill_brier_decomposition.get("reliability", 0.0))
        fill_calibration_resolution = float(fill_brier_decomposition.get("resolution", 0.0))
        for metric_name, metric_value in {
            "fill_brier_skill_score": fill_brier_skill_score,
            "fill_calibration_reliability": fill_calibration_reliability,
            "fill_calibration_resolution": fill_calibration_resolution,
        }.items():
            if not math.isfinite(metric_value):
                raise ValueError(f"{metric_name} must be finite")

    path_risk_label = "execution_path_not_evaluated"
    fragile_path_share = 0.0
    fragile_path_count = 0
    worst_path_id = "none"
    worst_path_drawdown = 0.0
    worst_path_turnover_rate = 0.0
    worst_path_total_edge = 0.0
    if path_risk_scorecard is not None:
        if path_risk_scorecard.empty:
            path_risk_label = "execution_path_no_evidence"
        else:
            required_path_columns = {
                "path_id",
                "tradable_rows",
                "total_edge_ticks",
                "max_drawdown_ticks",
                "turnover_rate",
                "path_risk_label",
            }
            _require_columns(
                path_risk_scorecard,
                required_path_columns,
                "execution publishability path risk",
            )
            path_values = _finite_values(
                path_risk_scorecard,
                ["tradable_rows", "total_edge_ticks", "max_drawdown_ticks", "turnover_rate"],
                "execution publishability path risk",
            )
            if (path_values[["tradable_rows", "max_drawdown_ticks", "turnover_rate"]] < 0.0).any().any():
                raise ValueError("execution publishability path risk metrics must be non-negative")
            labels = path_risk_scorecard["path_risk_label"].astype(str)
            non_overall = path_risk_scorecard["path_id"].astype(str) != "overall"
            group_mask = non_overall if bool(non_overall.any()) else pd.Series(True, index=path_risk_scorecard.index)
            fragile_mask = labels == "execution_path_fragile"
            fragile_group_mask = group_mask & fragile_mask
            group_tradable_rows = float(path_values.loc[group_mask, "tradable_rows"].sum())
            fragile_tradable_rows = float(path_values.loc[fragile_group_mask, "tradable_rows"].sum())
            fragile_path_share = (
                float(fragile_tradable_rows / group_tradable_rows) if group_tradable_rows else 0.0
            )
            fragile_path_count = int(fragile_group_mask.sum())
            overall_rows = path_risk_scorecard[path_risk_scorecard["path_id"].astype(str) == "overall"]
            overall_fragile = (
                not overall_rows.empty
                and str(overall_rows.iloc[0]["path_risk_label"]) == "execution_path_fragile"
            )
            path_risk_label = (
                "execution_path_fragile"
                if overall_fragile or fragile_path_share > max_fragile_path_share
                else "execution_path_stable"
            )
            candidate_mask = fragile_group_mask if bool(fragile_group_mask.any()) else group_mask
            candidates = path_values.loc[candidate_mask].copy()
            fragility_score = (
                candidates["max_drawdown_ticks"]
                + candidates["turnover_rate"]
                - candidates["total_edge_ticks"]
            )
            worst_index = fragility_score.idxmax()
            worst_path_id = str(path_risk_scorecard.loc[worst_index, "path_id"])
            worst_path_drawdown = float(path_values.loc[worst_index, "max_drawdown_ticks"])
            worst_path_turnover_rate = float(path_values.loc[worst_index, "turnover_rate"])
            worst_path_total_edge = float(path_values.loc[worst_index, "total_edge_ticks"])

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
    if lcri_regime_label == "lcri_regime_execution_not_preserved":
        blocking_reasons.append(lcri_regime_label)
    if lcri_event_window_decision == "block":
        blocking_reasons.append(lcri_event_window_label)
        if lcri_event_window_blocking_reasons != "none":
            blocking_reasons.append(lcri_event_window_blocking_reasons)
    elif lcri_event_window_decision == "review":
        review_reasons.append(lcri_event_window_label)
        if lcri_event_window_review_reasons != "none":
            review_reasons.append(lcri_event_window_review_reasons)
    if latency_label == "queue_latency_fragile":
        blocking_reasons.append(latency_label)
    elif latency_label == "queue_latency_insufficient_evidence":
        review_reasons.append(latency_label)
    if fill_calibration_label in {"worse_than_base_rate", "miscalibrated_low_skill"}:
        blocking_reasons.append(fill_calibration_label)
    elif fill_calibration_label in {
        "empty",
        "fill_calibration_missing_label",
        "low_resolution",
        "resolved_but_needs_calibration",
    }:
        review_reasons.append(fill_calibration_label)
    if path_risk_label == "execution_path_fragile":
        blocking_reasons.append(path_risk_label)
    elif path_risk_label == "execution_path_no_evidence":
        review_reasons.append(path_risk_label)

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
        "lcri_regime_survival_label": lcri_regime_label,
        "weak_lcri_regime_sides": weak_lcri_regime_sides,
        "worst_lcri_regime": worst_lcri_regime,
        "worst_lcri_side": worst_lcri_side,
        "min_lcri_execution_survival_share": min_lcri_survival,
        "max_lcri_execution_conflict_share": max_lcri_conflict,
        "lcri_event_window_release_label": lcri_event_window_label,
        "lcri_event_window_high_lcri_rows": lcri_event_window_high_lcri_rows,
        "lcri_event_window_toxic_high_lcri_row_share": lcri_event_window_toxic_high_lcri_row_share,
        "lcri_event_window_event_toxic_high_lcri_row_share": lcri_event_window_event_toxic_high_lcri_row_share,
        "lcri_event_window_signal_survival_ratio": lcri_event_window_signal_survival_ratio,
        "lcri_event_window_fill_adverse_spread": lcri_event_window_fill_adverse_spread,
        "worst_lcri_event_window_regime": worst_lcri_event_window_regime,
        "worst_lcri_event_window_bucket": worst_lcri_event_window_bucket,
        "worst_lcri_event_window_label": worst_lcri_event_window_label,
        "latency_sensitivity_label": latency_label,
        "worst_latency_steps": worst_latency_steps,
        "worst_latency_fill_gap": worst_latency_fill_gap,
        "min_latency_candidate_retention_share": min_latency_retention,
        "fill_calibration_label": fill_calibration_label,
        "fill_brier_skill_score": fill_brier_skill_score,
        "fill_calibration_reliability": fill_calibration_reliability,
        "fill_calibration_resolution": fill_calibration_resolution,
        "path_risk_label": path_risk_label,
        "fragile_path_count": fragile_path_count,
        "fragile_path_share": fragile_path_share,
        "worst_path_id": worst_path_id,
        "worst_path_drawdown_ticks": worst_path_drawdown,
        "worst_path_turnover_rate": worst_path_turnover_rate,
        "worst_path_total_edge_ticks": worst_path_total_edge,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
        "decision": decision,
        "passes": passes,
        "release_gate_label": label,
    }


def queue_position_latency_sensitivity(
    frame: pd.DataFrame,
    *,
    latencies: list[int] | tuple[int, ...] = (0, 1, 2, 5),
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    edge_col: str = "execution_adjusted_edge_ticks",
    max_realized_fill_decay: float = 0.10,
) -> pd.DataFrame:
    """Audit how queue-aware passive fill evidence decays under decision latency.

    Passive execution claims are fragile when quote decisions are computed on stale
    queue state. This diagnostic keeps the decision-side and predicted fill odds at
    row ``t`` fixed, then scores realized fills after ``latency`` snapshot steps
    within each symbol/session group. The resulting curve distinguishes genuinely
    executable queue signals from effects that disappear when a child order joins a
    few book updates later.
    """
    columns = [
        "latency_steps",
        "candidates",
        "long_candidates",
        "short_candidates",
        "mean_decision_fill_probability",
        "realized_fill_rate",
        "realized_fill_gap_vs_immediate",
        "mean_execution_adjusted_edge_ticks",
        "latency_label",
    ]
    if not latencies:
        raise ValueError("latencies must be a non-empty sequence")
    if any(not isinstance(latency, int) or isinstance(latency, bool) or latency < 0 for latency in latencies):
        raise ValueError("latencies must be non-negative integers")
    if len(set(latencies)) != len(latencies):
        raise ValueError("latencies must be unique")
    if not math.isfinite(max_realized_fill_decay) or max_realized_fill_decay < 0.0:
        raise ValueError("max_realized_fill_decay must be finite and non-negative")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
        edge_col,
    }
    _require_columns(frame, required, "queue position latency sensitivity")
    grouping_columns = _normalize_group_columns(frame, group_cols, "queue position latency sensitivity group")
    if frame.empty:
        return pd.DataFrame(columns=columns)

    values = _finite_values(
        frame,
        ["bid_fill_probability", "ask_fill_probability", bid_realized_col, ask_realized_col, edge_col],
        "queue position latency sensitivity",
    )
    sides = frame[side_col].astype(str)
    tradable = sides.isin({"long", "short"})
    decision_fill_probability = pd.Series(
        np.select(
            [sides == "long", sides == "short"],
            [values["bid_fill_probability"], values["ask_fill_probability"]],
            default=np.nan,
        ),
        index=frame.index,
    )

    def latency_realized(column: str, latency: int) -> pd.Series:
        if latency == 0:
            return values[column]
        if not grouping_columns:
            return values[column].shift(-latency)
        keys = [frame[group_col] for group_col in grouping_columns]
        return values[column].groupby(keys, sort=False, dropna=False).shift(-latency)

    rows: list[dict[str, float | int | str]] = []
    anchor_fill_rate: float | None = None
    for latency in sorted(latencies):
        bid_realized = latency_realized(bid_realized_col, latency)
        ask_realized = latency_realized(ask_realized_col, latency)
        selected_realized = pd.Series(
            np.select(
                [sides == "long", sides == "short"],
                [bid_realized, ask_realized],
                default=np.nan,
            ),
            index=frame.index,
        )
        mask = tradable & selected_realized.notna()
        candidates = int(mask.sum())
        if candidates == 0:
            fill_rate = 0.0
            mean_probability = 0.0
            mean_edge = 0.0
        else:
            fill_rate = float(selected_realized[mask].mean())
            mean_probability = float(decision_fill_probability[mask].mean())
            mean_edge = float(values.loc[mask, edge_col].mean())
        if anchor_fill_rate is None:
            anchor_fill_rate = fill_rate
        gap = fill_rate - anchor_fill_rate
        if latency == 0:
            label = "anchor_latency"
        elif gap < -max_realized_fill_decay:
            label = "latency_fragile"
        else:
            label = "latency_robust"
        rows.append(
            {
                "latency_steps": int(latency),
                "candidates": candidates,
                "long_candidates": int(((sides == "long") & mask).sum()),
                "short_candidates": int(((sides == "short") & mask).sum()),
                "mean_decision_fill_probability": mean_probability,
                "realized_fill_rate": fill_rate,
                "realized_fill_gap_vs_immediate": float(gap),
                "mean_execution_adjusted_edge_ticks": mean_edge,
                "latency_label": label,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def queue_position_latency_edge_survival(
    frame: pd.DataFrame,
    *,
    latencies: list[int] | tuple[int, ...] = (0, 1, 2, 5),
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    edge_col: str = "execution_adjusted_edge_ticks",
    max_realized_edge_decay: float = 0.10,
) -> pd.DataFrame:
    """Price how much execution-adjusted edge survives stale queue decisions.

    Fill-rate latency curves can look acceptable while the lost fills are exactly
    the high-edge opportunities. This diagnostic keeps the decision side and edge
    at row ``t`` fixed, then applies realized selected-side fills from future
    snapshot latencies. The resulting realized-edge curve gives reviewers a direct
    tick-valued estimate of edge lost to queue-state staleness.
    """
    columns = [
        "latency_steps",
        "candidates",
        "long_candidates",
        "short_candidates",
        "realized_fill_rate",
        "mean_decision_edge_ticks",
        "realized_edge_ticks",
        "realized_edge_gap_vs_immediate",
        "edge_survival_ratio",
        "edge_latency_label",
    ]
    if not latencies:
        raise ValueError("latencies must be a non-empty sequence")
    if any(not isinstance(latency, int) or isinstance(latency, bool) or latency < 0 for latency in latencies):
        raise ValueError("latencies must be non-negative integers")
    if len(set(latencies)) != len(latencies):
        raise ValueError("latencies must be unique")
    if not math.isfinite(max_realized_edge_decay) or max_realized_edge_decay < 0.0:
        raise ValueError("max_realized_edge_decay must be finite and non-negative")

    required = {side_col, bid_realized_col, ask_realized_col, edge_col}
    _require_columns(frame, required, "queue position latency edge survival")
    grouping_columns = _normalize_group_columns(frame, group_cols, "queue position latency edge survival group")
    if frame.empty:
        return pd.DataFrame(columns=columns)

    values = _finite_values(
        frame,
        [bid_realized_col, ask_realized_col, edge_col],
        "queue position latency edge survival",
    )
    sides = frame[side_col].astype(str)
    tradable = sides.isin({"long", "short"})

    def latency_realized(column: str, latency: int) -> pd.Series:
        if latency == 0:
            return values[column]
        if not grouping_columns:
            return values[column].shift(-latency)
        keys = [frame[group_col] for group_col in grouping_columns]
        return values[column].groupby(keys, sort=False, dropna=False).shift(-latency)

    rows: list[dict[str, float | int | str]] = []
    anchor_realized_edge: float | None = None
    for latency in sorted(latencies):
        bid_realized = latency_realized(bid_realized_col, latency)
        ask_realized = latency_realized(ask_realized_col, latency)
        selected_realized = pd.Series(
            np.select(
                [sides == "long", sides == "short"],
                [bid_realized, ask_realized],
                default=np.nan,
            ),
            index=frame.index,
        )
        mask = tradable & selected_realized.notna()
        candidates = int(mask.sum())
        if candidates == 0:
            fill_rate = 0.0
            mean_edge = 0.0
            realized_edge = 0.0
        else:
            fill_rate = float(selected_realized[mask].mean())
            mean_edge = float(values.loc[mask, edge_col].mean())
            realized_edge = float((values.loc[mask, edge_col] * selected_realized[mask]).mean())
        if anchor_realized_edge is None:
            anchor_realized_edge = realized_edge
        edge_gap = realized_edge - anchor_realized_edge
        if latency == 0:
            label = "anchor_latency"
        elif edge_gap < -max_realized_edge_decay:
            label = "edge_latency_fragile"
        else:
            label = "edge_latency_robust"
        survival_ratio = 0.0 if anchor_realized_edge == 0.0 else realized_edge / anchor_realized_edge
        rows.append(
            {
                "latency_steps": int(latency),
                "candidates": candidates,
                "long_candidates": int(((sides == "long") & mask).sum()),
                "short_candidates": int(((sides == "short") & mask).sum()),
                "realized_fill_rate": fill_rate,
                "mean_decision_edge_ticks": mean_edge,
                "realized_edge_ticks": realized_edge,
                "realized_edge_gap_vs_immediate": float(edge_gap),
                "edge_survival_ratio": float(survival_ratio),
                "edge_latency_label": label,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def queue_position_latency_edge_survival_scorecard(
    survival: pd.DataFrame,
    *,
    max_fragile_edge_candidate_share: float = 0.20,
    review_fragile_edge_candidate_share: float = 0.10,
    min_candidate_weighted_edge_gap: float = -0.10,
    review_candidate_weighted_edge_gap: float = -0.05,
    min_weighted_edge_survival_ratio: float = 0.80,
) -> dict[str, float | int | str]:
    """Summarize whether tick-valued queue-latency edge survives release gates."""
    for name, value in {
        "max_fragile_edge_candidate_share": max_fragile_edge_candidate_share,
        "review_fragile_edge_candidate_share": review_fragile_edge_candidate_share,
        "min_candidate_weighted_edge_gap": min_candidate_weighted_edge_gap,
        "review_candidate_weighted_edge_gap": review_candidate_weighted_edge_gap,
        "min_weighted_edge_survival_ratio": min_weighted_edge_survival_ratio,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_fragile_edge_candidate_share <= 1.0:
        raise ValueError("max_fragile_edge_candidate_share must be in [0, 1]")
    if not 0.0 <= review_fragile_edge_candidate_share <= 1.0:
        raise ValueError("review_fragile_edge_candidate_share must be in [0, 1]")
    if not 0.0 <= min_weighted_edge_survival_ratio <= 1.0:
        raise ValueError("min_weighted_edge_survival_ratio must be in [0, 1]")

    if survival.empty:
        return {
            "latency_rows": 0,
            "anchor_candidates": 0,
            "latency_candidates": 0,
            "anchor_edge_ticks": 0.0,
            "fragile_edge_latency_rows": 0,
            "fragile_edge_candidate_share": 0.0,
            "candidate_weighted_edge_gap": 0.0,
            "candidate_weighted_edge_survival_ratio": 0.0,
            "worst_latency_steps": 0,
            "worst_edge_gap": 0.0,
            "edge_survival_release_decision": "review",
            "edge_survival_release_label": "queue_latency_edge_survival_no_evidence",
            "blocking_reasons": "none",
            "review_reasons": "no_latency_edge_evidence",
        }

    required = {
        "latency_steps",
        "candidates",
        "realized_edge_ticks",
        "realized_edge_gap_vs_immediate",
        "edge_survival_ratio",
        "edge_latency_label",
    }
    _require_columns(survival, required, "queue position latency edge survival scorecard")
    values = _finite_values(
        survival,
        [
            "latency_steps",
            "candidates",
            "realized_edge_ticks",
            "realized_edge_gap_vs_immediate",
            "edge_survival_ratio",
        ],
        "queue position latency edge survival scorecard",
    )
    if (values[["latency_steps", "candidates"]] < 0.0).any().any():
        raise ValueError("queue position latency edge survival scorecard counts must be non-negative")
    if not (values["latency_steps"] % 1.0).eq(0.0).all():
        raise ValueError("queue position latency edge survival scorecard latency steps must be integers")

    labels = survival["edge_latency_label"].astype(str)
    anchor_mask = values["latency_steps"].eq(0.0)
    latency_mask = ~anchor_mask
    anchor_candidates = int(values.loc[anchor_mask, "candidates"].sum())
    latency_candidates = int(values.loc[latency_mask, "candidates"].sum())
    anchor_edge = (
        0.0
        if not anchor_mask.any()
        else float(values.loc[anchor_mask, "realized_edge_ticks"].iloc[0])
    )

    fragile_mask = latency_mask & labels.eq("edge_latency_fragile")
    fragile_candidates = float(values.loc[fragile_mask, "candidates"].sum())
    fragile_share = 0.0 if latency_candidates == 0 else fragile_candidates / latency_candidates
    if latency_candidates == 0:
        weighted_gap = 0.0
        weighted_ratio = 0.0
    else:
        weights = values.loc[latency_mask, "candidates"]
        weighted_gap = float(
            np.average(values.loc[latency_mask, "realized_edge_gap_vs_immediate"], weights=weights)
        )
        weighted_ratio = float(np.average(values.loc[latency_mask, "edge_survival_ratio"], weights=weights))

    if latency_mask.any():
        worst_idx = values.loc[latency_mask, "realized_edge_gap_vs_immediate"].idxmin()
        worst_latency_steps = int(values.loc[worst_idx, "latency_steps"])
        worst_edge_gap = float(values.loc[worst_idx, "realized_edge_gap_vs_immediate"])
    else:
        worst_latency_steps = 0
        worst_edge_gap = 0.0

    blocking_reasons: list[str] = []
    review_reasons: list[str] = []
    if latency_candidates == 0:
        review_reasons.append("no_latency_edge_evidence")
    if fragile_share > max_fragile_edge_candidate_share:
        blocking_reasons.append("fragile_edge_candidate_share")
    elif fragile_share > review_fragile_edge_candidate_share:
        review_reasons.append("fragile_edge_candidate_share")
    if weighted_gap < min_candidate_weighted_edge_gap:
        blocking_reasons.append("candidate_weighted_edge_gap")
    elif weighted_gap < review_candidate_weighted_edge_gap:
        review_reasons.append("candidate_weighted_edge_gap")
    if weighted_ratio < min_weighted_edge_survival_ratio and latency_candidates > 0:
        blocking_reasons.append("weighted_edge_survival_ratio")

    if blocking_reasons:
        decision = "block"
        label = "queue_latency_edge_survival_blocked"
    elif review_reasons:
        decision = "review"
        label = "queue_latency_edge_survival_review"
    else:
        decision = "pass"
        label = "queue_latency_edge_survival_pass"

    return {
        "latency_rows": int(latency_mask.sum()),
        "anchor_candidates": anchor_candidates,
        "latency_candidates": latency_candidates,
        "anchor_edge_ticks": anchor_edge,
        "fragile_edge_latency_rows": int(fragile_mask.sum()),
        "fragile_edge_candidate_share": float(fragile_share),
        "candidate_weighted_edge_gap": weighted_gap,
        "candidate_weighted_edge_survival_ratio": weighted_ratio,
        "worst_latency_steps": worst_latency_steps,
        "worst_edge_gap": worst_edge_gap,
        "edge_survival_release_decision": decision,
        "edge_survival_release_label": label,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
    }


def queue_position_latency_regime_surface(
    frame: pd.DataFrame,
    *,
    regime_col: str = "passive_fill_event_window_regime",
    latencies: list[int] | tuple[int, ...] = (0, 1, 2, 5),
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    edge_col: str = "execution_adjusted_edge_ticks",
    max_realized_fill_decay: float = 0.10,
) -> pd.DataFrame:
    """Measure latency-driven passive-fill decay inside decision-time regimes.

    ``queue_position_latency_sensitivity`` gives one aggregate curve. This surface
    preserves the regime observed when the quote decision was made (for example a
    passive-fill event-window label), then replays selected-side realized fills at
    future snapshot latencies within each symbol/session group. The output is a
    compact reviewer artifact for spotting regimes where nominal execution alpha
    only works with zero-latency queue state.
    """
    columns = [
        regime_col,
        "latency_steps",
        "candidates",
        "long_candidates",
        "short_candidates",
        "mean_decision_fill_probability",
        "realized_fill_rate",
        "realized_fill_gap_vs_immediate",
        "mean_execution_adjusted_edge_ticks",
        "latency_regime_label",
    ]
    if not latencies:
        raise ValueError("latencies must be a non-empty sequence")
    if any(not isinstance(latency, int) or isinstance(latency, bool) or latency < 0 for latency in latencies):
        raise ValueError("latencies must be non-negative integers")
    if len(set(latencies)) != len(latencies):
        raise ValueError("latencies must be unique")
    if not math.isfinite(max_realized_fill_decay) or max_realized_fill_decay < 0.0:
        raise ValueError("max_realized_fill_decay must be finite and non-negative")

    required = {
        regime_col,
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
        edge_col,
    }
    _require_columns(frame, required, "queue position latency regime surface")
    grouping_columns = _normalize_group_columns(frame, group_cols, "queue position latency regime surface group")
    if frame.empty:
        return pd.DataFrame(columns=columns)

    values = _finite_values(
        frame,
        ["bid_fill_probability", "ask_fill_probability", bid_realized_col, ask_realized_col, edge_col],
        "queue position latency regime surface",
    )
    sides = frame[side_col].astype(str)
    regimes = frame[regime_col].astype(str)
    tradable = sides.isin({"long", "short"})
    decision_fill_probability = pd.Series(
        np.select(
            [sides == "long", sides == "short"],
            [values["bid_fill_probability"], values["ask_fill_probability"]],
            default=np.nan,
        ),
        index=frame.index,
    )

    def latency_realized(column: str, latency: int) -> pd.Series:
        if latency == 0:
            return values[column]
        if not grouping_columns:
            return values[column].shift(-latency)
        keys = [frame[group_col] for group_col in grouping_columns]
        return values[column].groupby(keys, sort=False, dropna=False).shift(-latency)

    rows: list[dict[str, float | int | str]] = []
    for regime in sorted(regimes.unique()):
        regime_mask = regimes == regime
        anchor_fill_rate: float | None = None
        for latency in sorted(latencies):
            bid_realized = latency_realized(bid_realized_col, latency)
            ask_realized = latency_realized(ask_realized_col, latency)
            selected_realized = pd.Series(
                np.select(
                    [sides == "long", sides == "short"],
                    [bid_realized, ask_realized],
                    default=np.nan,
                ),
                index=frame.index,
            )
            mask = regime_mask & tradable & selected_realized.notna()
            candidates = int(mask.sum())
            if candidates == 0:
                fill_rate = 0.0
                mean_probability = 0.0
                mean_edge = 0.0
            else:
                fill_rate = float(selected_realized[mask].mean())
                mean_probability = float(decision_fill_probability[mask].mean())
                mean_edge = float(values.loc[mask, edge_col].mean())
            if anchor_fill_rate is None:
                anchor_fill_rate = fill_rate
            gap = fill_rate - anchor_fill_rate
            if latency == 0:
                label = "anchor_latency"
            elif gap < -max_realized_fill_decay:
                label = "latency_fragile"
            else:
                label = "latency_robust"
            rows.append(
                {
                    regime_col: str(regime),
                    "latency_steps": int(latency),
                    "candidates": candidates,
                    "long_candidates": int(((sides == "long") & mask).sum()),
                    "short_candidates": int(((sides == "short") & mask).sum()),
                    "mean_decision_fill_probability": mean_probability,
                    "realized_fill_rate": fill_rate,
                    "realized_fill_gap_vs_immediate": float(gap),
                    "mean_execution_adjusted_edge_ticks": mean_edge,
                    "latency_regime_label": label,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def queue_position_latency_edge_regime_surface(
    frame: pd.DataFrame,
    *,
    regime_col: str = "passive_fill_event_window_regime",
    latencies: list[int] | tuple[int, ...] = (0, 1, 2, 5),
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    edge_col: str = "execution_adjusted_edge_ticks",
    max_realized_edge_decay: float = 0.10,
) -> pd.DataFrame:
    """Measure tick-valued latency edge survival inside decision-time regimes."""
    columns = [
        regime_col,
        "latency_steps",
        "candidates",
        "long_candidates",
        "short_candidates",
        "realized_fill_rate",
        "mean_decision_edge_ticks",
        "realized_edge_ticks",
        "realized_edge_gap_vs_immediate",
        "edge_survival_ratio",
        "edge_latency_regime_label",
    ]
    if not latencies:
        raise ValueError("latencies must be a non-empty sequence")
    if any(not isinstance(latency, int) or isinstance(latency, bool) or latency < 0 for latency in latencies):
        raise ValueError("latencies must be non-negative integers")
    if len(set(latencies)) != len(latencies):
        raise ValueError("latencies must be unique")
    if not math.isfinite(max_realized_edge_decay) or max_realized_edge_decay < 0.0:
        raise ValueError("max_realized_edge_decay must be finite and non-negative")

    required = {regime_col, side_col, bid_realized_col, ask_realized_col, edge_col}
    _require_columns(frame, required, "queue position latency edge regime surface")
    grouping_columns = _normalize_group_columns(frame, group_cols, "queue position latency edge regime surface group")
    if frame.empty:
        return pd.DataFrame(columns=columns)

    values = _finite_values(
        frame,
        [bid_realized_col, ask_realized_col, edge_col],
        "queue position latency edge regime surface",
    )
    sides = frame[side_col].astype(str)
    regimes = frame[regime_col].astype(str)
    tradable = sides.isin({"long", "short"})

    def latency_realized(column: str, latency: int) -> pd.Series:
        if latency == 0:
            return values[column]
        if not grouping_columns:
            return values[column].shift(-latency)
        keys = [frame[group_col] for group_col in grouping_columns]
        return values[column].groupby(keys, sort=False, dropna=False).shift(-latency)

    rows: list[dict[str, float | int | str]] = []
    for regime in sorted(regimes.unique()):
        regime_mask = regimes == regime
        anchor_realized_edge: float | None = None
        for latency in sorted(latencies):
            bid_realized = latency_realized(bid_realized_col, latency)
            ask_realized = latency_realized(ask_realized_col, latency)
            selected_realized = pd.Series(
                np.select(
                    [sides == "long", sides == "short"],
                    [bid_realized, ask_realized],
                    default=np.nan,
                ),
                index=frame.index,
            )
            mask = regime_mask & tradable & selected_realized.notna()
            candidates = int(mask.sum())
            if candidates == 0:
                fill_rate = 0.0
                mean_edge = 0.0
                realized_edge = 0.0
            else:
                fill_rate = float(selected_realized[mask].mean())
                mean_edge = float(values.loc[mask, edge_col].mean())
                realized_edge = float((values.loc[mask, edge_col] * selected_realized[mask]).mean())
            if anchor_realized_edge is None:
                anchor_realized_edge = realized_edge
            edge_gap = realized_edge - anchor_realized_edge
            if latency == 0:
                label = "anchor_latency"
            elif edge_gap < -max_realized_edge_decay:
                label = "edge_latency_regime_fragile"
            else:
                label = "edge_latency_regime_robust"
            survival_ratio = 0.0 if anchor_realized_edge == 0.0 else realized_edge / anchor_realized_edge
            rows.append(
                {
                    regime_col: str(regime),
                    "latency_steps": int(latency),
                    "candidates": candidates,
                    "long_candidates": int(((sides == "long") & mask).sum()),
                    "short_candidates": int(((sides == "short") & mask).sum()),
                    "realized_fill_rate": fill_rate,
                    "mean_decision_edge_ticks": mean_edge,
                    "realized_edge_ticks": realized_edge,
                    "realized_edge_gap_vs_immediate": float(edge_gap),
                    "edge_survival_ratio": float(survival_ratio),
                    "edge_latency_regime_label": label,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def queue_position_latency_release_scorecard(
    surface: pd.DataFrame,
    *,
    regime_col: str = "passive_fill_event_window_regime",
    max_fragile_candidate_share: float = 0.20,
    review_fragile_candidate_share: float = 0.10,
    min_weighted_fill_gap: float = -0.10,
    review_weighted_fill_gap: float = -0.05,
    min_candidate_retention_share: float = 0.70,
) -> dict[str, float | int | str]:
    """Summarize whether queue-position evidence survives latency for release.

    The latency regime surface is useful for reviewers, but release gates need a
    compact decision artifact. This scorecard weights non-zero latency rows by
    candidate count, flags regimes where realized fills decay after queue-state
    staleness, and exposes the worst regime/latency pair so demo artifacts do not
    overstate passive execution quality from zero-latency labels.
    """
    for name, value in {
        "max_fragile_candidate_share": max_fragile_candidate_share,
        "review_fragile_candidate_share": review_fragile_candidate_share,
        "min_weighted_fill_gap": min_weighted_fill_gap,
        "review_weighted_fill_gap": review_weighted_fill_gap,
        "min_candidate_retention_share": min_candidate_retention_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_fragile_candidate_share <= 1.0:
        raise ValueError("max_fragile_candidate_share must be in [0, 1]")
    if not 0.0 <= review_fragile_candidate_share <= 1.0:
        raise ValueError("review_fragile_candidate_share must be in [0, 1]")
    if not 0.0 <= min_candidate_retention_share <= 1.0:
        raise ValueError("min_candidate_retention_share must be in [0, 1]")

    required = {regime_col, "latency_steps", "candidates", "realized_fill_gap_vs_immediate", "latency_regime_label"}
    _require_columns(surface, required, "queue position latency release scorecard")
    if surface.empty:
        return {
            "regimes": 0,
            "latency_rows": 0,
            "anchor_candidates": 0,
            "latency_candidates": 0,
            "candidate_retention_share": 0.0,
            "fragile_latency_rows": 0,
            "fragile_candidate_share": 0.0,
            "candidate_weighted_fill_gap": 0.0,
            "worst_regime": "none",
            "worst_latency_steps": 0,
            "worst_fill_gap": 0.0,
            "latency_release_decision": "review",
            "latency_release_label": "queue_latency_release_no_evidence",
            "blocking_reasons": "none",
            "review_reasons": "no_latency_evidence",
        }

    values = _finite_values(
        surface,
        ["latency_steps", "candidates", "realized_fill_gap_vs_immediate"],
        "queue position latency release scorecard",
    )
    if (values[["latency_steps", "candidates"]] < 0.0).any().any():
        raise ValueError("queue position latency release scorecard counts must be non-negative")
    if not (values["latency_steps"] % 1.0).eq(0.0).all():
        raise ValueError("queue position latency release scorecard latency steps must be integers")

    regimes = surface[regime_col].astype(str)
    labels = surface["latency_regime_label"].astype(str)
    anchor_mask = values["latency_steps"].eq(0.0)
    latency_mask = ~anchor_mask
    anchor_candidates = int(values.loc[anchor_mask, "candidates"].sum())
    latency_candidates = int(values.loc[latency_mask, "candidates"].sum())
    candidate_retention_share = 0.0 if anchor_candidates == 0 else latency_candidates / anchor_candidates

    fragile_mask = latency_mask & labels.eq("latency_fragile")
    fragile_candidates = float(values.loc[fragile_mask, "candidates"].sum())
    fragile_candidate_share = 0.0 if latency_candidates == 0 else fragile_candidates / latency_candidates
    if latency_candidates == 0:
        weighted_gap = 0.0
    else:
        weighted_gap = float(
            np.average(
                values.loc[latency_mask, "realized_fill_gap_vs_immediate"],
                weights=values.loc[latency_mask, "candidates"],
            )
        )

    if latency_mask.any():
        worst_idx = values.loc[latency_mask, "realized_fill_gap_vs_immediate"].idxmin()
        worst_regime = str(regimes.loc[worst_idx])
        worst_latency_steps = int(values.loc[worst_idx, "latency_steps"])
        worst_fill_gap = float(values.loc[worst_idx, "realized_fill_gap_vs_immediate"])
    else:
        worst_regime = "none"
        worst_latency_steps = 0
        worst_fill_gap = 0.0

    blocking_reasons: list[str] = []
    review_reasons: list[str] = []
    if latency_candidates == 0:
        review_reasons.append("no_latency_evidence")
    if fragile_candidate_share > max_fragile_candidate_share:
        blocking_reasons.append("fragile_candidate_share")
    elif fragile_candidate_share > review_fragile_candidate_share:
        review_reasons.append("fragile_candidate_share")
    if weighted_gap < min_weighted_fill_gap:
        blocking_reasons.append("weighted_fill_gap")
    elif weighted_gap < review_weighted_fill_gap:
        review_reasons.append("weighted_fill_gap")
    if candidate_retention_share < min_candidate_retention_share:
        review_reasons.append("candidate_retention")

    if blocking_reasons:
        decision = "block"
        label = "queue_latency_release_blocked"
    elif review_reasons:
        decision = "review"
        label = "queue_latency_release_review"
    else:
        decision = "pass"
        label = "queue_latency_release_pass"

    return {
        "regimes": int(regimes.nunique()),
        "latency_rows": int(latency_mask.sum()),
        "anchor_candidates": anchor_candidates,
        "latency_candidates": latency_candidates,
        "candidate_retention_share": float(candidate_retention_share),
        "fragile_latency_rows": int(fragile_mask.sum()),
        "fragile_candidate_share": float(fragile_candidate_share),
        "candidate_weighted_fill_gap": weighted_gap,
        "worst_regime": worst_regime,
        "worst_latency_steps": worst_latency_steps,
        "worst_fill_gap": worst_fill_gap,
        "latency_release_decision": decision,
        "latency_release_label": label,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
    }


def queue_position_path_drawdown_episodes(
    frame: pd.DataFrame,
    *,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    edge_col: str = "execution_adjusted_edge_ticks",
    event_window_col: str | None = None,
) -> pd.DataFrame:
    """Extract path-level drawdown episodes for queue-position execution policies.

    Aggregate path-risk scorecards can show that a passive execution policy has an
    unacceptable drawdown, but not where the underwater run starts, whether it is
    still open, which side churn occurred inside it, or which passive-fill event
    window dominates the damage. This helper replays executed non-abstain rows in
    chronological order and emits one row per contiguous underwater episode so a
    publishability review can inspect concrete execution failure windows rather
    than only summary drawdown scalars.
    """
    grouping_columns = _normalize_group_columns(frame, group_cols, "queue position path drawdown group")
    required = {side_col, edge_col, *grouping_columns}
    if event_window_col is not None:
        required.add(event_window_col)
    columns = [
        "path_id",
        "episode_id",
        "episode_start_row",
        "episode_end_row",
        "trough_row",
        "episode_rows",
        "max_drawdown_ticks",
        "recovery_edge_ticks",
        "episode_total_edge_ticks",
        "episode_turnover_events",
        "dominant_event_window_regime",
        "episode_risk_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(frame, required, "queue position path drawdown episodes")
    values = _finite_values(frame, [edge_col], "queue position path drawdown episodes")
    sides = frame[side_col].astype(str)
    row_positions = pd.Series(range(len(frame)), index=frame.index)

    def dominant_event(path: pd.DataFrame) -> str:
        if event_window_col is None:
            return "none"
        events = path[event_window_col].astype(str)
        if events.empty:
            return "none"
        counts: dict[str, int] = {}
        for event in events:
            counts[event] = counts.get(event, 0) + 1
        return max(counts, key=lambda event: counts[event])

    def path_episode_rows(path: pd.DataFrame, path_id: str) -> list[dict[str, float | int | str]]:
        executed = path.loc[sides.loc[path.index] != "abstain"]
        if executed.empty:
            return []
        executed_edges = values.loc[executed.index, edge_col]
        executed_sides = sides.loc[executed.index]
        cumulative = executed_edges.cumsum()
        running_peak = cumulative.cummax().clip(lower=0.0)
        drawdown = running_peak - cumulative

        rows: list[dict[str, float | int | str]] = []
        in_episode = False
        start_pos = 0
        episode_id = 0
        for pos, drawdown_value in enumerate(drawdown.to_numpy()):
            if drawdown_value > 0.0 and not in_episode:
                in_episode = True
                start_pos = pos
            is_last = pos == len(drawdown) - 1
            if in_episode and (drawdown_value <= 0.0 or is_last):
                end_pos = pos
                episode = executed.iloc[start_pos : end_pos + 1]
                episode_drawdown = drawdown.iloc[start_pos : end_pos + 1]
                episode_edges = executed_edges.loc[episode.index]
                episode_sides = executed_sides.loc[episode.index]
                trough_label = episode_drawdown.idxmax()
                trough_position = episode.index.get_loc(trough_label)
                recovery_edge = float(episode_edges.iloc[trough_position + 1 :].sum())
                side_changes = episode_sides.ne(episode_sides.shift()).iloc[1:]
                rows.append(
                    {
                        "path_id": path_id,
                        "episode_id": episode_id,
                        "episode_start_row": int(row_positions.loc[episode.index[0]]),
                        "episode_end_row": int(row_positions.loc[episode.index[-1]]),
                        "trough_row": int(row_positions.loc[trough_label]),
                        "episode_rows": int(len(episode)),
                        "max_drawdown_ticks": float(episode_drawdown.max()),
                        "recovery_edge_ticks": recovery_edge,
                        "episode_total_edge_ticks": float(episode_edges.sum()),
                        "episode_turnover_events": int(side_changes.sum()),
                        "dominant_event_window_regime": dominant_event(episode),
                        "episode_risk_label": "path_drawdown_open"
                        if is_last and drawdown_value > 0.0
                        else "path_drawdown_recovered",
                    }
                )
                episode_id += 1
                in_episode = False
        return rows

    rows: list[dict[str, float | int | str]] = []
    if grouping_columns:
        groupby_arg: str | list[str] = grouping_columns[0] if len(grouping_columns) == 1 else grouping_columns
        for key, group in frame.groupby(groupby_arg, sort=True, dropna=False):
            if isinstance(key, tuple):
                path_id = "|".join(str(part) for part in key)
            else:
                path_id = str(key)
            rows.extend(path_episode_rows(group, path_id))
    else:
        rows.extend(path_episode_rows(frame, "overall"))

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["max_drawdown_ticks", "path_id", "episode_id"], ascending=[False, True, True], ignore_index=True
    )


def queue_position_path_drawdown_summary(
    episodes: pd.DataFrame,
    *,
    severe_drawdown_ticks: float = 1.0,
    max_drawdown_ticks: float = 3.0,
    max_open_episode_share: float = 0.25,
    max_severe_episode_share: float = 0.25,
    max_top_regime_drawdown_share: float = 0.70,
) -> dict[str, float | int | str]:
    """Summarize drawdown episodes into a release-facing queue-risk artifact.

    Episode tables are useful for forensic review, but demos and CI need a compact
    answer to whether queue-position-aware execution risk is clustered, still
    unrecovered, or dominated by a specific passive-fill event-window regime. This
    summary preserves those path-dependent failure modes instead of letting them
    disappear into mean execution-adjusted edge.
    """
    for name, value in {
        "severe_drawdown_ticks": severe_drawdown_ticks,
        "max_drawdown_ticks": max_drawdown_ticks,
        "max_open_episode_share": max_open_episode_share,
        "max_severe_episode_share": max_severe_episode_share,
        "max_top_regime_drawdown_share": max_top_regime_drawdown_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if severe_drawdown_ticks < 0.0:
        raise ValueError("severe_drawdown_ticks must be non-negative")
    if max_drawdown_ticks < 0.0:
        raise ValueError("max_drawdown_ticks must be non-negative")
    if not 0.0 <= max_open_episode_share <= 1.0:
        raise ValueError("max_open_episode_share must be in [0, 1]")
    if not 0.0 <= max_severe_episode_share <= 1.0:
        raise ValueError("max_severe_episode_share must be in [0, 1]")
    if not 0.0 <= max_top_regime_drawdown_share <= 1.0:
        raise ValueError("max_top_regime_drawdown_share must be in [0, 1]")

    required = {
        "path_id",
        "max_drawdown_ticks",
        "recovery_edge_ticks",
        "dominant_event_window_regime",
        "episode_risk_label",
    }
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"missing queue position path drawdown summary columns: {missing}")
    if episodes.empty:
        return {
            "episodes": 0,
            "paths_with_drawdown": 0,
            "open_episodes": 0,
            "open_episode_share": 0.0,
            "severe_episodes": 0,
            "severe_episode_share": 0.0,
            "mean_drawdown_ticks": 0.0,
            "max_drawdown_ticks": 0.0,
            "total_drawdown_ticks": 0.0,
            "total_recovery_edge_ticks": 0.0,
            "recovery_coverage_ratio": 0.0,
            "dominant_drawdown_regime": "none",
            "dominant_regime_drawdown_share": 0.0,
            "top_path_id": "none",
            "top_path_drawdown_share": 0.0,
            "drawdown_summary_label": "queue_drawdown_pass",
            "blocking_reasons": "none",
            "review_reasons": "none",
        }

    numeric = _finite_values(
        episodes,
        ["max_drawdown_ticks", "recovery_edge_ticks"],
        "queue position path drawdown summary",
    )
    if (numeric < 0.0).any().any():
        raise ValueError("queue position path drawdown summary values must be non-negative")

    drawdown = numeric["max_drawdown_ticks"]
    recovery = numeric["recovery_edge_ticks"]
    total_drawdown = float(drawdown.sum())
    episode_count = int(len(episodes))
    labels = episodes["episode_risk_label"].astype(str)
    open_episodes = int((labels == "path_drawdown_open").sum())
    open_share = float(open_episodes / episode_count)
    severe_episodes = int((drawdown >= severe_drawdown_ticks).sum())
    severe_share = float(severe_episodes / episode_count)
    total_recovery = float(recovery.sum())
    recovery_ratio = float(total_recovery / total_drawdown) if total_drawdown > 0.0 else 0.0

    paths = episodes["path_id"].astype(str)
    paths_with_drawdown = int(paths.nunique())
    path_drawdown = drawdown.groupby(paths).sum()
    if total_drawdown > 0.0 and not path_drawdown.empty:
        top_path_id = str(path_drawdown.idxmax())
        top_path_share = float(path_drawdown.max() / total_drawdown)
    else:
        top_path_id = "none"
        top_path_share = 0.0

    regimes = episodes["dominant_event_window_regime"].astype(str)
    regime_drawdown = drawdown.groupby(regimes).sum()
    if total_drawdown > 0.0 and not regime_drawdown.empty:
        dominant_regime = str(regime_drawdown.idxmax())
        dominant_regime_share = float(regime_drawdown.max() / total_drawdown)
    else:
        dominant_regime = "none"
        dominant_regime_share = 0.0

    blocking_reasons: list[str] = []
    if float(drawdown.max()) > max_drawdown_ticks:
        blocking_reasons.append("max_drawdown")
    if dominant_regime_share > max_top_regime_drawdown_share:
        blocking_reasons.append("regime_drawdown_concentration")

    review_reasons: list[str] = []
    if open_share > max_open_episode_share:
        review_reasons.append("open_drawdown_share")
    if severe_share > max_severe_episode_share:
        review_reasons.append("severe_drawdown_share")

    if blocking_reasons:
        label = "queue_drawdown_blocked"
    elif review_reasons:
        label = "queue_drawdown_review"
    else:
        label = "queue_drawdown_pass"

    return {
        "episodes": episode_count,
        "paths_with_drawdown": paths_with_drawdown,
        "open_episodes": open_episodes,
        "open_episode_share": open_share,
        "severe_episodes": severe_episodes,
        "severe_episode_share": severe_share,
        "mean_drawdown_ticks": float(drawdown.mean()),
        "max_drawdown_ticks": float(drawdown.max()),
        "total_drawdown_ticks": total_drawdown,
        "total_recovery_edge_ticks": total_recovery,
        "recovery_coverage_ratio": recovery_ratio,
        "dominant_drawdown_regime": dominant_regime,
        "dominant_regime_drawdown_share": dominant_regime_share,
        "top_path_id": top_path_id,
        "top_path_drawdown_share": top_path_share,
        "drawdown_summary_label": label,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
    }


def queue_position_path_tail_loss_scorecard(
    frame: pd.DataFrame,
    *,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    edge_col: str = "execution_adjusted_edge_ticks",
    tail_probability: float = 0.95,
    severe_loss_ticks: float = 1.0,
    max_tail_loss_ticks: float = 1.0,
    max_severe_loss_share: float = 0.25,
    max_loss_run_length: int = 2,
) -> pd.DataFrame:
    """Score path-level left-tail loss for queue-position-aware execution.

    Mean execution-adjusted edge can hide concentrated downside: a passive policy
    may look publishable while losing most of its edge in clustered adverse fills.
    This diagnostic keeps only non-abstain rows, converts negative edge into loss
    ticks, and reports VaR/CVaR-style tail loss, severe-loss share, and the longest
    consecutive loss run for each optional path group plus an overall row.
    """
    for name, value in {
        "tail_probability": tail_probability,
        "severe_loss_ticks": severe_loss_ticks,
        "max_tail_loss_ticks": max_tail_loss_ticks,
        "max_severe_loss_share": max_severe_loss_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= tail_probability <= 1.0:
        raise ValueError("tail_probability must be in [0, 1]")
    if severe_loss_ticks < 0.0:
        raise ValueError("severe_loss_ticks must be non-negative")
    if max_tail_loss_ticks < 0.0:
        raise ValueError("max_tail_loss_ticks must be non-negative")
    if not 0.0 <= max_severe_loss_share <= 1.0:
        raise ValueError("max_severe_loss_share must be in [0, 1]")
    if not isinstance(max_loss_run_length, int) or isinstance(max_loss_run_length, bool):
        raise ValueError("max_loss_run_length must be an integer")
    if max_loss_run_length < 0:
        raise ValueError("max_loss_run_length must be non-negative")

    grouping_columns = _normalize_group_columns(frame, group_cols, "queue position path tail loss group")
    required = {side_col, edge_col, *grouping_columns}
    columns = [
        "path_id",
        "rows",
        "tradable_rows",
        "loss_rows",
        "mean_loss_ticks",
        "tail_loss_threshold_ticks",
        "conditional_tail_loss_ticks",
        "severe_loss_share",
        "max_loss_run_length",
        "tail_loss_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(frame, required, "queue position path tail loss")
    values = _finite_values(frame, [edge_col], "queue position path tail loss")
    sides = frame[side_col].astype(str)

    def max_consecutive_losses(edges: pd.Series) -> int:
        longest = 0
        current = 0
        for is_loss in (edges < 0.0).tolist():
            if is_loss:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    def score_path(path: pd.DataFrame, path_sides: pd.Series, path_id: str) -> dict[str, float | int | str]:
        path_edges = values.loc[path.index, edge_col]
        tradable = path_sides != "abstain"
        executed_edges = path_edges.loc[tradable]
        tradable_rows = int(tradable.sum())
        if tradable_rows == 0:
            loss_rows = 0
            mean_loss = 0.0
            tail_threshold = 0.0
            conditional_tail_loss = 0.0
            severe_loss_share = 0.0
            loss_run = 0
        else:
            losses = (-executed_edges).clip(lower=0.0)
            loss_rows = int((losses > 0.0).sum())
            mean_loss = float(losses.mean())
            tail_threshold = float(losses.quantile(tail_probability))
            if tail_threshold <= 0.0:
                tail_losses = losses.loc[losses > 0.0]
            else:
                tail_losses = losses.loc[losses >= tail_threshold]
            conditional_tail_loss = float(tail_losses.mean()) if not tail_losses.empty else 0.0
            severe_loss_share = float((losses >= severe_loss_ticks).mean()) if severe_loss_ticks > 0 else 0.0
            loss_run = max_consecutive_losses(executed_edges)
        fragile = (
            conditional_tail_loss > max_tail_loss_ticks
            or severe_loss_share > max_severe_loss_share
            or loss_run > max_loss_run_length
        )
        return {
            "path_id": path_id,
            "rows": len(path),
            "tradable_rows": tradable_rows,
            "loss_rows": loss_rows,
            "mean_loss_ticks": mean_loss,
            "tail_loss_threshold_ticks": tail_threshold,
            "conditional_tail_loss_ticks": conditional_tail_loss,
            "severe_loss_share": severe_loss_share,
            "max_loss_run_length": loss_run,
            "tail_loss_label": "execution_tail_loss_fragile" if fragile else "execution_tail_loss_stable",
        }

    rows: list[dict[str, float | int | str]] = []
    if grouping_columns:
        groupby_arg: str | list[str] = grouping_columns[0] if len(grouping_columns) == 1 else grouping_columns
        for key, group in frame.groupby(groupby_arg, sort=True, dropna=False):
            if isinstance(key, tuple):
                path_id = "|".join(str(part) for part in key)
            else:
                path_id = str(key)
            rows.append(score_path(group, sides.loc[group.index], path_id))
    rows.append(score_path(frame, sides, "overall"))
    return pd.DataFrame(rows, columns=columns)


def queue_position_path_tail_loss_release_gate(
    tail_loss_scorecard: pd.DataFrame,
    *,
    max_fragile_path_share: float = 0.25,
    max_overall_conditional_tail_loss_ticks: float = 1.0,
    max_overall_severe_loss_share: float = 0.25,
    max_overall_loss_run_length: int = 2,
) -> dict[str, float | int | str]:
    """Gate queue-position execution paths on clustered left-tail losses.

    ``queue_position_path_tail_loss_scorecard`` exposes VaR/CVaR-style downside
    that aggregate execution edge and drawdown summaries can miss. This release
    gate turns those path rows into a compact publishability artifact: grouped
    paths limit fragile-tail concentration while the ``overall`` row enforces
    release-level conditional tail loss, severe-loss share, and clustered loss-run
    ceilings.
    """
    for name, value in {
        "max_fragile_path_share": max_fragile_path_share,
        "max_overall_conditional_tail_loss_ticks": max_overall_conditional_tail_loss_ticks,
        "max_overall_severe_loss_share": max_overall_severe_loss_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if max_fragile_path_share < 0.0:
        raise ValueError("max_fragile_path_share must be non-negative")
    if max_overall_conditional_tail_loss_ticks < 0.0:
        raise ValueError("max_overall_conditional_tail_loss_ticks must be non-negative")
    if not 0.0 <= max_overall_severe_loss_share <= 1.0:
        raise ValueError("max_overall_severe_loss_share must be in [0, 1]")
    if not isinstance(max_overall_loss_run_length, int) or isinstance(max_overall_loss_run_length, bool):
        raise ValueError("max_overall_loss_run_length must be an integer")
    if max_overall_loss_run_length < 0:
        raise ValueError("max_overall_loss_run_length must be non-negative")

    required = {
        "path_id",
        "rows",
        "tradable_rows",
        "loss_rows",
        "conditional_tail_loss_ticks",
        "severe_loss_share",
        "max_loss_run_length",
        "tail_loss_label",
    }
    _require_columns(tail_loss_scorecard, required, "queue position path tail loss release gate")
    if tail_loss_scorecard.empty:
        return {
            "paths": 0,
            "fragile_paths": 0,
            "fragile_path_share": 0.0,
            "total_tradable_rows": 0,
            "overall_conditional_tail_loss_ticks": 0.0,
            "overall_severe_loss_share": 0.0,
            "overall_max_loss_run_length": 0,
            "worst_path_id": "none",
            "worst_path_tail_loss_label": "none",
            "tail_loss_release_decision": "review",
            "tail_loss_release_label": "queue_tail_loss_release_review",
            "blocking_reasons": "none",
            "review_reasons": "no_paths",
        }

    numeric_columns = [
        "rows",
        "tradable_rows",
        "loss_rows",
        "conditional_tail_loss_ticks",
        "severe_loss_share",
        "max_loss_run_length",
    ]
    numeric = _finite_values(
        tail_loss_scorecard,
        numeric_columns,
        "queue position path tail loss release gate",
    )
    non_negative = [
        "rows",
        "tradable_rows",
        "loss_rows",
        "conditional_tail_loss_ticks",
        "severe_loss_share",
        "max_loss_run_length",
    ]
    if (numeric[non_negative] < 0.0).any().any():
        raise ValueError("queue position path tail loss release gate metrics must be non-negative")
    if (numeric["severe_loss_share"] > 1.0).any():
        raise ValueError("queue position path tail loss release gate severe loss shares must be in [0, 1]")

    path_ids = tail_loss_scorecard["path_id"].astype(str)
    labels = tail_loss_scorecard["tail_loss_label"].astype(str)
    grouped_paths = tail_loss_scorecard.loc[path_ids != "overall"]
    if grouped_paths.empty:
        grouped_paths = tail_loss_scorecard
    grouped_numeric = numeric.loc[grouped_paths.index]
    grouped_labels = labels.loc[grouped_paths.index]
    paths = int(len(grouped_paths))
    fragile_mask = grouped_labels == "execution_tail_loss_fragile"
    fragile_paths = int(fragile_mask.sum())
    fragile_path_share = float(fragile_paths / paths) if paths else 0.0

    overall_rows = tail_loss_scorecard.loc[path_ids == "overall"]
    if overall_rows.empty:
        overall_numeric = numeric.iloc[-1]
        missing_overall = True
    else:
        overall_numeric = numeric.loc[overall_rows.iloc[-1].name]
        missing_overall = False

    total_tradable_rows = int(overall_numeric["tradable_rows"])
    overall_tail_loss = float(overall_numeric["conditional_tail_loss_ticks"])
    overall_severe_share = float(overall_numeric["severe_loss_share"])
    overall_loss_run = int(overall_numeric["max_loss_run_length"])

    tail_rank = (
        grouped_numeric["conditional_tail_loss_ticks"]
        + grouped_numeric["severe_loss_share"]
        + grouped_numeric["max_loss_run_length"]
    )
    if not grouped_paths.empty:
        worst_index = tail_rank.sort_values(ascending=False).index[0]
        worst_path_id = str(tail_loss_scorecard.loc[worst_index, "path_id"])
        worst_label = str(tail_loss_scorecard.loc[worst_index, "tail_loss_label"])
    else:
        worst_path_id = "none"
        worst_label = "none"

    blocking_reasons: list[str] = []
    if fragile_path_share > max_fragile_path_share:
        blocking_reasons.append("fragile_path_share")
    if overall_tail_loss > max_overall_conditional_tail_loss_ticks:
        blocking_reasons.append("overall_conditional_tail_loss")
    if overall_severe_share > max_overall_severe_loss_share:
        blocking_reasons.append("overall_severe_loss_share")
    if overall_loss_run > max_overall_loss_run_length:
        blocking_reasons.append("overall_loss_run_length")

    review_reasons: list[str] = []
    if missing_overall:
        review_reasons.append("missing_overall_path")
    if total_tradable_rows == 0:
        review_reasons.append("no_tradable_rows")

    if blocking_reasons:
        decision = "block"
        label = "queue_tail_loss_release_blocked"
    elif review_reasons:
        decision = "review"
        label = "queue_tail_loss_release_review"
    else:
        decision = "pass"
        label = "queue_tail_loss_release_pass"

    return {
        "paths": paths,
        "fragile_paths": fragile_paths,
        "fragile_path_share": fragile_path_share,
        "total_tradable_rows": total_tradable_rows,
        "overall_conditional_tail_loss_ticks": overall_tail_loss,
        "overall_severe_loss_share": overall_severe_share,
        "overall_max_loss_run_length": overall_loss_run,
        "worst_path_id": worst_path_id,
        "worst_path_tail_loss_label": worst_label,
        "tail_loss_release_decision": decision,
        "tail_loss_release_label": label,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
    }


def queue_position_path_risk_scorecard(
    frame: pd.DataFrame,
    *,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    edge_col: str = "execution_adjusted_edge_ticks",
    max_drawdown_ticks: float = 2.0,
    max_turnover_rate: float = 0.50,
    min_total_edge_ticks: float = 0.0,
) -> pd.DataFrame:
    """Summarize path-dependent risk for queue-position-aware execution policies.

    Point-in-time execution-adjusted edge can look publishable while the realized
    path is not: fill-aware signals may churn sides, cluster losses, or give back
    edge through drawdowns. This scorecard treats non-abstain rows as the executed
    path, measures cumulative edge, peak-to-trough drawdown from a zero starting
    capital point, and counts long/short side flips. Optional groups expose fragile
    sessions, symbols, or event windows while the appended overall row preserves a
    release-level view.
    """
    if not math.isfinite(max_drawdown_ticks) or max_drawdown_ticks < 0.0:
        raise ValueError("max_drawdown_ticks must be a finite non-negative value")
    if not math.isfinite(max_turnover_rate) or max_turnover_rate < 0.0:
        raise ValueError("max_turnover_rate must be a finite non-negative value")
    if not math.isfinite(min_total_edge_ticks):
        raise ValueError("min_total_edge_ticks must be finite")

    grouping_columns = _normalize_group_columns(frame, group_cols, "queue position path risk group")
    required = {side_col, edge_col, *grouping_columns}
    columns = [
        "path_id",
        "rows",
        "tradable_rows",
        "abstain_rows",
        "mean_edge_ticks",
        "total_edge_ticks",
        "max_drawdown_ticks",
        "hit_rate",
        "turnover_events",
        "turnover_rate",
        "path_risk_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(frame, required, "queue position path risk")
    values = _finite_values(frame, [edge_col], "queue position path risk")
    sides = frame[side_col].astype(str)

    def score_path(path: pd.DataFrame, path_sides: pd.Series, path_id: str) -> dict[str, float | int | str]:
        path_edges = values.loc[path.index, edge_col]
        path_tradable = path_sides != "abstain"
        executed_edges = path_edges.loc[path_tradable]
        executed_sides = path_sides.loc[path_tradable]
        tradable_rows = int(path_tradable.sum())
        if tradable_rows == 0:
            mean_edge = 0.0
            total_edge = 0.0
            max_drawdown = 0.0
            hit_rate = 0.0
            turnover_events = 0
            turnover_rate = 0.0
        else:
            cumulative = executed_edges.cumsum()
            running_peak = cumulative.cummax().clip(lower=0.0)
            drawdown = running_peak - cumulative
            side_changes = executed_sides.ne(executed_sides.shift()).iloc[1:]
            turnover_events = int(side_changes.sum())
            mean_edge = float(executed_edges.mean())
            total_edge = float(executed_edges.sum())
            max_drawdown = float(drawdown.max())
            hit_rate = float((executed_edges > 0.0).mean())
            turnover_rate = float(turnover_events / tradable_rows)
        fragile = (
            total_edge < min_total_edge_ticks
            or max_drawdown > max_drawdown_ticks
            or turnover_rate > max_turnover_rate
        )
        return {
            "path_id": path_id,
            "rows": len(path),
            "tradable_rows": tradable_rows,
            "abstain_rows": int((~path_tradable).sum()),
            "mean_edge_ticks": mean_edge,
            "total_edge_ticks": total_edge,
            "max_drawdown_ticks": max_drawdown,
            "hit_rate": hit_rate,
            "turnover_events": turnover_events,
            "turnover_rate": turnover_rate,
            "path_risk_label": "execution_path_fragile" if fragile else "execution_path_stable",
        }

    rows: list[dict[str, float | int | str]] = []
    if grouping_columns:
        groupby_arg: str | list[str] = grouping_columns[0] if len(grouping_columns) == 1 else grouping_columns
        for key, group in frame.groupby(groupby_arg, sort=True, dropna=False):
            if isinstance(key, tuple):
                path_id = "|".join(str(part) for part in key)
            else:
                path_id = str(key)
            rows.append(score_path(group, sides.loc[group.index], path_id))
    rows.append(score_path(frame, sides, "overall"))
    return pd.DataFrame(rows, columns=columns)


def queue_position_path_risk_concentration(
    path_scorecard: pd.DataFrame,
    *,
    max_top_edge_share: float = 0.70,
    max_top_drawdown_share: float = 0.70,
    max_fragile_path_share: float = 0.25,
) -> dict[str, float | int | str]:
    """Audit whether execution path edge and drawdown are concentrated.

    A queue-position policy can pass aggregate path-risk checks while most of its
    edge comes from one event path or most of its drawdown comes from one fragile
    session. This release-facing diagnostic excludes the synthetic ``overall`` row
    and reports edge/drawdown concentration shares plus HHI-style concentration so
    reviewers can distinguish diversified execution evidence from a crowded path.
    """
    for name, value in {
        "max_top_edge_share": max_top_edge_share,
        "max_top_drawdown_share": max_top_drawdown_share,
        "max_fragile_path_share": max_fragile_path_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_top_edge_share <= 1.0:
        raise ValueError("max_top_edge_share must be in [0, 1]")
    if not 0.0 <= max_top_drawdown_share <= 1.0:
        raise ValueError("max_top_drawdown_share must be in [0, 1]")
    if not 0.0 <= max_fragile_path_share <= 1.0:
        raise ValueError("max_fragile_path_share must be in [0, 1]")

    required = {
        "path_id",
        "tradable_rows",
        "total_edge_ticks",
        "max_drawdown_ticks",
        "path_risk_label",
    }
    missing = sorted(required - set(path_scorecard.columns))
    if missing:
        raise ValueError(f"missing queue position path concentration columns: {missing}")

    paths = path_scorecard[path_scorecard["path_id"].astype(str) != "overall"].copy()
    if paths.empty:
        return {
            "paths": 0,
            "fragile_paths": 0,
            "fragile_path_share": 0.0,
            "positive_edge_paths": 0,
            "drawdown_paths": 0,
            "top_edge_path_id": "none",
            "top_edge_share": 0.0,
            "edge_concentration_hhi": 0.0,
            "top_drawdown_path_id": "none",
            "top_drawdown_share": 0.0,
            "drawdown_concentration_hhi": 0.0,
            "path_concentration_label": "queue_path_concentration_empty",
            "review_reasons": "no_paths",
        }

    numeric = paths[["tradable_rows", "total_edge_ticks", "max_drawdown_ticks"]].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("queue position path concentration metrics must be finite")
    if (numeric[["tradable_rows", "max_drawdown_ticks"]] < 0.0).any().any():
        raise ValueError("queue position path concentration counts and drawdowns must be non-negative")

    fragile_mask = paths["path_risk_label"].astype(str) == "execution_path_fragile"
    fragile_paths = int(fragile_mask.sum())
    path_count = int(len(paths))
    fragile_path_share = fragile_paths / path_count

    positive_edge = numeric["total_edge_ticks"].clip(lower=0.0)
    positive_edge_total = float(positive_edge.sum())
    if positive_edge_total > 0.0:
        edge_shares = positive_edge / positive_edge_total
        top_edge_idx = edge_shares.idxmax()
        top_edge_share = float(edge_shares.loc[top_edge_idx])
        edge_hhi = float((edge_shares**2).sum())
        top_edge_path_id = str(paths.loc[top_edge_idx, "path_id"])
        positive_edge_paths = int((positive_edge > 0.0).sum())
    else:
        top_edge_path_id = "none"
        top_edge_share = 0.0
        edge_hhi = 0.0
        positive_edge_paths = 0

    drawdown = numeric["max_drawdown_ticks"]
    drawdown_total = float(drawdown.sum())
    if drawdown_total > 0.0:
        drawdown_shares = drawdown / drawdown_total
        top_drawdown_idx = drawdown_shares.idxmax()
        top_drawdown_share = float(drawdown_shares.loc[top_drawdown_idx])
        drawdown_hhi = float((drawdown_shares**2).sum())
        top_drawdown_path_id = str(paths.loc[top_drawdown_idx, "path_id"])
        drawdown_paths = int((drawdown > 0.0).sum())
    else:
        top_drawdown_path_id = "none"
        top_drawdown_share = 0.0
        drawdown_hhi = 0.0
        drawdown_paths = 0

    reasons: list[str] = []
    if top_edge_share > max_top_edge_share:
        reasons.append("edge_concentration")
    if top_drawdown_share > max_top_drawdown_share:
        reasons.append("drawdown_concentration")
    if fragile_path_share > max_fragile_path_share:
        reasons.append("fragile_path_share")
    label = "queue_path_concentration_fragile" if reasons else "queue_path_concentration_diversified"

    return {
        "paths": path_count,
        "fragile_paths": fragile_paths,
        "fragile_path_share": float(fragile_path_share),
        "positive_edge_paths": positive_edge_paths,
        "drawdown_paths": drawdown_paths,
        "top_edge_path_id": top_edge_path_id,
        "top_edge_share": top_edge_share,
        "edge_concentration_hhi": edge_hhi,
        "top_drawdown_path_id": top_drawdown_path_id,
        "top_drawdown_share": top_drawdown_share,
        "drawdown_concentration_hhi": drawdown_hhi,
        "path_concentration_label": label,
        "review_reasons": ";".join(reasons) if reasons else "none",
    }


def queue_position_path_risk_release_gate(
    path_scorecard: pd.DataFrame,
    *,
    max_fragile_path_share: float = 0.25,
    max_overall_drawdown_ticks: float = 2.0,
    min_overall_total_edge_ticks: float = 0.0,
    max_overall_turnover_rate: float = 0.50,
) -> dict[str, float | int | str]:
    """Gate queue-position execution paths on drawdown, edge, and side churn.

    ``queue_position_path_risk_scorecard`` exposes the path risks that pointwise
    execution-adjusted edge can hide. This release gate turns that scorecard into
    a compact publishability artifact: grouped paths carry concentration risk,
    while the ``overall`` row enforces aggregate drawdown, total edge, and churn
    limits before an execution-aware LCRI demo is allowed to pass.
    """
    for name, value in {
        "max_fragile_path_share": max_fragile_path_share,
        "max_overall_drawdown_ticks": max_overall_drawdown_ticks,
        "min_overall_total_edge_ticks": min_overall_total_edge_ticks,
        "max_overall_turnover_rate": max_overall_turnover_rate,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if max_fragile_path_share < 0.0:
        raise ValueError("max_fragile_path_share must be non-negative")
    if max_overall_drawdown_ticks < 0.0:
        raise ValueError("max_overall_drawdown_ticks must be non-negative")
    if max_overall_turnover_rate < 0.0:
        raise ValueError("max_overall_turnover_rate must be non-negative")

    required = {
        "path_id",
        "rows",
        "tradable_rows",
        "total_edge_ticks",
        "max_drawdown_ticks",
        "turnover_rate",
        "path_risk_label",
    }
    _require_columns(path_scorecard, required, "queue position path risk release gate")
    if path_scorecard.empty:
        return {
            "paths": 0,
            "fragile_paths": 0,
            "fragile_path_share": 0.0,
            "total_tradable_rows": 0,
            "overall_total_edge_ticks": 0.0,
            "overall_max_drawdown_ticks": 0.0,
            "overall_turnover_rate": 0.0,
            "worst_path_id": "none",
            "worst_path_risk_label": "none",
            "path_risk_release_decision": "review",
            "path_risk_release_label": "queue_path_risk_release_review",
            "blocking_reasons": "none",
            "review_reasons": "no_paths",
        }

    numeric_columns = [
        "rows",
        "tradable_rows",
        "total_edge_ticks",
        "max_drawdown_ticks",
        "turnover_rate",
    ]
    numeric = _finite_values(path_scorecard, numeric_columns, "queue position path risk release gate")
    if (numeric[["rows", "tradable_rows", "max_drawdown_ticks", "turnover_rate"]] < 0.0).any().any():
        raise ValueError("queue position path risk release gate counts and risk values must be non-negative")

    path_ids = path_scorecard["path_id"].astype(str)
    labels = path_scorecard["path_risk_label"].astype(str)
    grouped_paths = path_scorecard.loc[path_ids != "overall"]
    if grouped_paths.empty:
        grouped_paths = path_scorecard
    grouped_numeric = numeric.loc[grouped_paths.index]
    grouped_labels = labels.loc[grouped_paths.index]
    paths = int(len(grouped_paths))
    fragile_mask = grouped_labels == "execution_path_fragile"
    fragile_paths = int(fragile_mask.sum())
    fragile_path_share = float(fragile_paths / paths) if paths else 0.0

    overall_rows = path_scorecard.loc[path_ids == "overall"]
    if overall_rows.empty:
        overall = path_scorecard.iloc[-1]
        overall_numeric = numeric.iloc[-1]
        missing_overall = True
    else:
        overall = overall_rows.iloc[-1]
        overall_numeric = numeric.loc[overall.name]
        missing_overall = False

    total_tradable_rows = int(overall_numeric["tradable_rows"])
    overall_total_edge = float(overall_numeric["total_edge_ticks"])
    overall_drawdown = float(overall_numeric["max_drawdown_ticks"])
    overall_turnover = float(overall_numeric["turnover_rate"])

    risk_rank = grouped_numeric["max_drawdown_ticks"] + grouped_numeric["turnover_rate"]
    risk_rank = risk_rank - grouped_numeric["total_edge_ticks"].clip(upper=0.0)
    if not grouped_paths.empty:
        worst_index = risk_rank.sort_values(ascending=False).index[0]
        worst_path_id = str(path_scorecard.loc[worst_index, "path_id"])
        worst_label = str(path_scorecard.loc[worst_index, "path_risk_label"])
    else:
        worst_path_id = "none"
        worst_label = "none"

    blocking_reasons: list[str] = []
    if fragile_path_share > max_fragile_path_share:
        blocking_reasons.append("fragile_path_share")
    if overall_drawdown > max_overall_drawdown_ticks:
        blocking_reasons.append("overall_drawdown")
    if overall_total_edge < min_overall_total_edge_ticks:
        blocking_reasons.append("overall_total_edge")
    if overall_turnover > max_overall_turnover_rate:
        blocking_reasons.append("overall_turnover")

    review_reasons: list[str] = []
    if missing_overall:
        review_reasons.append("missing_overall_path")
    if total_tradable_rows == 0:
        review_reasons.append("no_tradable_rows")

    if blocking_reasons:
        decision = "block"
        label = "queue_path_risk_release_blocked"
    elif review_reasons:
        decision = "review"
        label = "queue_path_risk_release_review"
    else:
        decision = "pass"
        label = "queue_path_risk_release_pass"

    return {
        "paths": paths,
        "fragile_paths": fragile_paths,
        "fragile_path_share": fragile_path_share,
        "total_tradable_rows": total_tradable_rows,
        "overall_total_edge_ticks": overall_total_edge,
        "overall_max_drawdown_ticks": overall_drawdown,
        "overall_turnover_rate": overall_turnover,
        "worst_path_id": worst_path_id,
        "worst_path_risk_label": worst_label,
        "path_risk_release_decision": decision,
        "path_risk_release_label": label,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
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


def execution_adjusted_lcri_side_release_scorecard(
    attribution: pd.DataFrame,
    *,
    min_tradable_share: float = 0.50,
    max_conflict_share: float = 0.25,
    min_mean_edge_ticks: float = 0.0,
) -> dict[str, bool | float | int | str]:
    """Gate execution-adjusted LCRI side survival before publishing side claims.

    ``execution_adjusted_lcri_side_attribution`` explains whether raw long/short
    LCRI pressure survives the passive execution layer. This scorecard compresses
    that artifact into release criteria: directional LCRI sides need enough
    tradable coverage, bounded queue-induced side conflicts, no dominant side
    inversion, and non-negative execution-adjusted edge after fill/adverse drag.
    """
    for name, value in {
        "min_tradable_share": min_tradable_share,
        "max_conflict_share": max_conflict_share,
    }.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not math.isfinite(min_mean_edge_ticks):
        raise ValueError("min_mean_edge_ticks must be finite")

    required = {
        "lcri_side",
        "rows",
        "tradable_rows",
        "execution_conflict_share",
        "mean_execution_adjusted_edge_ticks",
        "mean_fill_probability_advantage",
        "dominant_execution_side",
        "review_label",
    }
    missing = sorted(required - set(attribution.columns))
    if missing:
        raise ValueError(f"missing execution-adjusted LCRI side scorecard columns: {missing}")

    if attribution.empty:
        return {
            "side_rows": 0,
            "directional_rows": 0,
            "directional_tradable_rows": 0,
            "directional_tradable_share": 0.0,
            "max_directional_conflict_share": 0.0,
            "inverted_side_count": 0,
            "negative_edge_side_count": 0,
            "weak_fill_advantage_side_count": 0,
            "worst_side": "none",
            "release_decision": "review",
            "review_note": "execution_lcri_side_insufficient_coverage",
        }

    values = _finite_values(
        attribution,
        [
            "rows",
            "tradable_rows",
            "execution_conflict_share",
            "mean_execution_adjusted_edge_ticks",
            "mean_fill_probability_advantage",
        ],
        "execution-adjusted LCRI side scorecard",
    )
    if (values[["rows", "tradable_rows"]] < 0.0).any().any():
        raise ValueError("execution-adjusted LCRI side scorecard row counts must be non-negative")
    if not values["execution_conflict_share"].between(0.0, 1.0).all():
        raise ValueError("execution-adjusted LCRI side scorecard conflict shares must be in [0, 1]")

    lcri_side = attribution["lcri_side"].astype(str)
    directional = attribution[lcri_side.isin(["long", "short"])].copy()
    if directional.empty:
        return {
            "side_rows": int(len(attribution)),
            "directional_rows": 0,
            "directional_tradable_rows": 0,
            "directional_tradable_share": 0.0,
            "max_directional_conflict_share": 0.0,
            "inverted_side_count": 0,
            "negative_edge_side_count": 0,
            "weak_fill_advantage_side_count": 0,
            "worst_side": "none",
            "release_decision": "review",
            "review_note": "execution_lcri_side_insufficient_coverage",
        }

    directional_values = values.loc[directional.index]
    directional_rows = int(round(float(directional_values["rows"].sum())))
    directional_tradable_rows = int(round(float(directional_values["tradable_rows"].sum())))
    directional_tradable_share = (
        float(directional_tradable_rows / directional_rows) if directional_rows else 0.0
    )
    max_directional_conflict_share = float(directional_values["execution_conflict_share"].max())
    inverted = directional["review_label"].astype(str) == "execution_side_inversion_review"
    negative_edge = directional_values["mean_execution_adjusted_edge_ticks"] < min_mean_edge_ticks
    weak_fill_advantage = directional_values["mean_fill_probability_advantage"] < 0.0
    risk_score = (
        directional_values["execution_conflict_share"]
        - directional_values["mean_execution_adjusted_edge_ticks"].clip(upper=0.0)
        - directional_values["mean_fill_probability_advantage"].clip(upper=0.0)
    )
    worst_side = str(directional.iloc[int(np.argmax(risk_score.to_numpy()))]["lcri_side"])

    inverted_side_count = int(inverted.sum())
    negative_edge_side_count = int(negative_edge.sum())
    weak_fill_advantage_side_count = int(weak_fill_advantage.sum())
    coverage_blocked = directional_tradable_share < min_tradable_share
    conflict_blocked = max_directional_conflict_share > max_conflict_share
    edge_blocked = negative_edge_side_count > 0
    fill_blocked = weak_fill_advantage_side_count > 0
    if inverted_side_count > 0:
        release_decision = "block"
        review_note = "execution_lcri_side_inversion_blocked"
    elif coverage_blocked:
        release_decision = "review"
        review_note = "execution_lcri_side_insufficient_coverage"
    elif conflict_blocked or edge_blocked or fill_blocked:
        release_decision = "review"
        review_note = "execution_lcri_side_friction_review"
    else:
        release_decision = "pass"
        review_note = "execution_lcri_side_supported"

    return {
        "side_rows": int(len(attribution)),
        "directional_rows": directional_rows,
        "directional_tradable_rows": directional_tradable_rows,
        "directional_tradable_share": directional_tradable_share,
        "max_directional_conflict_share": max_directional_conflict_share,
        "inverted_side_count": inverted_side_count,
        "negative_edge_side_count": negative_edge_side_count,
        "weak_fill_advantage_side_count": weak_fill_advantage_side_count,
        "worst_side": worst_side,
        "release_decision": release_decision,
        "review_note": review_note,
    }


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


def _empty_execution_adjusted_lcri_absorption_attribution() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "absorption_regime",
            "rows",
            "publishable_rows",
            "executable_rows",
            "conflict_rows",
            "conflict_share",
            "negative_edge_share",
            "mean_execution_adjusted_edge_ticks",
            "mean_selected_fill_probability",
            "mean_selected_adverse_fill_probability",
            "mean_fill_minus_adverse_probability",
            "absorption_execution_label",
        ]
    )


def _absorption_execution_label(
    *,
    conflict_share: float,
    negative_edge_share: float,
    mean_fill_minus_adverse_probability: float,
    executable_rows: int,
    min_fill_minus_adverse_probability: float,
    max_negative_edge_share: float,
    max_conflict_share: float,
) -> str:
    if executable_rows == 0:
        return "absorption_execution_sparse"
    if (
        negative_edge_share > max_negative_edge_share
        or mean_fill_minus_adverse_probability < min_fill_minus_adverse_probability
    ):
        return "absorption_execution_toxic"
    if conflict_share > max_conflict_share:
        return "absorption_execution_conflicted"
    return "absorption_execution_publishable"


def execution_adjusted_lcri_absorption_attribution(
    frame: pd.DataFrame,
    *,
    absorption_col: str = "absorption_regime",
    min_fill_minus_adverse_probability: float = 0.05,
    max_negative_edge_share: float = 0.50,
    max_conflict_share: float = 0.25,
) -> pd.DataFrame:
    """Audit execution-adjusted LCRI tradability across absorption regimes.

    Shadow absorption can make a residual imbalance look directionally interesting
    while preventing passive execution from monetizing it. This diagnostic groups
    rows by absorption regime and reports whether pre-execution publishable sides
    survive queue-aware execution, whether selected-side fills are toxic, and where
    absorption regimes should block or focus publishability review.
    """
    for name, value in {
        "min_fill_minus_adverse_probability": min_fill_minus_adverse_probability,
        "max_negative_edge_share": max_negative_edge_share,
        "max_conflict_share": max_conflict_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_negative_edge_share <= 1.0:
        raise ValueError("max_negative_edge_share must be in [0, 1]")
    if not 0.0 <= max_conflict_share <= 1.0:
        raise ValueError("max_conflict_share must be in [0, 1]")
    if frame.empty:
        return _empty_execution_adjusted_lcri_absorption_attribution()

    required = {
        absorption_col,
        "publishable_side",
        "best_execution_side",
        "execution_adjusted_edge_ticks",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    _require_columns(frame, required, "execution-adjusted LCRI absorption attribution")
    values = _finite_values(
        frame,
        [
            "execution_adjusted_edge_ticks",
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ],
        "execution-adjusted LCRI absorption attribution",
    )
    probability_columns = [
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    ]
    if not values[probability_columns].apply(lambda column: column.between(0.0, 1.0).all()).all():
        raise ValueError("execution-adjusted LCRI absorption probabilities must be in [0, 1]")

    publishable_side = frame["publishable_side"].astype(str)
    best_side = frame["best_execution_side"].astype(str)
    valid_publishable_sides = {"long", "short", "abstain"}
    valid_execution_sides = {"long", "short", "abstain"}
    unknown_publishable = sorted(set(publishable_side) - valid_publishable_sides)
    unknown_best = sorted(set(best_side) - valid_execution_sides)
    if unknown_publishable:
        raise ValueError(f"unknown publishable sides: {unknown_publishable}")
    if unknown_best:
        raise ValueError(f"unknown execution sides: {unknown_best}")

    diagnostics = pd.DataFrame(index=frame.index)
    diagnostics["absorption_regime"] = frame[absorption_col].astype(str)
    diagnostics["publishable"] = publishable_side != "abstain"
    diagnostics["executable"] = best_side != "abstain"
    diagnostics["conflict"] = publishable_side != best_side
    diagnostics["execution_adjusted_edge_ticks"] = values["execution_adjusted_edge_ticks"]
    diagnostics["negative_edge"] = values["execution_adjusted_edge_ticks"] < 0.0
    diagnostics["selected_fill_probability"] = _side_probability(
        best_side,
        bid=values["bid_fill_probability"],
        ask=values["ask_fill_probability"],
    )
    diagnostics["selected_adverse_fill_probability"] = _side_probability(
        best_side,
        bid=values["bid_adverse_fill_probability"],
        ask=values["ask_adverse_fill_probability"],
    )
    diagnostics["fill_minus_adverse_probability"] = (
        diagnostics["selected_fill_probability"]
        - diagnostics["selected_adverse_fill_probability"]
    )

    rows: list[dict[str, float | int | str]] = []
    for absorption_regime, group in diagnostics.groupby("absorption_regime", sort=True):
        conflict_share = float(group["conflict"].mean())
        negative_edge_share = float(group["negative_edge"].mean())
        mean_fill_minus_adverse_probability = float(group["fill_minus_adverse_probability"].mean())
        executable_rows = int(group["executable"].sum())
        rows.append(
            {
                "absorption_regime": str(absorption_regime),
                "rows": int(len(group)),
                "publishable_rows": int(group["publishable"].sum()),
                "executable_rows": executable_rows,
                "conflict_rows": int(group["conflict"].sum()),
                "conflict_share": conflict_share,
                "negative_edge_share": negative_edge_share,
                "mean_execution_adjusted_edge_ticks": float(
                    group["execution_adjusted_edge_ticks"].mean()
                ),
                "mean_selected_fill_probability": float(group["selected_fill_probability"].mean()),
                "mean_selected_adverse_fill_probability": float(
                    group["selected_adverse_fill_probability"].mean()
                ),
                "mean_fill_minus_adverse_probability": mean_fill_minus_adverse_probability,
                "absorption_execution_label": _absorption_execution_label(
                    conflict_share=conflict_share,
                    negative_edge_share=negative_edge_share,
                    mean_fill_minus_adverse_probability=mean_fill_minus_adverse_probability,
                    executable_rows=executable_rows,
                    min_fill_minus_adverse_probability=min_fill_minus_adverse_probability,
                    max_negative_edge_share=max_negative_edge_share,
                    max_conflict_share=max_conflict_share,
                ),
            }
        )
    output = pd.DataFrame(rows)[list(_empty_execution_adjusted_lcri_absorption_attribution().columns)]
    if absorption_col != "absorption_regime":
        output = output.rename(columns={"absorption_regime": absorption_col})
    return output.sort_values(
        ["absorption_execution_label", "conflict_share", "negative_edge_share", absorption_col],
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


def _empty_execution_adjusted_lcri_event_window_attribution() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "passive_fill_event_window_regime",
            "bucket",
            "rows",
            "row_share",
            "mean_abs_lcri",
            "mean_abs_execution_adjusted_lcri_score",
            "signal_survival_ratio",
            "tradable_share",
            "mean_selected_fill_probability",
            "mean_selected_adverse_fill_probability",
            "fill_minus_adverse_probability_spread",
            "mean_execution_adjusted_edge_ticks",
            "negative_edge_share",
            "edge_drag_vs_raw_abs_lcri",
            "event_window_execution_label",
        ]
    )


def _is_high_lcri_bucket(bucket: str) -> bool:
    if bucket == "high_abs_lcri":
        return True
    if bucket.startswith("abs_lcri_q"):
        try:
            quantile_index = int(bucket.removeprefix("abs_lcri_q"))
        except ValueError:
            return False
        return quantile_index >= 3
    return False


def _event_window_execution_label(
    *,
    bucket: str,
    regime: str,
    signal_survival_ratio: float,
    negative_edge_share: float,
    mean_selected_adverse_fill_probability: float,
    mean_execution_adjusted_edge_ticks: float,
) -> str:
    is_high_lcri = _is_high_lcri_bucket(bucket)
    is_event_window = regime in {"pre_event", "event", "post_event"}
    if not is_high_lcri:
        return "low_lcri_reference"
    if (
        is_event_window
        and negative_edge_share >= 0.50
        and (signal_survival_ratio < 0.35 or mean_selected_adverse_fill_probability >= 0.50)
    ):
        return "high_lcri_event_toxicity"
    if mean_execution_adjusted_edge_ticks < 0.0:
        return "execution_edge_negative"
    return "event_window_edge_survives" if is_event_window else "non_event_edge_survives"


def execution_adjusted_lcri_event_window_attribution(
    frame: pd.DataFrame,
    *,
    bins: int = 3,
    event_window_col: str = "passive_fill_event_window_regime",
    signal_col: str = "lcri",
    execution_signal_col: str = "execution_adjusted_lcri_score",
) -> pd.DataFrame:
    """Attribute execution-adjusted LCRI survival inside passive-fill event windows.

    Large residual imbalance is only publishable if it remains economically
    tradable near queue-position fill events. This diagnostic cross-tabulates raw
    absolute LCRI strength with row-level passive-fill event-window regimes so
    reviews can separate genuine high-LCRI opportunities from event-neighborhoods
    where passive fills are available but adverse selection erases the edge.
    """
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if frame.empty:
        return _empty_execution_adjusted_lcri_event_window_attribution()

    required = {
        event_window_col,
        signal_col,
        execution_signal_col,
        "execution_adjusted_edge_ticks",
        "best_execution_side",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    _require_columns(frame, required, "execution-adjusted LCRI event-window attribution")
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
        "execution-adjusted LCRI event-window attribution",
    )

    attribution = pd.DataFrame(index=frame.index)
    attribution["passive_fill_event_window_regime"] = frame[event_window_col].astype(str)
    attribution["abs_lcri"] = values[signal_col].abs()
    attribution["abs_execution_adjusted_lcri_score"] = values[execution_signal_col].abs()
    attribution["execution_adjusted_edge_ticks"] = values["execution_adjusted_edge_ticks"]
    best_side = frame["best_execution_side"].astype(str)
    attribution["tradable"] = best_side != "abstain"
    attribution["selected_fill_probability"] = np.select(
        [best_side == "long", best_side == "short"],
        [values["bid_fill_probability"], values["ask_fill_probability"]],
        default=0.0,
    )
    attribution["selected_adverse_fill_probability"] = np.select(
        [best_side == "long", best_side == "short"],
        [values["bid_adverse_fill_probability"], values["ask_adverse_fill_probability"]],
        default=0.0,
    )

    actual_bins = min(bins, len(attribution))
    ranks = attribution["abs_lcri"].rank(method="first")
    attribution["bucket_id"] = pd.qcut(ranks, q=actual_bins, labels=False, duplicates="drop")
    actual_bins = int(attribution["bucket_id"].max()) + 1
    if actual_bins == 1:
        labels = ["all_abs_lcri"]
    elif actual_bins == 2:
        labels = ["low_abs_lcri", "high_abs_lcri"]
    elif actual_bins == 3:
        labels = ["low_abs_lcri", "mid_abs_lcri", "high_abs_lcri"]
    else:
        labels = [f"abs_lcri_q{index + 1:02d}" for index in range(actual_bins)]
    attribution["bucket"] = attribution["bucket_id"].map(lambda bucket_id: labels[int(bucket_id)])

    total_rows = float(len(attribution))
    rows: list[dict[str, float | int | str]] = []
    for (regime, bucket), group in attribution.groupby(
        ["passive_fill_event_window_regime", "bucket"], sort=True
    ):
        mean_abs_lcri = float(group["abs_lcri"].mean())
        mean_abs_execution_signal = float(group["abs_execution_adjusted_lcri_score"].mean())
        mean_selected_fill_probability = float(group["selected_fill_probability"].mean())
        mean_selected_adverse_fill_probability = float(
            group["selected_adverse_fill_probability"].mean()
        )
        signal_survival_ratio = (
            mean_abs_execution_signal / mean_abs_lcri if mean_abs_lcri > 0.0 else 0.0
        )
        mean_execution_adjusted_edge_ticks = float(group["execution_adjusted_edge_ticks"].mean())
        negative_edge_share = float((group["execution_adjusted_edge_ticks"] < 0.0).mean())
        rows.append(
            {
                "passive_fill_event_window_regime": str(regime),
                "bucket": str(bucket),
                "rows": int(len(group)),
                "row_share": float(len(group)) / total_rows if total_rows else 0.0,
                "mean_abs_lcri": mean_abs_lcri,
                "mean_abs_execution_adjusted_lcri_score": mean_abs_execution_signal,
                "signal_survival_ratio": signal_survival_ratio,
                "tradable_share": float(group["tradable"].mean()),
                "mean_selected_fill_probability": mean_selected_fill_probability,
                "mean_selected_adverse_fill_probability": mean_selected_adverse_fill_probability,
                "fill_minus_adverse_probability_spread": float(
                    mean_selected_fill_probability - mean_selected_adverse_fill_probability
                ),
                "mean_execution_adjusted_edge_ticks": mean_execution_adjusted_edge_ticks,
                "negative_edge_share": negative_edge_share,
                "edge_drag_vs_raw_abs_lcri": float(
                    mean_abs_lcri - mean_execution_adjusted_edge_ticks
                ),
                "event_window_execution_label": _event_window_execution_label(
                    bucket=str(bucket),
                    regime=str(regime),
                    signal_survival_ratio=signal_survival_ratio,
                    negative_edge_share=negative_edge_share,
                    mean_selected_adverse_fill_probability=mean_selected_adverse_fill_probability,
                    mean_execution_adjusted_edge_ticks=mean_execution_adjusted_edge_ticks,
                ),
            }
        )
    columns = list(_empty_execution_adjusted_lcri_event_window_attribution().columns)
    return pd.DataFrame(rows)[columns].sort_values(
        [
            "event_window_execution_label",
            "passive_fill_event_window_regime",
            "bucket",
        ],
        ascending=[True, True, True],
        ignore_index=True,
    )


def execution_adjusted_lcri_event_window_release_scorecard(
    attribution: pd.DataFrame,
    *,
    max_toxic_high_lcri_row_share: float = 0.25,
    max_event_toxic_high_lcri_row_share: float = 0.50,
    min_high_lcri_survival_ratio: float = 0.50,
    min_high_lcri_fill_adverse_spread: float = 0.05,
) -> dict[str, float | int | str]:
    """Gate execution-adjusted LCRI release on event-window survivability.

    ``execution_adjusted_lcri_event_window_attribution`` exposes where raw LCRI
    strength survives queue-position-aware passive execution. This scorecard turns
    that surface into a publishability decision: high-LCRI rows should not be
    concentrated in toxic passive-fill event windows, and the row-weighted
    execution-adjusted signal should retain enough magnitude and fill-minus-
    adverse spread to be economically credible.
    """
    for name, value in {
        "max_toxic_high_lcri_row_share": max_toxic_high_lcri_row_share,
        "max_event_toxic_high_lcri_row_share": max_event_toxic_high_lcri_row_share,
        "min_high_lcri_survival_ratio": min_high_lcri_survival_ratio,
        "min_high_lcri_fill_adverse_spread": min_high_lcri_fill_adverse_spread,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_toxic_high_lcri_row_share <= 1.0:
        raise ValueError("max_toxic_high_lcri_row_share must be between 0 and 1")
    if not 0.0 <= max_event_toxic_high_lcri_row_share <= 1.0:
        raise ValueError("max_event_toxic_high_lcri_row_share must be between 0 and 1")

    required = {
        "passive_fill_event_window_regime",
        "bucket",
        "rows",
        "signal_survival_ratio",
        "fill_minus_adverse_probability_spread",
        "negative_edge_share",
        "event_window_execution_label",
    }
    _require_columns(
        attribution,
        required,
        "execution-adjusted LCRI event-window release scorecard",
    )
    if attribution.empty:
        return {
            "high_lcri_rows": 0,
            "toxic_high_lcri_rows": 0,
            "toxic_high_lcri_row_share": 0.0,
            "event_high_lcri_rows": 0,
            "event_toxic_high_lcri_rows": 0,
            "event_toxic_high_lcri_row_share": 0.0,
            "weighted_high_lcri_signal_survival_ratio": 0.0,
            "weighted_high_lcri_fill_adverse_spread": 0.0,
            "weighted_high_lcri_negative_edge_share": 0.0,
            "worst_event_window_regime": "none",
            "worst_event_window_bucket": "none",
            "worst_event_window_label": "none",
            "release_decision": "review",
            "release_label": "execution_lcri_event_window_review",
            "blocking_reasons": "none",
            "review_reasons": "no_high_lcri_rows",
        }

    values = _finite_values(
        attribution,
        [
            "rows",
            "signal_survival_ratio",
            "fill_minus_adverse_probability_spread",
            "negative_edge_share",
        ],
        "execution-adjusted LCRI event-window release scorecard",
    )
    if (values["rows"] < 0.0).any():
        raise ValueError("execution-adjusted LCRI event-window release rows must be non-negative")

    data = attribution.copy()
    for column in values.columns:
        data[column] = values[column]
    high_lcri = data["bucket"].astype(str).map(_is_high_lcri_bucket)
    toxic = data["event_window_execution_label"].astype(str).eq("high_lcri_event_toxicity")
    event_window = data["passive_fill_event_window_regime"].astype(str).eq("event")
    high_lcri_rows = int(data.loc[high_lcri, "rows"].sum())
    toxic_high_lcri_rows = int(data.loc[high_lcri & toxic, "rows"].sum())
    event_high_lcri_rows = int(data.loc[high_lcri & event_window, "rows"].sum())
    event_toxic_high_lcri_rows = int(data.loc[high_lcri & event_window & toxic, "rows"].sum())
    toxic_share = toxic_high_lcri_rows / high_lcri_rows if high_lcri_rows else 0.0
    event_toxic_share = (
        event_toxic_high_lcri_rows / event_high_lcri_rows if event_high_lcri_rows else 0.0
    )

    if high_lcri_rows:
        weights = data.loc[high_lcri, "rows"].astype(float)
        weighted_survival = float(
            np.average(data.loc[high_lcri, "signal_survival_ratio"], weights=weights)
        )
        weighted_spread = float(
            np.average(data.loc[high_lcri, "fill_minus_adverse_probability_spread"], weights=weights)
        )
        weighted_negative_edge = float(
            np.average(data.loc[high_lcri, "negative_edge_share"], weights=weights)
        )
    else:
        weighted_survival = 0.0
        weighted_spread = 0.0
        weighted_negative_edge = 0.0

    if high_lcri.any():
        worst_candidates = data.loc[high_lcri].assign(
            _toxicity_rank=toxic.loc[high_lcri].astype(int).to_numpy(),
        )
        worst = worst_candidates.sort_values(
            ["_toxicity_rank", "negative_edge_share", "signal_survival_ratio", "rows"],
            ascending=[False, False, True, False],
        ).iloc[0]
        worst_regime = str(worst["passive_fill_event_window_regime"])
        worst_bucket = str(worst["bucket"])
        worst_label = str(worst["event_window_execution_label"])
    else:
        worst_regime = "none"
        worst_bucket = "none"
        worst_label = "none"

    blocking_reasons = []
    review_reasons = []
    if high_lcri_rows == 0:
        review_reasons.append("no_high_lcri_rows")
    else:
        if toxic_share > max_toxic_high_lcri_row_share:
            blocking_reasons.append("toxic_high_lcri_share")
        if event_toxic_share > max_event_toxic_high_lcri_row_share:
            blocking_reasons.append("event_toxic_high_lcri_share")
        if weighted_survival < min_high_lcri_survival_ratio:
            blocking_reasons.append("low_signal_survival")
        if weighted_spread < min_high_lcri_fill_adverse_spread:
            blocking_reasons.append("low_fill_adverse_spread")

    if blocking_reasons:
        release_decision = "block"
        release_label = "execution_lcri_event_window_blocked"
    elif review_reasons:
        release_decision = "review"
        release_label = "execution_lcri_event_window_review"
    else:
        release_decision = "pass"
        release_label = "execution_lcri_event_window_pass"

    return {
        "high_lcri_rows": high_lcri_rows,
        "toxic_high_lcri_rows": toxic_high_lcri_rows,
        "toxic_high_lcri_row_share": float(toxic_share),
        "event_high_lcri_rows": event_high_lcri_rows,
        "event_toxic_high_lcri_rows": event_toxic_high_lcri_rows,
        "event_toxic_high_lcri_row_share": float(event_toxic_share),
        "weighted_high_lcri_signal_survival_ratio": weighted_survival,
        "weighted_high_lcri_fill_adverse_spread": weighted_spread,
        "weighted_high_lcri_negative_edge_share": weighted_negative_edge,
        "worst_event_window_regime": worst_regime,
        "worst_event_window_bucket": worst_bucket,
        "worst_event_window_label": worst_label,
        "release_decision": release_decision,
        "release_label": release_label,
        "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "none",
        "review_reasons": ";".join(review_reasons) if review_reasons else "none",
    }


def _empty_queue_position_trade_confirmation_surface() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "queue_clear_bucket",
            "rows",
            "mean_queue_clear_share",
            "mean_predicted_fill_probability",
            "trade_confirmed_fill_rate",
            "cancel_only_clear_rate",
            "mean_trade_confirmed_fill_latency",
            "stale_trade_confirmed_fill_share",
            "confirmation_calibration_error",
            "confirmation_shortfall",
            "confirmation_surface_label",
        ]
    )


def _trade_confirmation_surface_label(
    *,
    confirmation_shortfall: float,
    cancel_only_clear_rate: float,
    trade_confirmed_fill_rate: float,
    stale_trade_confirmed_fill_share: float,
) -> str:
    if stale_trade_confirmed_fill_share > 0.50:
        return "latency_risk"
    if confirmation_shortfall >= 0.20:
        return "high_prediction_not_trade_confirmed"
    if cancel_only_clear_rate > trade_confirmed_fill_rate:
        return "cancel_driven_queue_clearance"
    return "trade_confirmed_execution_ok"


def _empty_queue_position_trade_confirmation_regime_scorecard() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "cells",
            "supported_cells",
            "rows",
            "supported_rows",
            "unsupported_rows",
            "weighted_predicted_fill_probability",
            "weighted_trade_confirmed_fill_rate",
            "weighted_confirmation_shortfall",
            "weighted_cancel_only_clear_rate",
            "weighted_stale_trade_confirmed_fill_share",
            "max_confirmation_shortfall",
            "max_cancel_only_clear_rate",
            "max_stale_trade_confirmed_fill_share",
            "worst_confirmation_cell",
            "worst_confirmation_cell_rows",
            "worst_confirmation_cell_label",
            "trade_confirmation_regime_label",
            "publishable",
            "blocking_reasons",
            "review_reasons",
            "regime_priority_rank",
        ]
    )


def queue_position_trade_confirmation_regime_scorecard(
    surface: pd.DataFrame,
    *,
    min_cell_rows: int = 1,
    max_confirmation_shortfall: float = 0.20,
    max_cancel_only_clear_rate: float = 0.15,
    max_stale_trade_confirmed_fill_share: float = 0.25,
) -> pd.DataFrame:
    """Rank trade-confirmation failures by regime for execution review.

    The release scorecard compresses the full queue-position confirmation surface
    into one decision. This regime scorecard preserves where the evidence breaks:
    for each regime, it row-weights predicted vs trade-confirmed passive fills,
    cancel-only queue clearance, stale confirmations, and the worst queue-clear
    bucket so execution reviewers can target the market state that invalidates
    passive fill probabilities.
    """
    if not isinstance(min_cell_rows, int) or isinstance(min_cell_rows, bool):
        raise ValueError("min_cell_rows must be a positive integer")
    if min_cell_rows < 1:
        raise ValueError("min_cell_rows must be a positive integer")
    for name, value in {
        "max_confirmation_shortfall": max_confirmation_shortfall,
        "max_cancel_only_clear_rate": max_cancel_only_clear_rate,
        "max_stale_trade_confirmed_fill_share": max_stale_trade_confirmed_fill_share,
    }.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if surface.empty:
        return _empty_queue_position_trade_confirmation_regime_scorecard()

    required = {
        "regime",
        "queue_clear_bucket",
        "rows",
        "mean_predicted_fill_probability",
        "trade_confirmed_fill_rate",
        "cancel_only_clear_rate",
        "stale_trade_confirmed_fill_share",
        "confirmation_shortfall",
        "confirmation_surface_label",
    }
    _require_columns(surface, required, "queue position trade confirmation regime scorecard")
    numeric_columns = [
        "rows",
        "mean_predicted_fill_probability",
        "trade_confirmed_fill_rate",
        "cancel_only_clear_rate",
        "stale_trade_confirmed_fill_share",
        "confirmation_shortfall",
    ]
    values = _finite_values(surface, numeric_columns, "queue position trade confirmation regime scorecard")
    if (values["rows"] < 0.0).any():
        raise ValueError("queue position trade confirmation regime scorecard rows must be non-negative")
    rate_columns = [
        "mean_predicted_fill_probability",
        "trade_confirmed_fill_rate",
        "cancel_only_clear_rate",
        "stale_trade_confirmed_fill_share",
    ]
    if not values[rate_columns].apply(lambda column: column.between(0.0, 1.0).all()).all():
        raise ValueError("queue position trade confirmation regime scorecard rates must be in [0, 1]")

    data = values.copy()
    data["regime"] = surface["regime"].astype(str)
    data["queue_clear_bucket"] = surface["queue_clear_bucket"].astype(str)
    data["confirmation_surface_label"] = surface["confirmation_surface_label"].astype(str)
    rows: list[dict[str, float | int | str | bool]] = []
    for regime, group in data.groupby("regime", sort=True):
        regime_rows = int(group["rows"].sum())
        supported = group[group["rows"] >= float(min_cell_rows)].copy()
        if regime_rows == 0 or supported.empty:
            rows.append(
                {
                    "regime": str(regime),
                    "cells": int(len(group)),
                    "supported_cells": int(len(supported)),
                    "rows": regime_rows,
                    "supported_rows": 0,
                    "unsupported_rows": regime_rows,
                    "weighted_predicted_fill_probability": 0.0,
                    "weighted_trade_confirmed_fill_rate": 0.0,
                    "weighted_confirmation_shortfall": 0.0,
                    "weighted_cancel_only_clear_rate": 0.0,
                    "weighted_stale_trade_confirmed_fill_share": 0.0,
                    "max_confirmation_shortfall": 0.0,
                    "max_cancel_only_clear_rate": 0.0,
                    "max_stale_trade_confirmed_fill_share": 0.0,
                    "worst_confirmation_cell": f"{regime}:none",
                    "worst_confirmation_cell_rows": 0,
                    "worst_confirmation_cell_label": "none",
                    "trade_confirmation_regime_label": "review",
                    "publishable": False,
                    "blocking_reasons": "none",
                    "review_reasons": "insufficient_trade_confirmation_evidence",
                }
            )
            continue

        supported_rows = int(supported["rows"].sum())
        weights = supported["rows"] / float(supported_rows)
        max_shortfall = float(supported["confirmation_shortfall"].max())
        max_cancel = float(supported["cancel_only_clear_rate"].max())
        max_stale = float(supported["stale_trade_confirmed_fill_share"].max())
        worst = supported.sort_values(
            ["confirmation_shortfall", "cancel_only_clear_rate", "stale_trade_confirmed_fill_share", "rows"],
            ascending=[False, False, False, False],
        ).iloc[0]
        blocking_reasons: list[str] = []
        if max_shortfall > float(max_confirmation_shortfall):
            blocking_reasons.append("confirmation_shortfall")
        if max_cancel > float(max_cancel_only_clear_rate):
            blocking_reasons.append("cancel_only_queue_clearance")
        review_reasons: list[str] = []
        if max_stale > float(max_stale_trade_confirmed_fill_share):
            review_reasons.append("stale_trade_confirmed_fills")
        if supported_rows < regime_rows:
            review_reasons.append("unsupported_trade_confirmation_cells")
        label = "block" if blocking_reasons else "review" if review_reasons else "pass"
        rows.append(
            {
                "regime": str(regime),
                "cells": int(len(group)),
                "supported_cells": int(len(supported)),
                "rows": regime_rows,
                "supported_rows": supported_rows,
                "unsupported_rows": int(regime_rows - supported_rows),
                "weighted_predicted_fill_probability": float(
                    (supported["mean_predicted_fill_probability"] * weights).sum()
                ),
                "weighted_trade_confirmed_fill_rate": float(
                    (supported["trade_confirmed_fill_rate"] * weights).sum()
                ),
                "weighted_confirmation_shortfall": float(
                    (supported["confirmation_shortfall"] * weights).sum()
                ),
                "weighted_cancel_only_clear_rate": float(
                    (supported["cancel_only_clear_rate"] * weights).sum()
                ),
                "weighted_stale_trade_confirmed_fill_share": float(
                    (supported["stale_trade_confirmed_fill_share"] * weights).sum()
                ),
                "max_confirmation_shortfall": max_shortfall,
                "max_cancel_only_clear_rate": max_cancel,
                "max_stale_trade_confirmed_fill_share": max_stale,
                "worst_confirmation_cell": f"{regime}:{worst['queue_clear_bucket']}",
                "worst_confirmation_cell_rows": int(worst["rows"]),
                "worst_confirmation_cell_label": str(worst["confirmation_surface_label"]),
                "trade_confirmation_regime_label": label,
                "publishable": label == "pass",
                "blocking_reasons": ",".join(blocking_reasons) if blocking_reasons else "none",
                "review_reasons": ",".join(review_reasons) if review_reasons else "none",
            }
        )
    scorecard = pd.DataFrame(rows)
    label_rank = {"block": 0, "review": 1, "pass": 2}
    scorecard["_label_rank"] = scorecard["trade_confirmation_regime_label"].map(label_rank)
    scorecard = scorecard.sort_values(
        [
            "_label_rank",
            "max_confirmation_shortfall",
            "max_cancel_only_clear_rate",
            "max_stale_trade_confirmed_fill_share",
            "rows",
            "regime",
        ],
        ascending=[True, False, False, False, False, True],
        ignore_index=True,
    )
    scorecard["regime_priority_rank"] = np.arange(1, len(scorecard) + 1, dtype=int)
    columns = list(_empty_queue_position_trade_confirmation_regime_scorecard().columns)
    return scorecard[columns]


def _empty_queue_position_trade_confirmation_release_scorecard() -> dict[str, float | int | str | bool]:
    return {
        "evaluated_cells": 0,
        "supported_cells": 0,
        "total_rows": 0,
        "supported_rows": 0,
        "unsupported_rows": 0,
        "weighted_confirmation_shortfall": 0.0,
        "weighted_cancel_only_clear_rate": 0.0,
        "weighted_stale_trade_confirmed_fill_share": 0.0,
        "max_confirmation_shortfall": 0.0,
        "max_cancel_only_clear_rate": 0.0,
        "max_stale_trade_confirmed_fill_share": 0.0,
        "worst_confirmation_cell": "none",
        "worst_confirmation_cell_rows": 0,
        "worst_confirmation_cell_label": "none",
        "trade_confirmation_release_label": "review",
        "publishable": False,
        "blocking_reasons": "none",
        "review_reasons": "insufficient_trade_confirmation_evidence",
    }


def queue_position_trade_confirmation_release_scorecard(
    surface: pd.DataFrame,
    *,
    min_cell_rows: int = 1,
    max_confirmation_shortfall: float = 0.20,
    max_cancel_only_clear_rate: float = 0.15,
    max_stale_trade_confirmed_fill_share: float = 0.25,
) -> dict[str, float | int | str | bool]:
    """Summarize trade-confirmed queue execution evidence for release gates.

    The confirmation surface is intentionally granular. This reducer creates a compact
    pass/review/block decision that penalizes passive fill probabilities which are
    not confirmed by trades, queue clears driven by cancels, and fills that arrive
    outside the configured latency budget.
    """
    if not isinstance(min_cell_rows, int) or isinstance(min_cell_rows, bool):
        raise ValueError("min_cell_rows must be a positive integer")
    if min_cell_rows < 1:
        raise ValueError("min_cell_rows must be a positive integer")
    for name, value in {
        "max_confirmation_shortfall": max_confirmation_shortfall,
        "max_cancel_only_clear_rate": max_cancel_only_clear_rate,
        "max_stale_trade_confirmed_fill_share": max_stale_trade_confirmed_fill_share,
    }.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if surface.empty:
        return _empty_queue_position_trade_confirmation_release_scorecard()

    required = {
        "queue_clear_bucket",
        "rows",
        "mean_predicted_fill_probability",
        "trade_confirmed_fill_rate",
        "cancel_only_clear_rate",
        "stale_trade_confirmed_fill_share",
        "confirmation_shortfall",
        "confirmation_surface_label",
    }
    regime_col = "regime" if "regime" in surface.columns else None
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(surface, required, "queue position trade confirmation release scorecard")
    numeric_columns = [
        "rows",
        "mean_predicted_fill_probability",
        "trade_confirmed_fill_rate",
        "cancel_only_clear_rate",
        "stale_trade_confirmed_fill_share",
        "confirmation_shortfall",
    ]
    values = _finite_values(surface, numeric_columns, "queue position trade confirmation release scorecard")
    if (values["rows"] < 0.0).any():
        raise ValueError("queue position trade confirmation release scorecard rows must be non-negative")
    rate_columns = [
        "mean_predicted_fill_probability",
        "trade_confirmed_fill_rate",
        "cancel_only_clear_rate",
        "stale_trade_confirmed_fill_share",
    ]
    if not values[rate_columns].apply(lambda column: column.between(0.0, 1.0).all()).all():
        raise ValueError("queue position trade confirmation release scorecard rates must be in [0, 1]")

    data = values.copy()
    data["regime"] = surface[regime_col].astype(str) if regime_col is not None else "all"
    data["queue_clear_bucket"] = surface["queue_clear_bucket"].astype(str)
    data["confirmation_surface_label"] = surface["confirmation_surface_label"].astype(str)
    total_rows = int(data["rows"].sum())
    scorecard = _empty_queue_position_trade_confirmation_release_scorecard()
    scorecard.update({"evaluated_cells": int(len(surface)), "total_rows": total_rows})
    if total_rows == 0:
        return scorecard

    supported = data[data["rows"] >= float(min_cell_rows)].copy()
    if supported.empty:
        scorecard["unsupported_rows"] = total_rows
        return scorecard

    supported_rows = int(supported["rows"].sum())
    supported_weights = supported["rows"] / float(supported_rows)
    worst_idx = supported.sort_values(
        ["confirmation_shortfall", "cancel_only_clear_rate", "stale_trade_confirmed_fill_share", "rows"],
        ascending=[False, False, False, False],
    ).index[0]
    worst = supported.loc[worst_idx]
    worst_cell = f"{worst['regime']}:{worst['queue_clear_bucket']}"

    observed_max_shortfall = float(supported["confirmation_shortfall"].max())
    observed_max_cancel = float(supported["cancel_only_clear_rate"].max())
    observed_max_stale = float(supported["stale_trade_confirmed_fill_share"].max())
    blocking_reasons: list[str] = []
    if observed_max_shortfall > float(max_confirmation_shortfall):
        blocking_reasons.append("confirmation_shortfall")
    if observed_max_cancel > float(max_cancel_only_clear_rate):
        blocking_reasons.append("cancel_only_queue_clearance")
    review_reasons: list[str] = []
    if observed_max_stale > float(max_stale_trade_confirmed_fill_share):
        review_reasons.append("stale_trade_confirmed_fills")
    if supported_rows < total_rows:
        review_reasons.append("unsupported_trade_confirmation_cells")
    label = "block" if blocking_reasons else "review" if review_reasons else "pass"

    scorecard.update(
        {
            "supported_cells": int(len(supported)),
            "supported_rows": supported_rows,
            "unsupported_rows": int(total_rows - supported_rows),
            "weighted_confirmation_shortfall": float(
                (supported["confirmation_shortfall"] * supported_weights).sum()
            ),
            "weighted_cancel_only_clear_rate": float(
                (supported["cancel_only_clear_rate"] * supported_weights).sum()
            ),
            "weighted_stale_trade_confirmed_fill_share": float(
                (supported["stale_trade_confirmed_fill_share"] * supported_weights).sum()
            ),
            "max_confirmation_shortfall": observed_max_shortfall,
            "max_cancel_only_clear_rate": observed_max_cancel,
            "max_stale_trade_confirmed_fill_share": observed_max_stale,
            "worst_confirmation_cell": worst_cell,
            "worst_confirmation_cell_rows": int(worst["rows"]),
            "worst_confirmation_cell_label": str(worst["confirmation_surface_label"]),
            "trade_confirmation_release_label": label,
            "publishable": label == "pass",
            "blocking_reasons": ",".join(blocking_reasons) if blocking_reasons else "none",
            "review_reasons": ",".join(review_reasons) if review_reasons else "none",
        }
    )
    return scorecard


def _empty_queue_position_trade_confirmation_calibration_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "probability_bin",
            "rows",
            "mean_predicted_fill_probability",
            "trade_confirmed_fill_rate",
            "prompt_trade_confirmed_fill_rate",
            "late_trade_confirmed_fill_rate",
            "cancel_only_clear_rate",
            "brier_score",
            "calibration_error",
            "absolute_calibration_error",
            "trade_confirmation_calibration_label",
        ]
    )


def queue_position_trade_confirmation_calibration_curve(
    frame: pd.DataFrame,
    *,
    bins: int = 5,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    max_latency: float | None = None,
    max_absolute_calibration_error: float = 0.20,
    max_cancel_only_clear_rate: float = 0.25,
    max_late_trade_confirmed_fill_rate: float = 0.25,
) -> pd.DataFrame:
    """Calibrate selected passive fill probability against trade-confirmed fills.

    Queue-depletion fills are only publishable if selected-side probability bins
    line up with fills confirmed by trades inside the latency budget. This curve
    buckets the chosen bid/ask fill probability by side, then reports prompt trade
    confirmation, late confirmation, cancel-only queue clearance, and Brier error
    so high-confidence passive fills cannot hide behind cancel-driven depletion.
    """
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if max_latency is not None and (not math.isfinite(max_latency) or max_latency < 0.0):
        raise ValueError("max_latency must be finite and non-negative when provided")
    for name, value in {
        "max_absolute_calibration_error": max_absolute_calibration_error,
        "max_cancel_only_clear_rate": max_cancel_only_clear_rate,
        "max_late_trade_confirmed_fill_rate": max_late_trade_confirmed_fill_rate,
    }.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_trade_confirmed_fill",
        "ask_trade_confirmed_fill",
        "bid_trade_confirmed_fill_latency",
        "ask_trade_confirmed_fill_latency",
        "bid_queue_advance_without_trade",
        "ask_queue_advance_without_trade",
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position trade confirmation calibration")
    if frame.empty:
        return _empty_queue_position_trade_confirmation_calibration_curve()

    non_latency_columns = sorted(
        required
        - {side_col, "bid_trade_confirmed_fill_latency", "ask_trade_confirmed_fill_latency"}
        - ({regime_col} if regime_col is not None else set())
    )
    values = _finite_values(frame, non_latency_columns, "queue position trade confirmation calibration")
    rate_columns = [
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_trade_confirmed_fill",
        "ask_trade_confirmed_fill",
        "bid_queue_advance_without_trade",
        "ask_queue_advance_without_trade",
    ]
    if not values[rate_columns].apply(lambda column: column.between(0.0, 1.0).all()).all():
        raise ValueError("queue position trade confirmation calibration probabilities must be in [0, 1]")
    latency_values = frame[
        ["bid_trade_confirmed_fill_latency", "ask_trade_confirmed_fill_latency"]
    ].astype(float)
    observed_latency = latency_values.to_numpy(dtype=float)
    observed_latency = observed_latency[~np.isnan(observed_latency)]
    if (observed_latency < 0.0).any() or not np.isfinite(observed_latency).all():
        raise ValueError("queue position trade confirmation calibration latencies must be non-negative")

    side = frame[side_col].astype(str)
    selected = pd.DataFrame(index=frame.index)
    selected["regime"] = frame[regime_col].astype(str) if regime_col is not None else "all"
    selected["predicted_fill_probability"] = np.select(
        [side == "long", side == "short"],
        [values["bid_fill_probability"], values["ask_fill_probability"]],
        default=np.nan,
    )
    selected["trade_confirmed_fill"] = np.select(
        [side == "long", side == "short"],
        [values["bid_trade_confirmed_fill"], values["ask_trade_confirmed_fill"]],
        default=np.nan,
    )
    selected["trade_confirmed_fill_latency"] = np.select(
        [side == "long", side == "short"],
        [latency_values["bid_trade_confirmed_fill_latency"], latency_values["ask_trade_confirmed_fill_latency"]],
        default=np.nan,
    )
    selected["cancel_only_clear"] = np.select(
        [side == "long", side == "short"],
        [values["bid_queue_advance_without_trade"], values["ask_queue_advance_without_trade"]],
        default=np.nan,
    )
    selected = selected.loc[side.isin(["long", "short"])].copy()
    if selected.empty:
        return _empty_queue_position_trade_confirmation_calibration_curve()

    actual_bins = min(bins, len(selected))
    ranks = selected["predicted_fill_probability"].rank(method="first")
    selected["probability_bin"] = pd.qcut(ranks, q=actual_bins, labels=False, duplicates="drop").astype(int) + 1
    selected["prompt_trade_confirmed_fill"] = selected["trade_confirmed_fill"]
    selected["late_trade_confirmed_fill"] = 0.0
    if max_latency is not None:
        late = (selected["trade_confirmed_fill"] > 0.0) & (selected["trade_confirmed_fill_latency"] > max_latency)
        selected.loc[late, "prompt_trade_confirmed_fill"] = 0.0
        selected.loc[late, "late_trade_confirmed_fill"] = 1.0

    rows: list[dict[str, float | int | str]] = []
    for (regime, probability_bin), group in selected.groupby(["regime", "probability_bin"], sort=True):
        predicted = float(group["predicted_fill_probability"].mean())
        confirmed_rate = float(group["trade_confirmed_fill"].mean())
        prompt_rate = float(group["prompt_trade_confirmed_fill"].mean())
        late_rate = float(group["late_trade_confirmed_fill"].mean())
        cancel_rate = float(group["cancel_only_clear"].mean())
        calibration_error = float(confirmed_rate - predicted)
        absolute_error = abs(calibration_error)
        if cancel_rate > max_cancel_only_clear_rate:
            label = "cancel_only_calibration_risk"
        elif late_rate > max_late_trade_confirmed_fill_rate:
            label = "late_confirmation_calibration_risk"
        elif absolute_error > max_absolute_calibration_error:
            label = "trade_confirmation_miscalibrated"
        else:
            label = "trade_confirmation_calibration_ok"
        rows.append(
            {
                "regime": str(regime),
                "probability_bin": int(probability_bin),
                "rows": int(len(group)),
                "mean_predicted_fill_probability": predicted,
                "trade_confirmed_fill_rate": confirmed_rate,
                "prompt_trade_confirmed_fill_rate": prompt_rate,
                "late_trade_confirmed_fill_rate": late_rate,
                "cancel_only_clear_rate": cancel_rate,
                "brier_score": float(
                    ((group["predicted_fill_probability"] - group["trade_confirmed_fill"]) ** 2).mean()
                ),
                "calibration_error": calibration_error,
                "absolute_calibration_error": absolute_error,
                "trade_confirmation_calibration_label": label,
            }
        )
    return pd.DataFrame(rows)[list(_empty_queue_position_trade_confirmation_calibration_curve().columns)]


def queue_position_trade_confirmation_surface(
    frame: pd.DataFrame,
    *,
    bins: int = 5,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    max_latency: float | None = None,
) -> pd.DataFrame:
    """Audit queue-position fill probabilities against trade-confirmed fills.

    Snapshot queue depletion can overstate passive executability when the queue
    advance comes from cancels rather than aggressing trades, or when fills arrive
    after a latency budget. This surface selects the probability/fill/latency fields
    implied by ``best_execution_side``, buckets by selected queue-clear burden, and
    reports where predicted passive fill probability is not trade-confirmed.
    """
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if max_latency is not None and (not math.isfinite(max_latency) or max_latency < 0.0):
        raise ValueError("max_latency must be finite and non-negative when provided")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_trade_confirmed_fill",
        "ask_trade_confirmed_fill",
        "bid_trade_confirmed_fill_latency",
        "ask_trade_confirmed_fill_latency",
        "bid_queue_advance_without_trade",
        "ask_queue_advance_without_trade",
        "bid_queue_clear_size",
        "ask_queue_clear_size",
        "bid_queue_clear_share",
        "ask_queue_clear_share",
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position trade confirmation surface")
    if frame.empty:
        return _empty_queue_position_trade_confirmation_surface()

    numeric_columns = sorted(required - {side_col} - ({regime_col} if regime_col is not None else set()))
    values = _finite_values(
        frame,
        [
            column
            for column in numeric_columns
            if column
            not in {
                "bid_trade_confirmed_fill_latency",
                "ask_trade_confirmed_fill_latency",
            }
        ],
        "queue position trade confirmation surface",
    )
    latency_values = frame[
        ["bid_trade_confirmed_fill_latency", "ask_trade_confirmed_fill_latency"]
    ].astype(float)
    finite_latency = latency_values.to_numpy(dtype=float)
    finite_latency = finite_latency[~np.isnan(finite_latency)]
    if not np.isfinite(finite_latency).all():
        raise ValueError("queue position trade confirmation surface latencies must be finite or NaN")
    if not values[
        [
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_trade_confirmed_fill",
            "ask_trade_confirmed_fill",
        ]
    ].apply(lambda column: column.between(0.0, 1.0).all()).all():
        raise ValueError("queue position trade confirmation probabilities must be in [0, 1]")
    nonnegative_columns = [
        "bid_queue_advance_without_trade",
        "ask_queue_advance_without_trade",
        "bid_queue_clear_size",
        "ask_queue_clear_size",
        "bid_queue_clear_share",
        "ask_queue_clear_share",
    ]
    if not values[nonnegative_columns].ge(0.0).all().all():
        raise ValueError("queue position trade confirmation queue values must be non-negative")

    side = frame[side_col].astype(str)
    selected = pd.DataFrame(index=frame.index)
    selected["regime"] = frame[regime_col].astype(str) if regime_col is not None else "all"
    selected["side"] = side
    selected["predicted_fill_probability"] = np.select(
        [side == "long", side == "short"],
        [values["bid_fill_probability"], values["ask_fill_probability"]],
        default=np.nan,
    )
    selected["trade_confirmed_fill"] = np.select(
        [side == "long", side == "short"],
        [values["bid_trade_confirmed_fill"], values["ask_trade_confirmed_fill"]],
        default=np.nan,
    )
    selected["queue_advance_without_trade"] = np.select(
        [side == "long", side == "short"],
        [values["bid_queue_advance_without_trade"], values["ask_queue_advance_without_trade"]],
        default=np.nan,
    )
    selected["queue_clear_size"] = np.select(
        [side == "long", side == "short"],
        [values["bid_queue_clear_size"], values["ask_queue_clear_size"]],
        default=np.nan,
    )
    selected["queue_clear_share"] = np.select(
        [side == "long", side == "short"],
        [values["bid_queue_clear_share"], values["ask_queue_clear_share"]],
        default=np.nan,
    )
    selected["trade_confirmed_fill_latency"] = np.select(
        [side == "long", side == "short"],
        [latency_values["bid_trade_confirmed_fill_latency"], latency_values["ask_trade_confirmed_fill_latency"]],
        default=np.nan,
    )
    selected = selected.loc[side.isin(["long", "short"])].copy()
    if selected.empty:
        return _empty_queue_position_trade_confirmation_surface()
    selected["cancel_only_clear"] = (
        (selected["trade_confirmed_fill"] <= 0.0)
        & (
            selected["queue_advance_without_trade"]
            >= selected["queue_clear_size"].where(selected["queue_clear_size"] > 0.0, np.inf)
        )
    ).astype(float)
    if max_latency is None:
        selected["stale_trade_confirmed_fill"] = 0.0
    else:
        selected["stale_trade_confirmed_fill"] = (
            (selected["trade_confirmed_fill"] > 0.0)
            & (selected["trade_confirmed_fill_latency"] > max_latency)
        ).astype(float)

    actual_bins = min(bins, len(selected))
    ranks = selected["queue_clear_share"].rank(method="first")
    selected["bucket_id"] = pd.qcut(ranks, q=actual_bins, labels=False, duplicates="drop")
    actual_bins = int(selected["bucket_id"].max()) + 1
    labels = [f"q{index + 1:02d}" for index in range(actual_bins)]
    selected["queue_clear_bucket"] = selected["bucket_id"].map(lambda bucket_id: labels[int(bucket_id)])

    rows: list[dict[str, float | int | str]] = []
    for (regime, bucket), group in selected.groupby(["regime", "queue_clear_bucket"], sort=True):
        confirmed = group.loc[group["trade_confirmed_fill"] > 0.0, "trade_confirmed_fill_latency"]
        predicted = float(group["predicted_fill_probability"].mean())
        confirmed_rate = float(group["trade_confirmed_fill"].mean())
        cancel_rate = float(group["cancel_only_clear"].mean())
        stale_share = (
            float(group.loc[group["trade_confirmed_fill"] > 0.0, "stale_trade_confirmed_fill"].mean())
            if not confirmed.empty
            else 0.0
        )
        confirmation_shortfall = float(predicted - confirmed_rate)
        rows.append(
            {
                "regime": str(regime),
                "queue_clear_bucket": str(bucket),
                "rows": int(len(group)),
                "mean_queue_clear_share": float(group["queue_clear_share"].mean()),
                "mean_predicted_fill_probability": predicted,
                "trade_confirmed_fill_rate": confirmed_rate,
                "cancel_only_clear_rate": cancel_rate,
                "mean_trade_confirmed_fill_latency": float(confirmed.mean()) if not confirmed.empty else 0.0,
                "stale_trade_confirmed_fill_share": stale_share,
                "confirmation_calibration_error": float(confirmed_rate - predicted),
                "confirmation_shortfall": confirmation_shortfall,
                "confirmation_surface_label": _trade_confirmation_surface_label(
                    confirmation_shortfall=confirmation_shortfall,
                    cancel_only_clear_rate=cancel_rate,
                    trade_confirmed_fill_rate=confirmed_rate,
                    stale_trade_confirmed_fill_share=stale_share,
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_queue_position_trade_confirmation_surface().columns)]
