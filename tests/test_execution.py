import pandas as pd
import pytest

from lcri_lab.execution import (
    FillProbabilityConfig,
    add_execution_adjusted_edge,
    add_passive_fill_probabilities,
    add_event_level_realized_fill_proxy,
    add_queue_position_features,
    add_queue_position_realized_fill_proxy,
    execution_adjusted_edge_summary,
    execution_adjusted_lcri_quantile_diagnostics,
    passive_fill_calibration_curve,
    passive_fill_calibration_summary,
    passive_fill_event_regime_summary,
    passive_fill_event_lead_lag_profile,
    passive_fill_event_lead_lag_scorecard,
    passive_fill_event_lifecycle_policy_curve,
    passive_fill_event_lifecycle_scorecard,
    passive_fill_event_lifecycle_summary,
    passive_fill_event_toxicity_scorecard,
    passive_fill_event_transition_policy_curve,
    passive_fill_event_transition_scorecard,
    passive_fill_event_transition_summary,
    passive_fill_event_window_diagnostics,
    execution_publishability_release_gate,
    execution_publishability_review_packet,
    passive_fill_edge_curve,
    passive_fill_realization_horizon_sweep,
    passive_fill_threshold_policy_curve,
    queue_position_capacity_frontier,
    queue_position_capacity_stability,
    queue_position_execution_quality_gate,
    queue_position_regime_capacity_concentration,
    queue_position_regime_capacity_frontier,
    queue_position_edge_decay,
    queue_position_fill_calibration_surface,
    queue_position_fill_surface,
    queue_position_fraction_sweep,
    queue_position_regime_fraction_sweep,
)


def _book_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bid_sz_1": [100.0, 400.0],
            "ask_sz_1": [200.0, 100.0],
            "bid_sz_2": [50.0, 50.0],
            "ask_sz_2": [50.0, 50.0],
            "spread_ticks": [1.0, 3.0],
            "volatility": [0.2, 0.8],
            "replenishment_rate": [0.8, 0.2],
            "lcri": [-2.0, 2.0],
        }
    )


def test_queue_position_features_estimate_visible_queue_ahead() -> None:
    output = add_queue_position_features(_book_frame(), levels=2, queue_position_fraction=0.25)

    assert output["bid_queue_ahead"].tolist() == pytest.approx([25.0, 100.0])
    assert output["ask_queue_ahead"].tolist() == pytest.approx([50.0, 25.0])
    assert output["bid_queue_share"].tolist() == pytest.approx([25.0 / 150.0, 100.0 / 450.0])
    assert output["ask_queue_share"].tolist() == pytest.approx([50.0 / 250.0, 25.0 / 150.0])
    assert output["queue_position_imbalance"].tolist() == pytest.approx([-25.0, 75.0])


def test_queue_position_realized_fill_proxy_uses_visible_depletion_and_price_loss() -> None:
    frame = pd.DataFrame(
        {
            "bid_px_1": [100.0, 100.0, 99.0],
            "ask_px_1": [101.0, 101.0, 102.0],
            "bid_sz_1": [100.0, 60.0, 120.0],
            "ask_sz_1": [80.0, 50.0, 90.0],
            "bid_queue_ahead": [40.0, 40.0, 40.0],
            "ask_queue_ahead": [30.0, 30.0, 30.0],
        }
    )

    output = add_queue_position_realized_fill_proxy(frame)

    assert output["bid_visible_depletion"].tolist() == pytest.approx([40.0, 60.0, 0.0])
    assert output["ask_visible_depletion"].tolist() == pytest.approx([30.0, 50.0, 0.0])
    assert output["bid_queue_depletion_ratio"].tolist() == pytest.approx([1.00, 1.50, 0.0])
    assert output["ask_queue_depletion_ratio"].tolist() == pytest.approx([1.0, 5.0 / 3.0, 0.0])
    assert output["bid_realized_fill"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert output["ask_realized_fill"].tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_queue_position_realized_fill_proxy_handles_front_of_queue_without_last_row_fill() -> None:
    frame = pd.DataFrame(
        {
            "bid_px_1": [100.0, 100.0],
            "ask_px_1": [101.0, 101.0],
            "bid_sz_1": [100.0, 100.0],
            "ask_sz_1": [80.0, 80.0],
            "bid_queue_ahead": [0.0, 0.0],
            "ask_queue_ahead": [0.0, 0.0],
        }
    )

    output = add_queue_position_realized_fill_proxy(frame)

    assert output["bid_queue_depletion_ratio"].tolist() == pytest.approx([0.0, 0.0])
    assert output["ask_queue_depletion_ratio"].tolist() == pytest.approx([0.0, 0.0])
    assert output["bid_realized_fill"].tolist() == pytest.approx([0.0, 0.0])
    assert output["ask_realized_fill"].tolist() == pytest.approx([0.0, 0.0])


def test_queue_position_realized_fill_proxy_horizon_captures_later_queue_clear() -> None:
    frame = pd.DataFrame(
        {
            "bid_px_1": [100.0, 100.0, 99.0],
            "ask_px_1": [101.0, 101.0, 102.0],
            "bid_sz_1": [100.0, 85.0, 120.0],
            "ask_sz_1": [80.0, 70.0, 90.0],
            "bid_queue_ahead": [40.0, 40.0, 40.0],
            "ask_queue_ahead": [30.0, 30.0, 30.0],
        }
    )

    one_step = add_queue_position_realized_fill_proxy(frame, horizon=1)
    two_step = add_queue_position_realized_fill_proxy(frame, horizon=2)

    assert one_step["bid_visible_depletion"].tolist() == pytest.approx([15.0, 85.0, 0.0])
    assert one_step["ask_visible_depletion"].tolist() == pytest.approx([10.0, 70.0, 0.0])
    assert one_step["bid_realized_fill"].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert one_step["ask_realized_fill"].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert two_step["bid_visible_depletion"].tolist() == pytest.approx([100.0, 85.0, 0.0])
    assert two_step["ask_visible_depletion"].tolist() == pytest.approx([80.0, 70.0, 0.0])
    assert two_step["bid_realized_fill"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert two_step["ask_realized_fill"].tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_queue_position_realized_fill_proxy_respects_group_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "B"],
            "bid_px_1": [100.0, 99.0, 99.0],
            "ask_px_1": [101.0, 102.0, 102.0],
            "bid_sz_1": [100.0, 120.0, 70.0],
            "ask_sz_1": [80.0, 90.0, 45.0],
            "bid_queue_ahead": [40.0, 40.0, 40.0],
            "ask_queue_ahead": [30.0, 30.0, 30.0],
        }
    )

    output = add_queue_position_realized_fill_proxy(frame, group_cols=["symbol"])

    assert output["bid_visible_depletion"].tolist() == pytest.approx([0.0, 50.0, 0.0])
    assert output["ask_visible_depletion"].tolist() == pytest.approx([0.0, 45.0, 0.0])
    assert output["bid_realized_fill"].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert output["ask_realized_fill"].tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_queue_position_realized_fill_proxy_rejects_missing_queue_state() -> None:
    with pytest.raises(ValueError, match="missing queue position realized fill proxy columns"):
        add_queue_position_realized_fill_proxy(pd.DataFrame({"bid_px_1": [100.0]}))


def test_event_level_realized_fill_proxy_uses_trades_and_cancels_until_horizon() -> None:
    snapshots = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0],
            "bid_px_1": [100.0, 100.0],
            "ask_px_1": [101.0, 101.0],
            "bid_queue_ahead": [45.0, 20.0],
            "ask_queue_ahead": [35.0, 30.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [0.25, 0.50, 0.75, 1.40],
            "event_type": ["trade", "cancel", "trade", "trade"],
            "side": ["sell", "bid", "buy", "sell"],
            "price": [100.0, 100.0, 101.0, 100.0],
            "size": [25.0, 20.0, 35.0, 20.0],
        }
    )

    output = add_event_level_realized_fill_proxy(snapshots, events, horizon=1.0)

    assert output["bid_event_depletion"].tolist() == pytest.approx([45.0, 20.0])
    assert output["ask_event_depletion"].tolist() == pytest.approx([35.0, 0.0])
    assert output["bid_event_depletion_ratio"].tolist() == pytest.approx([1.0, 1.0])
    assert output["ask_event_depletion_ratio"].tolist() == pytest.approx([1.0, 0.0])
    assert output["bid_realized_fill"].tolist() == pytest.approx([1.0, 1.0])
    assert output["ask_realized_fill"].tolist() == pytest.approx([1.0, 0.0])


def test_event_level_realized_fill_proxy_respects_group_boundaries() -> None:
    snapshots = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": [0.0, 0.0],
            "bid_px_1": [100.0, 50.0],
            "ask_px_1": [101.0, 51.0],
            "bid_queue_ahead": [20.0, 20.0],
            "ask_queue_ahead": [20.0, 20.0],
        }
    )
    events = pd.DataFrame(
        {
            "symbol": ["B"],
            "timestamp": [0.50],
            "event_type": ["trade"],
            "side": ["sell"],
            "price": [50.0],
            "size": [25.0],
        }
    )

    output = add_event_level_realized_fill_proxy(snapshots, events, horizon=1.0, group_cols="symbol")

    assert output["bid_event_depletion"].tolist() == pytest.approx([0.0, 25.0])
    assert output["bid_realized_fill"].tolist() == pytest.approx([0.0, 1.0])
    assert output["ask_realized_fill"].tolist() == pytest.approx([0.0, 0.0])


def test_event_level_realized_fill_proxy_accepts_venue_side_aliases() -> None:
    snapshots = pd.DataFrame(
        {
            "timestamp": [0.0],
            "bid_px_1": [100.0],
            "ask_px_1": [101.0],
            "bid_queue_ahead": [10.0],
            "ask_queue_ahead": [10.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [0.25, 0.50],
            "event_type": ["execution", "remove"],
            "side": ["BID_HIT", "OFFER"],
            "price": [100.0, 101.0],
            "size": [10.0, 10.0],
        }
    )

    output = add_event_level_realized_fill_proxy(
        snapshots,
        events,
        horizon=1.0,
        trade_event_types=("execution",),
        cancel_event_types=("remove",),
        bid_trade_sides=("bid_hit",),
        ask_cancel_sides=("offer",),
    )

    assert output["bid_event_depletion"].tolist() == pytest.approx([10.0])
    assert output["ask_event_depletion"].tolist() == pytest.approx([10.0])
    assert output["bid_realized_fill"].tolist() == pytest.approx([1.0])
    assert output["ask_realized_fill"].tolist() == pytest.approx([1.0])


def test_passive_fill_probabilities_move_with_pressure_and_queue() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2, queue_position_fraction=0.50)
    output = add_passive_fill_probabilities(queued)

    assert output.loc[0, "bid_fill_probability"] > output.loc[1, "bid_fill_probability"]
    assert output.loc[1, "ask_fill_probability"] > output.loc[0, "ask_fill_probability"]
    assert output.loc[0, "bid_fill_probability"] > output.loc[0, "ask_fill_probability"]
    assert output.loc[1, "ask_fill_probability"] > output.loc[1, "bid_fill_probability"]
    assert output["passive_fill_regime"].tolist() == ["bid_depletion", "ask_depletion"]


def test_passive_fill_probabilities_bound_outputs() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2)
    output = add_passive_fill_probabilities(
        queued,
        config=FillProbabilityConfig(min_probability=0.10, max_probability=0.80),
    )

    for column in [
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    ]:
        assert output[column].between(0.10, 0.80).all()


def test_fill_probability_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="min_probability"):
        FillProbabilityConfig(min_probability=-0.1)
    with pytest.raises(ValueError, match="max_probability"):
        FillProbabilityConfig(min_probability=0.8, max_probability=0.4)
    with pytest.raises(ValueError, match="queue_position_fraction"):
        FillProbabilityConfig(queue_position_fraction=1.5)


def test_execution_adjusted_edge_selects_best_side() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [2.0, -2.0],
            "lcri_probability": [0.72, 0.28],
            "long_net_return_ticks": [2.0, -1.0],
            "short_net_return_ticks": [-1.0, 2.0],
            "bid_fill_probability": [0.60, 0.70],
            "ask_fill_probability": [0.20, 0.60],
            "bid_adverse_fill_probability": [0.10, 0.30],
            "ask_adverse_fill_probability": [0.25, 0.10],
        }
    )

    output = add_execution_adjusted_edge(frame)

    assert output["best_execution_side"].tolist() == ["long", "short"]
    assert output["execution_adjusted_edge_ticks"].tolist() == pytest.approx([1.0, 1.0])
    assert output["execution_adjusted_lcri_score"].tolist() == pytest.approx([2.0, -2.0])


def test_execution_adjusted_edge_abstains_when_both_edges_negative() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [1.5],
            "lcri_probability": [0.55],
            "long_net_return_ticks": [-0.5],
            "short_net_return_ticks": [-0.4],
            "bid_fill_probability": [0.40],
            "ask_fill_probability": [0.40],
            "bid_adverse_fill_probability": [0.50],
            "ask_adverse_fill_probability": [0.50],
        }
    )

    output = add_execution_adjusted_edge(frame)

    assert output.loc[0, "best_execution_side"] == "abstain"
    assert output.loc[0, "execution_adjusted_edge_ticks"] == pytest.approx(-0.36)
    assert output.loc[0, "execution_adjusted_lcri_score"] == 0.0


def test_execution_adjusted_edge_summary_quantifies_tradability_drag() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "short", "abstain", "long"],
            "execution_adjusted_edge_ticks": [0.80, 0.40, -0.20, 1.00],
            "long_fill_adjusted_edge_ticks": [0.80, -0.30, -0.20, 1.00],
            "short_fill_adjusted_edge_ticks": [-0.10, 0.40, -0.25, -0.50],
            "bid_fill_probability": [0.60, 0.20, 0.10, 0.80],
            "ask_fill_probability": [0.30, 0.70, 0.10, 0.40],
            "bid_adverse_fill_probability": [0.10, 0.20, 0.20, 0.30],
            "ask_adverse_fill_probability": [0.20, 0.10, 0.20, 0.40],
            "publishable_side": ["long", "short", "long", "abstain"],
        }
    )

    summary = execution_adjusted_edge_summary(frame)

    assert summary["rows"] == 4
    assert summary["tradable_rows"] == 3
    assert summary["abstain_rows"] == 1
    assert summary["tradable_share"] == pytest.approx(0.75)
    assert summary["mean_execution_adjusted_edge_ticks"] == pytest.approx(0.50)
    assert summary["median_execution_adjusted_edge_ticks"] == pytest.approx(0.60)
    assert summary["mean_bid_fill_probability"] == pytest.approx(0.425)
    assert summary["mean_ask_fill_probability"] == pytest.approx(0.375)
    assert summary["mean_adverse_fill_probability"] == pytest.approx(0.2125)
    assert summary["publishable_side_conflict_rows"] == 2
    assert summary["publishable_side_conflict_share"] == pytest.approx(0.50)
    assert summary["dominant_execution_side"] == "long"


def test_execution_adjusted_lcri_side_attribution_explains_execution_conflicts() -> None:
    from lcri_lab.execution import execution_adjusted_lcri_side_attribution

    frame = pd.DataFrame(
        {
            "lcri": [2.0, 1.5, -2.0, -1.0, 0.0],
            "lcri_probability": [0.80, 0.70, 0.25, 0.35, 0.50],
            "best_execution_side": ["long", "abstain", "long", "long", "abstain"],
            "execution_adjusted_edge_ticks": [0.8, -0.2, 0.7, 0.1, 0.0],
            "bid_fill_probability": [0.70, 0.20, 0.20, 0.60, 0.40],
            "ask_fill_probability": [0.30, 0.30, 0.80, 0.20, 0.40],
            "bid_adverse_fill_probability": [0.10, 0.50, 0.20, 0.10, 0.20],
            "ask_adverse_fill_probability": [0.20, 0.40, 0.10, 0.60, 0.20],
        }
    )

    attribution = execution_adjusted_lcri_side_attribution(frame)

    assert attribution.columns.tolist() == [
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
    assert attribution["lcri_side"].tolist() == ["long", "short", "neutral"]
    assert attribution["rows"].tolist() == [2, 2, 1]
    assert attribution["tradable_rows"].tolist() == [1, 2, 0]
    assert attribution["execution_conflict_rows"].tolist() == [1, 2, 0]
    assert attribution["execution_conflict_share"].tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert attribution["mean_signal_confidence"].tolist() == pytest.approx([0.75, 0.70, 0.50])
    assert attribution["mean_fill_probability_advantage"].tolist() == pytest.approx([0.15, 0.10, 0.0])
    assert attribution["mean_adverse_fill_probability_advantage"].tolist() == pytest.approx([0.30, 0.35, 0.0])
    assert attribution["dominant_execution_side"].tolist() == ["long", "long", "none"]
    assert attribution["review_label"].tolist() == [
        "execution_friction_review",
        "execution_side_inversion_review",
        "neutral_signal",
    ]


def test_execution_adjusted_lcri_quantile_diagnostics_measures_signal_survival() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "execution_adjusted_lcri_score": [-3.0, 0.0, -1.0, 0.0, 2.0, 3.0],
            "execution_adjusted_edge_ticks": [1.2, -0.2, 0.3, -0.1, 0.5, 1.4],
            "best_execution_side": ["short", "abstain", "short", "abstain", "long", "long"],
        }
    )

    diagnostics = execution_adjusted_lcri_quantile_diagnostics(frame, bins=3)

    assert diagnostics.columns.tolist() == [
        "bucket",
        "rows",
        "mean_abs_lcri",
        "mean_abs_execution_adjusted_lcri_score",
        "signal_survival_ratio",
        "tradable_share",
        "mean_execution_adjusted_edge_ticks",
        "edge_drag_vs_raw_abs_lcri",
    ]
    assert diagnostics["bucket"].tolist() == ["low_abs_lcri", "mid_abs_lcri", "high_abs_lcri"]
    assert diagnostics["rows"].tolist() == [2, 2, 2]
    assert diagnostics["mean_abs_lcri"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert diagnostics["mean_abs_execution_adjusted_lcri_score"].tolist() == pytest.approx(
        [0.5, 1.0, 3.0]
    )
    assert diagnostics["signal_survival_ratio"].tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert diagnostics["tradable_share"].tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert diagnostics["edge_drag_vs_raw_abs_lcri"].tolist() == pytest.approx([0.9, 1.85, 1.7])


def test_event_level_passive_fill_horizon_sweep_relabels_from_trade_and_cancel_flow() -> None:
    snapshots = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "timestamp": [0.0, 1.0, 2.0],
            "bid_px_1": [100.0, 100.0, 100.0],
            "ask_px_1": [101.0, 101.0, 101.0],
            "bid_queue_ahead": [30.0, 30.0, 30.0],
            "ask_queue_ahead": [30.0, 30.0, 30.0],
            "best_execution_side": ["long", "short", "long"],
            "regime": ["open", "open", "close"],
            "bid_fill_probability": [0.60, 0.20, 0.80],
            "ask_fill_probability": [0.30, 0.70, 0.40],
        }
    )
    events = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "A"],
            "timestamp": [0.40, 0.70, 1.80, 2.40],
            "event_type": ["trade", "cancel", "trade", "trade"],
            "side": ["sell", "bid", "buy", "sell"],
            "price": [100.0, 100.0, 101.0, 100.0],
            "size": [15.0, 15.0, 30.0, 30.0],
        }
    )

    from lcri_lab.execution import event_level_passive_fill_horizon_sweep

    sweep = event_level_passive_fill_horizon_sweep(
        snapshots,
        events,
        horizons=[0.5, 1.0],
        bins=1,
        group_cols="symbol",
        regime_col="regime",
    )

    assert sweep["horizon"].tolist() == pytest.approx([0.5, 1.0])
    assert sweep["rows"].tolist() == [3, 3]
    assert sweep["weighted_mean_predicted_fill_probability"].tolist() == pytest.approx([0.70, 0.70])
    assert sweep["weighted_realized_fill_rate"].tolist() == pytest.approx([1.0 / 3.0, 1.0])
    assert sweep["realized_fill_rate_gap_vs_shortest"].tolist() == pytest.approx([0.0, 2.0 / 3.0])
    assert sweep["event_depletion_source"].tolist() == ["events", "events"]
    assert sweep["horizon_stability_label"].tolist() == ["anchor_horizon", "later_fill_realization"]


def test_event_level_passive_fill_horizon_sweep_rejects_invalid_horizons() -> None:
    from lcri_lab.execution import event_level_passive_fill_horizon_sweep

    with pytest.raises(ValueError, match="horizons must be a non-empty sequence"):
        event_level_passive_fill_horizon_sweep(pd.DataFrame(), pd.DataFrame(), horizons=[])
    with pytest.raises(ValueError, match="horizon values must be finite positive values"):
        event_level_passive_fill_horizon_sweep(pd.DataFrame(), pd.DataFrame(), horizons=[0.0])


def test_passive_fill_realization_horizon_sweep_relabels_and_calibrates_each_horizon() -> None:
    frame = pd.DataFrame(
        {
            "bid_px_1": [100.0, 100.0, 99.0],
            "ask_px_1": [101.0, 101.0, 102.0],
            "bid_sz_1": [100.0, 85.0, 120.0],
            "ask_sz_1": [80.0, 70.0, 90.0],
            "bid_queue_ahead": [40.0, 40.0, 40.0],
            "ask_queue_ahead": [30.0, 30.0, 30.0],
            "best_execution_side": ["long", "short", "long"],
            "bid_fill_probability": [0.60, 0.20, 0.80],
            "ask_fill_probability": [0.30, 0.70, 0.40],
        }
    )

    sweep = passive_fill_realization_horizon_sweep(frame, horizons=[1, 2], bins=1)

    assert sweep["horizon"].tolist() == [1, 2]
    assert sweep["rows"].tolist() == [3, 3]
    assert sweep["weighted_mean_predicted_fill_probability"].tolist() == pytest.approx([0.70, 0.70])
    assert sweep["weighted_realized_fill_rate"].tolist() == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert sweep["realized_fill_rate_gap_vs_shortest"].tolist() == pytest.approx([0.0, 1.0 / 3.0])
    assert sweep["horizon_stability_label"].tolist() == ["anchor_horizon", "later_fill_realization"]


def test_passive_fill_realization_horizon_sweep_rejects_invalid_horizons() -> None:
    with pytest.raises(ValueError, match="horizons must be a non-empty sequence"):
        passive_fill_realization_horizon_sweep(pd.DataFrame(), horizons=[])
    with pytest.raises(ValueError, match="horizon values must be positive integers"):
        passive_fill_realization_horizon_sweep(pd.DataFrame(), horizons=[1, 0])


def test_passive_fill_edge_curve_bins_execution_quality_by_predicted_fill() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "abstain"],
            "bid_fill_probability": [0.20, 0.80, 0.10, 0.30, 0.50],
            "ask_fill_probability": [0.10, 0.20, 0.40, 0.90, 0.50],
            "bid_adverse_fill_probability": [0.05, 0.20, 0.10, 0.10, 0.30],
            "ask_adverse_fill_probability": [0.10, 0.20, 0.15, 0.25, 0.30],
            "execution_adjusted_edge_ticks": [0.10, 0.70, -0.20, 1.20, -0.10],
            "long_net_return_ticks": [0.40, 1.00, -0.50, -0.20, 0.00],
            "short_net_return_ticks": [-0.20, -0.60, -0.40, 1.50, 0.00],
        }
    )

    curve = passive_fill_edge_curve(frame, bins=2)

    assert curve["bin"].tolist() == [1, 2]
    assert curve["rows"].tolist() == [2, 2]
    assert curve["mean_predicted_fill_probability"].tolist() == pytest.approx([0.30, 0.85])
    assert curve["mean_adverse_fill_probability"].tolist() == pytest.approx([0.10, 0.225])
    assert curve["mean_realized_edge_ticks"].tolist() == pytest.approx([0.0, 1.25])
    assert curve["positive_edge_rate"].tolist() == pytest.approx([0.50, 1.00])
    assert curve["long_rows"].tolist() == [1, 1]
    assert curve["short_rows"].tolist() == [1, 1]


def test_passive_fill_edge_curve_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="bins"):
        passive_fill_edge_curve(pd.DataFrame(), bins=0)
    with pytest.raises(ValueError, match="missing passive fill edge curve columns"):
        passive_fill_edge_curve(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_fill_surface_crosses_queue_depth_with_realized_fills() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "abstain", "long"],
            "regime": ["open", "open", "open", "midday", "open", "midday"],
            "bid_queue_share": [0.10, 0.80, 0.20, 0.50, 0.40, 0.90],
            "ask_queue_share": [0.40, 0.50, 0.30, 0.85, 0.40, 0.20],
            "bid_fill_probability": [0.20, 0.90, 0.10, 0.30, 0.50, 0.80],
            "ask_fill_probability": [0.10, 0.20, 0.40, 0.95, 0.50, 0.10],
            "bid_realized_fill": [0, 1, 0, 0, 1, 1],
            "ask_realized_fill": [0, 0, 0, 1, 1, 1],
            "execution_adjusted_edge_ticks": [0.10, 0.70, -0.20, 1.20, -0.10, 0.50],
        }
    )

    surface = queue_position_fill_surface(
        frame,
        queue_bins=2,
        probability_bins=2,
        regime_col="regime",
        bid_realized_col="bid_realized_fill",
        ask_realized_col="ask_realized_fill",
    )

    assert surface["regime"].tolist() == ["midday", "midday", "open", "open"]
    assert surface["queue_bin"].tolist() == [1, 2, 1, 2]
    assert surface["fill_probability_bin"].tolist() == [2, 1, 1, 2]
    assert surface["rows"].tolist() == [1, 1, 2, 1]
    assert surface["mean_queue_share"].tolist() == pytest.approx([0.85, 0.90, 0.20, 0.80])
    assert surface["realized_fill_rate"].tolist() == pytest.approx([1.00, 1.00, 0.00, 1.00])
    assert surface["calibration_error"].tolist() == pytest.approx([0.05, 0.20, -0.30, 0.10])
    assert surface["mean_execution_adjusted_edge_ticks"].tolist() == pytest.approx([1.20, 0.50, -0.05, 0.70])


def test_queue_position_fill_surface_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="queue_bins"):
        queue_position_fill_surface(pd.DataFrame(), queue_bins=0)
    with pytest.raises(ValueError, match="probability_bins"):
        queue_position_fill_surface(pd.DataFrame(), probability_bins=0)
    with pytest.raises(ValueError, match="missing queue position fill surface columns"):
        queue_position_fill_surface(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_fraction_sweep_quantifies_quote_placement_decay() -> None:
    frame = _book_frame().assign(
        lcri_probability=[0.25, 0.75],
        long_net_return_ticks=[1.0, 1.5],
        short_net_return_ticks=[1.5, 1.0],
    )

    sweep = queue_position_fraction_sweep(
        frame,
        fractions=[0.0, 0.5, 1.0],
        levels=2,
        fill_config=FillProbabilityConfig(adverse_selection_scale=0.25),
    )

    assert sweep["queue_position_fraction"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert sweep["rows"].tolist() == [2, 2, 2]
    assert sweep["mean_bid_queue_share"].is_monotonic_increasing
    assert sweep["mean_ask_queue_share"].is_monotonic_increasing
    assert sweep["mean_bid_fill_probability"].is_monotonic_decreasing
    assert sweep["mean_ask_fill_probability"].is_monotonic_decreasing
    assert sweep["mean_execution_adjusted_edge_ticks"].iloc[0] > sweep[
        "mean_execution_adjusted_edge_ticks"
    ].iloc[-1]
    assert sweep["tradable_share"].between(0.0, 1.0).all()
    assert sweep["abstain_share"].between(0.0, 1.0).all()
    assert set(sweep["dominant_execution_side"]).issubset({"long", "short", "none"})


def test_queue_position_fraction_sweep_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="fractions"):
        queue_position_fraction_sweep(_book_frame(), fractions=[])
    with pytest.raises(ValueError, match="queue_position_fraction"):
        queue_position_fraction_sweep(_book_frame(), fractions=[-0.1])


def test_queue_position_regime_fraction_sweep_keeps_state_capacity_auditable() -> None:
    frame = _book_frame().assign(
        lcri_probability=[0.25, 0.75],
        long_net_return_ticks=[1.0, 1.5],
        short_net_return_ticks=[1.5, 1.0],
        liquidity_state=["open", "thin"],
    )

    sweep = queue_position_regime_fraction_sweep(
        frame,
        regime_col="liquidity_state",
        fractions=[0.0, 0.5],
        levels=2,
        fill_config=FillProbabilityConfig(adverse_selection_scale=0.25),
    )

    assert sweep["liquidity_state"].tolist() == ["open", "open", "thin", "thin"]
    assert sweep["queue_position_fraction"].tolist() == pytest.approx([0.0, 0.5, 0.0, 0.5])
    assert sweep["rows"].tolist() == [1, 1, 1, 1]
    assert set(sweep["dominant_execution_side"]).issubset({"long", "short", "none"})


def test_queue_position_regime_fraction_sweep_rejects_bad_regime_inputs() -> None:
    with pytest.raises(ValueError, match="regime_col"):
        queue_position_regime_fraction_sweep(pd.DataFrame(), regime_col="")
    with pytest.raises(ValueError, match="missing queue position regime fraction sweep columns"):
        queue_position_regime_fraction_sweep(pd.DataFrame({"regime": ["open"]}))


def test_queue_position_capacity_frontier_finds_deepest_viable_quote_placement() -> None:
    sweep = pd.DataFrame(
        {
            "queue_position_fraction": [0.0, 0.25, 0.50, 0.75, 1.0],
            "rows": [100, 100, 100, 100, 100],
            "mean_bid_fill_probability": [0.82, 0.74, 0.65, 0.52, 0.40],
            "mean_ask_fill_probability": [0.78, 0.70, 0.61, 0.48, 0.36],
            "mean_execution_adjusted_edge_ticks": [0.90, 0.72, 0.51, 0.18, -0.05],
            "tradable_share": [0.94, 0.88, 0.74, 0.45, 0.20],
            "abstain_share": [0.06, 0.12, 0.26, 0.55, 0.80],
            "dominant_execution_side": ["long", "long", "long", "short", "short"],
        }
    )

    frontier = queue_position_capacity_frontier(
        sweep,
        min_edge_ticks=0.50,
        min_tradable_share=0.70,
    )

    assert frontier == {
        "rows": 5,
        "viable_rows": 3,
        "front_queue_position_fraction": pytest.approx(0.0),
        "max_viable_queue_position_fraction": pytest.approx(0.50),
        "front_mean_execution_adjusted_edge_ticks": pytest.approx(0.90),
        "max_viable_mean_execution_adjusted_edge_ticks": pytest.approx(0.51),
        "edge_decay_to_capacity_ticks": pytest.approx(0.39),
        "front_tradable_share": pytest.approx(0.94),
        "max_viable_tradable_share": pytest.approx(0.74),
        "tradable_share_decay_to_capacity": pytest.approx(0.20),
        "dominant_execution_side_at_capacity": "long",
        "capacity_label": "queue_capacity_constrained",
    }


def test_queue_position_capacity_frontier_labels_no_viable_capacity() -> None:
    sweep = pd.DataFrame(
        {
            "queue_position_fraction": [0.0, 0.5],
            "rows": [10, 10],
            "mean_bid_fill_probability": [0.40, 0.30],
            "mean_ask_fill_probability": [0.35, 0.20],
            "mean_execution_adjusted_edge_ticks": [0.10, -0.10],
            "tradable_share": [0.40, 0.20],
            "abstain_share": [0.60, 0.80],
            "dominant_execution_side": ["none", "none"],
        }
    )

    frontier = queue_position_capacity_frontier(sweep, min_edge_ticks=0.50, min_tradable_share=0.70)

    assert frontier["viable_rows"] == 0
    assert frontier["max_viable_queue_position_fraction"] == pytest.approx(0.0)
    assert frontier["capacity_label"] == "no_viable_passive_capacity"


def test_queue_position_capacity_frontier_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="min_tradable_share"):
        queue_position_capacity_frontier(pd.DataFrame(), min_tradable_share=1.5)
    with pytest.raises(ValueError, match="missing queue position capacity frontier columns"):
        queue_position_capacity_frontier(pd.DataFrame({"queue_position_fraction": [0.0]}))


def test_queue_position_regime_capacity_frontier_finds_brittle_regime_capacity() -> None:
    sweep = pd.DataFrame(
        {
            "regime": ["open", "open", "open", "stress", "stress", "stress"],
            "queue_position_fraction": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "rows": [50, 50, 50, 40, 40, 40],
            "mean_execution_adjusted_edge_ticks": [0.80, 0.45, 0.20, 0.70, 0.10, -0.20],
            "tradable_share": [0.90, 0.72, 0.55, 0.82, 0.45, 0.20],
            "dominant_execution_side": ["long", "long", "long", "short", "short", "none"],
        }
    )

    frontier = queue_position_regime_capacity_frontier(
        sweep,
        min_edge_ticks=0.25,
        min_tradable_share=0.70,
    )

    assert frontier["regime"].tolist() == ["open", "stress"]
    assert frontier["rows"].tolist() == [3, 3]
    assert frontier["viable_rows"].tolist() == [2, 1]
    assert frontier["max_viable_queue_position_fraction"].tolist() == pytest.approx([0.5, 0.0])
    assert frontier["capacity_shortfall_fraction"].tolist() == pytest.approx([0.5, 1.0])
    assert frontier["capacity_brittleness_label"].tolist() == [
        "regime_capacity_partial",
        "regime_capacity_front_only",
    ]


def test_queue_position_regime_capacity_frontier_rejects_bad_sweep() -> None:
    with pytest.raises(ValueError, match="regime_col"):
        queue_position_regime_capacity_frontier(pd.DataFrame(), regime_col="")
    with pytest.raises(ValueError, match="missing queue position regime capacity frontier columns"):
        queue_position_regime_capacity_frontier(pd.DataFrame({"regime": ["open"]}))


def test_queue_position_regime_capacity_concentration_flags_state_dependency() -> None:
    frontier = pd.DataFrame(
        {
            "regime": ["open", "thin", "stress"],
            "viable_rows": [3, 1, 0],
            "max_viable_queue_position_fraction": [0.75, 0.0, 0.0],
            "max_viable_mean_execution_adjusted_edge_ticks": [0.42, 0.08, 0.0],
            "capacity_shortfall_fraction": [0.25, 1.0, 1.0],
            "capacity_brittleness_label": [
                "regime_capacity_partial",
                "regime_capacity_front_only",
                "regime_no_viable_capacity",
            ],
        }
    )

    concentration = queue_position_regime_capacity_concentration(frontier)

    assert concentration == {
        "regimes": 3,
        "viable_regimes": 2,
        "viable_regime_share": pytest.approx(2.0 / 3.0),
        "front_only_or_no_capacity_regimes": 2,
        "front_only_or_no_capacity_share": pytest.approx(2.0 / 3.0),
        "full_capacity_regimes": 0,
        "mean_max_viable_queue_position_fraction": pytest.approx(0.25),
        "median_max_viable_queue_position_fraction": pytest.approx(0.0),
        "mean_capacity_shortfall_fraction": pytest.approx(0.75),
        "worst_capacity_regime": "stress",
        "worst_capacity_brittleness_label": "regime_no_viable_capacity",
        "capacity_concentration_label": "capacity_regime_concentrated",
    }


def test_queue_position_regime_capacity_concentration_rejects_bad_frontier() -> None:
    with pytest.raises(ValueError, match="regime_col"):
        queue_position_regime_capacity_concentration(pd.DataFrame(), regime_col="")
    with pytest.raises(ValueError, match="missing queue position regime capacity concentration columns"):
        queue_position_regime_capacity_concentration(pd.DataFrame({"regime": ["open"]}))


def test_queue_position_capacity_stability_compares_research_and_heldout_frontiers() -> None:
    research = {
        "rows": 5,
        "viable_rows": 4,
        "front_queue_position_fraction": 0.0,
        "max_viable_queue_position_fraction": 0.75,
        "front_mean_execution_adjusted_edge_ticks": 0.90,
        "max_viable_mean_execution_adjusted_edge_ticks": 0.40,
        "edge_decay_to_capacity_ticks": 0.50,
        "front_tradable_share": 0.95,
        "max_viable_tradable_share": 0.80,
        "tradable_share_decay_to_capacity": 0.15,
        "dominant_execution_side_at_capacity": "long",
        "capacity_label": "queue_capacity_constrained",
    }
    heldout = {
        **research,
        "viable_rows": 3,
        "max_viable_queue_position_fraction": 0.50,
        "max_viable_mean_execution_adjusted_edge_ticks": 0.20,
        "max_viable_tradable_share": 0.62,
        "dominant_execution_side_at_capacity": "short",
    }

    stability = queue_position_capacity_stability(research, heldout)

    assert stability == {
        "research_capacity_label": "queue_capacity_constrained",
        "heldout_capacity_label": "queue_capacity_constrained",
        "capacity_fraction_gap": pytest.approx(-0.25),
        "capacity_edge_gap_ticks": pytest.approx(-0.20),
        "capacity_tradable_share_gap": pytest.approx(-0.18),
        "capacity_viable_row_gap": -1,
        "dominant_side_changed": True,
        "capacity_stability_label": "capacity_fragile",
    }


def test_queue_position_capacity_stability_labels_stable_capacity() -> None:
    research = {
        "viable_rows": 3,
        "max_viable_queue_position_fraction": 0.50,
        "max_viable_mean_execution_adjusted_edge_ticks": 0.40,
        "max_viable_tradable_share": 0.70,
        "dominant_execution_side_at_capacity": "long",
        "capacity_label": "queue_capacity_constrained",
    }
    heldout = {
        **research,
        "max_viable_queue_position_fraction": 0.50,
        "max_viable_mean_execution_adjusted_edge_ticks": 0.36,
        "max_viable_tradable_share": 0.68,
    }

    stability = queue_position_capacity_stability(research, heldout)

    assert stability["dominant_side_changed"] is False
    assert stability["capacity_stability_label"] == "capacity_stable"


def test_queue_position_capacity_stability_rejects_bad_frontiers() -> None:
    with pytest.raises(ValueError, match="missing research capacity frontier keys"):
        queue_position_capacity_stability({"capacity_label": "x"}, {})


def test_queue_position_edge_decay_quantifies_deep_queue_degradation() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["open", "open", "open", "thin", "thin"],
            "queue_bin": [1, 2, 3, 1, 2],
            "fill_probability_bin": [2, 2, 1, 1, 2],
            "rows": [10, 20, 10, 5, 5],
            "mean_queue_share": [0.10, 0.50, 0.90, 0.20, 0.80],
            "mean_predicted_fill_probability": [0.80, 0.60, 0.40, 0.30, 0.70],
            "realized_fill_rate": [0.70, 0.50, 0.20, 0.20, 0.60],
            "calibration_error": [-0.10, -0.10, -0.20, -0.10, -0.10],
            "absolute_calibration_error": [0.10, 0.10, 0.20, 0.10, 0.10],
            "brier_score": [0.10, 0.12, 0.30, 0.20, 0.15],
            "mean_execution_adjusted_edge_ticks": [1.20, 0.50, -0.10, 0.10, 0.40],
        }
    )

    decay = queue_position_edge_decay(surface)

    open_row = decay.set_index("regime").loc["open"]
    assert open_row["queue_bins"] == 3
    assert open_row["rows"] == 40
    assert open_row["front_mean_queue_share"] == pytest.approx(0.10)
    assert open_row["back_mean_queue_share"] == pytest.approx(0.90)
    assert open_row["fill_rate_decay"] == pytest.approx(0.50)
    assert open_row["predicted_fill_decay"] == pytest.approx(0.40)
    assert open_row["edge_decay_ticks"] == pytest.approx(1.30)
    assert open_row["calibration_error_widening"] == pytest.approx(0.10)
    assert bool(open_row["monotonic_edge_decay"]) is True
    assert open_row["worst_queue_bin"] == 3
    assert open_row["queue_decay_label"] == "front_queue_preferred"

    thin_row = decay.set_index("regime").loc["thin"]
    assert thin_row["queue_decay_label"] == "deep_queue_resilient"


def test_queue_position_edge_decay_rejects_bad_surface() -> None:
    with pytest.raises(ValueError, match="min_rows"):
        queue_position_edge_decay(pd.DataFrame(), min_rows=0)
    with pytest.raises(ValueError, match="missing queue position edge decay columns"):
        queue_position_edge_decay(pd.DataFrame({"regime": ["open"]}))


def test_queue_position_execution_quality_gate_blocks_fragile_queue_surfaces() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["open", "open", "stress", "stress"],
            "queue_bin": [1, 2, 1, 2],
            "rows": [20, 20, 10, 10],
            "absolute_calibration_error": [0.05, 0.08, 0.20, 0.40],
            "brier_score": [0.04, 0.07, 0.20, 0.50],
        }
    )
    decay = pd.DataFrame(
        {
            "regime": ["open", "stress"],
            "rows": [40, 20],
            "edge_decay_ticks": [0.30, -0.20],
            "fill_rate_decay": [0.20, -0.10],
            "calibration_error_widening": [0.03, 0.20],
            "monotonic_edge_decay": [True, False],
            "queue_decay_label": ["front_queue_preferred", "calibration_watch"],
        }
    )

    gate = queue_position_execution_quality_gate(
        surface,
        decay,
        max_expected_calibration_error=0.20,
        max_expected_brier_score=0.20,
        max_calibration_widening=0.10,
    )

    assert gate == {
        "surface_rows": 60,
        "decay_rows": 60,
        "surface_regimes": 2,
        "decay_regimes": 2,
        "eligible_regimes": 2,
        "blocked_regimes": 1,
        "weighted_absolute_calibration_error": pytest.approx(0.1433333333),
        "weighted_brier_score": pytest.approx(0.1533333333),
        "weighted_edge_decay_ticks": pytest.approx(0.1333333333),
        "worst_calibration_regime": "stress",
        "worst_decay_regime": "stress",
        "max_regime_absolute_calibration_error": pytest.approx(0.30),
        "max_calibration_error_widening": pytest.approx(0.20),
        "non_monotonic_decay_regimes": 1,
        "quality_gate_label": "queue_execution_blocked",
    }


def test_queue_position_execution_quality_gate_labels_publishable_surface() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["open", "open"],
            "queue_bin": [1, 2],
            "rows": [50, 50],
            "absolute_calibration_error": [0.04, 0.06],
            "brier_score": [0.03, 0.05],
        }
    )
    decay = pd.DataFrame(
        {
            "regime": ["open"],
            "rows": [100],
            "edge_decay_ticks": [0.20],
            "fill_rate_decay": [0.10],
            "calibration_error_widening": [0.02],
            "monotonic_edge_decay": [True],
            "queue_decay_label": ["front_queue_preferred"],
        }
    )

    gate = queue_position_execution_quality_gate(surface, decay)

    assert gate["blocked_regimes"] == 0
    assert gate["quality_gate_label"] == "queue_execution_publishable"


def test_passive_fill_calibration_curve_scores_realized_side_fills_by_regime() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "abstain", "long"],
            "regime": ["thin", "thin", "thin", "stressed", "thin", "stressed"],
            "bid_fill_probability": [0.20, 0.80, 0.10, 0.30, 0.50, 0.90],
            "ask_fill_probability": [0.10, 0.20, 0.40, 0.90, 0.50, 0.10],
            "bid_realized_fill": [0, 1, 0, 0, 1, 1],
            "ask_realized_fill": [0, 0, 0, 1, 1, 1],
        }
    )

    curve = passive_fill_calibration_curve(
        frame,
        bins=2,
        regime_col="regime",
        bid_realized_col="bid_realized_fill",
        ask_realized_col="ask_realized_fill",
    )

    assert curve["regime"].tolist() == ["stressed", "stressed", "thin", "thin"]
    assert curve["bin"].tolist() == [1, 2, 1, 2]
    assert curve["rows"].tolist() == [1, 1, 2, 1]
    assert curve["realized_fill_rate"].tolist() == pytest.approx([1.00, 1.00, 0.00, 1.00])
    assert curve["mean_predicted_fill_probability"].tolist() == pytest.approx([0.90, 0.90, 0.30, 0.80])
    assert curve["calibration_error"].tolist() == pytest.approx([0.10, 0.10, -0.30, 0.20])
    assert curve["brier_score"].tolist() == pytest.approx([0.01, 0.01, 0.10, 0.04])


def test_passive_fill_calibration_summary_exposes_weighted_fill_error() -> None:
    curve = pd.DataFrame(
        {
            "regime": ["thin", "thin", "stressed"],
            "bin": [1, 2, 1],
            "rows": [2, 3, 5],
            "mean_predicted_fill_probability": [0.30, 0.80, 0.60],
            "realized_fill_rate": [0.00, 1.00, 0.40],
            "calibration_error": [-0.30, 0.20, -0.20],
            "absolute_calibration_error": [0.30, 0.20, 0.20],
            "brier_score": [0.10, 0.04, 0.24],
        }
    )

    summary = passive_fill_calibration_summary(curve)

    assert summary == {
        "rows": 10,
        "bins": 3,
        "regimes": 2,
        "weighted_mean_predicted_fill_probability": pytest.approx(0.60),
        "weighted_realized_fill_rate": pytest.approx(0.50),
        "weighted_calibration_error": pytest.approx(-0.10),
        "expected_calibration_error": pytest.approx(0.22),
        "weighted_brier_score": pytest.approx(0.152),
        "worst_regime": "thin",
        "worst_absolute_calibration_error": pytest.approx(0.30),
    }


def test_queue_position_fill_calibration_surface_bins_by_side_queue_depth() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "abstain"],
            "bid_queue_share": [0.10, 0.70, 0.20, 0.80, 0.30],
            "ask_queue_share": [0.90, 0.40, 0.25, 0.75, 0.10],
            "bid_fill_probability": [0.80, 0.30, 0.40, 0.20, 0.90],
            "ask_fill_probability": [0.10, 0.20, 0.85, 0.35, 0.10],
            "bid_realized_fill": [1.0, 0.0, 1.0, 0.0, 1.0],
            "ask_realized_fill": [0.0, 1.0, 1.0, 0.0, 0.0],
            "execution_adjusted_edge_ticks": [0.50, 0.10, 0.80, -0.20, 0.00],
            "regime": ["stable", "thin", "stable", "thin", "stable"],
        }
    )

    surface = queue_position_fill_calibration_surface(frame, queue_bins=2, probability_bins=2)

    assert surface["best_execution_side"].tolist() == ["long", "long", "short", "short"]
    assert surface["queue_share_bin"].tolist() == [1, 2, 1, 2]
    assert surface["fill_probability_bin"].tolist() == [2, 1, 2, 1]
    assert surface["rows"].tolist() == [1, 1, 1, 1]
    assert surface["mean_queue_share"].tolist() == pytest.approx([0.10, 0.70, 0.25, 0.75])
    assert surface["mean_predicted_fill_probability"].tolist() == pytest.approx([0.80, 0.30, 0.85, 0.35])
    assert surface["realized_fill_rate"].tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0])
    assert surface["calibration_error"].tolist() == pytest.approx([0.20, -0.30, 0.15, -0.35])
    assert surface["mean_execution_adjusted_edge_ticks"].tolist() == pytest.approx([0.50, 0.10, 0.80, -0.20])


def test_queue_position_fill_calibration_surface_splits_optional_regimes() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "long", "short"],
            "bid_queue_share": [0.10, 0.20, 0.80, 0.10],
            "ask_queue_share": [0.10, 0.20, 0.80, 0.40],
            "bid_fill_probability": [0.80, 0.70, 0.30, 0.20],
            "ask_fill_probability": [0.20, 0.20, 0.10, 0.60],
            "bid_realized_fill": [1.0, 1.0, 0.0, 0.0],
            "ask_realized_fill": [0.0, 0.0, 0.0, 1.0],
            "regime": ["stable", "stable", "thin", "thin"],
        }
    )

    surface = queue_position_fill_calibration_surface(
        frame,
        queue_bins=1,
        probability_bins=1,
        regime_col="regime",
    )

    assert surface["regime"].tolist() == ["stable", "thin", "thin"]
    assert surface["best_execution_side"].tolist() == ["long", "long", "short"]
    assert surface["rows"].tolist() == [2, 1, 1]
    assert surface.loc[0, "realized_fill_rate"] == pytest.approx(1.0)
    assert surface.loc[0, "mean_predicted_fill_probability"] == pytest.approx(0.75)


def test_queue_position_fill_calibration_surface_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="queue_bins"):
        queue_position_fill_calibration_surface(pd.DataFrame(), queue_bins=0)
    with pytest.raises(ValueError, match="missing queue position fill calibration surface columns"):
        queue_position_fill_calibration_surface(pd.DataFrame({"best_execution_side": ["long"]}))


def test_passive_fill_event_window_diagnostics_tracks_side_specific_drift() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "abstain"],
            "regime": ["thin", "thin", "stressed", "stressed", "thin"],
            "bid_fill_probability": [0.20, 0.90, 0.40, 0.30, 0.95],
            "ask_fill_probability": [0.10, 0.20, 0.85, 0.88, 0.95],
            "bid_adverse_fill_probability": [0.05, 0.30, 0.20, 0.20, 0.10],
            "ask_adverse_fill_probability": [0.10, 0.20, 0.35, 0.40, 0.10],
            "execution_adjusted_edge_ticks": [0.10, 0.70, 0.90, 1.10, -0.20],
            "long_net_return_ticks": [0.20, 0.40, -0.10, -0.30, 0.00],
            "short_net_return_ticks": [-0.20, -0.60, 1.00, -0.20, 0.00],
        }
    )

    events = passive_fill_event_window_diagnostics(
        frame,
        threshold=0.80,
        window=1,
        regime_col="regime",
    )

    assert events["event_index"].tolist() == [1, 2, 3]
    assert events["event_side"].tolist() == ["long", "short", "short"]
    assert events["event_regime"].tolist() == ["thin", "stressed", "stressed"]
    assert events["event_fill_probability"].tolist() == pytest.approx([0.90, 0.85, 0.88])
    assert events["event_adverse_fill_probability"].tolist() == pytest.approx([0.30, 0.35, 0.40])
    assert events["pre_realized_edge_sum"].tolist() == pytest.approx([0.20, -0.60, 1.00])
    assert events["post_realized_edge_sum"].tolist() == pytest.approx([-0.10, -0.20, 0.00])
    assert events["post_minus_pre_realized_edge"].tolist() == pytest.approx([-0.30, 0.40, -1.00])


def test_passive_fill_event_lead_lag_profile_tracks_offset_toxicity_by_regime() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "abstain"],
            "regime": ["thin", "thin", "stressed", "stressed", "thin"],
            "bid_fill_probability": [0.20, 0.90, 0.40, 0.30, 0.95],
            "ask_fill_probability": [0.10, 0.20, 0.85, 0.88, 0.95],
            "bid_adverse_fill_probability": [0.05, 0.30, 0.20, 0.20, 0.10],
            "ask_adverse_fill_probability": [0.10, 0.20, 0.35, 0.40, 0.10],
            "execution_adjusted_edge_ticks": [0.10, 0.70, 0.90, 1.10, -0.20],
            "long_net_return_ticks": [0.20, 0.40, -0.10, -0.30, 0.00],
            "short_net_return_ticks": [-0.20, -0.60, 1.00, -0.20, 0.00],
        }
    )

    profile = passive_fill_event_lead_lag_profile(
        frame,
        threshold=0.80,
        window=1,
        regime_col="regime",
    )

    assert profile.columns.tolist() == [
        "event_regime",
        "relative_offset",
        "observations",
        "mean_realized_edge_ticks",
        "adverse_realized_edge_share",
        "cumulative_mean_realized_edge_ticks",
    ]
    assert profile["event_regime"].tolist() == [
        "stressed",
        "stressed",
        "stressed",
        "thin",
        "thin",
        "thin",
    ]
    assert profile["relative_offset"].tolist() == [-1, 0, 1, -1, 0, 1]
    assert profile["observations"].tolist() == [2, 2, 2, 1, 1, 1]
    assert profile["mean_realized_edge_ticks"].tolist() == pytest.approx([0.20, 0.40, -0.10, 0.20, 0.40, -0.10])
    assert profile["adverse_realized_edge_share"].tolist() == pytest.approx([0.50, 0.50, 0.50, 0.00, 0.00, 1.00])
    assert profile["cumulative_mean_realized_edge_ticks"].tolist() == pytest.approx([0.20, 0.60, 0.50, 0.20, 0.60, 0.50])


def test_passive_fill_event_lead_lag_scorecard_flags_toxic_reversal_regimes() -> None:
    profile = pd.DataFrame(
        {
            "event_regime": ["thin", "thin", "thin", "calm", "calm", "calm"],
            "relative_offset": [-1, 0, 1, -1, 0, 1],
            "observations": [8, 10, 9, 7, 7, 7],
            "mean_realized_edge_ticks": [0.40, 0.10, -0.70, 0.20, 0.30, 0.10],
            "adverse_realized_edge_share": [0.25, 0.40, 0.80, 0.10, 0.20, 0.30],
            "cumulative_mean_realized_edge_ticks": [0.40, 0.50, -0.20, 0.20, 0.50, 0.60],
        }
    )

    scorecard = passive_fill_event_lead_lag_scorecard(profile)

    assert scorecard.columns.tolist() == [
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
    assert scorecard["event_regime"].tolist() == ["thin", "calm"]
    assert scorecard["offset_observations"].tolist() == [27, 21]
    assert scorecard["min_offset_observations"].tolist() == [8, 7]
    assert scorecard["pre_cumulative_mean_edge_ticks"].tolist() == pytest.approx([0.40, 0.20])
    assert scorecard["event_mean_edge_ticks"].tolist() == pytest.approx([0.10, 0.30])
    assert scorecard["post_cumulative_mean_edge_ticks"].tolist() == pytest.approx([-0.70, 0.10])
    assert scorecard["post_adverse_realized_edge_share"].tolist() == pytest.approx([0.80, 0.30])
    assert scorecard["lead_lag_decay_ticks"].tolist() == pytest.approx([-1.10, -0.10])
    assert scorecard["toxicity_inversion"].tolist() == [True, False]
    assert scorecard["warning_label"].tolist() == ["toxic_reversal", "edge_persistent"]


def test_passive_fill_event_lead_lag_scorecard_rejects_bad_profile() -> None:
    with pytest.raises(ValueError, match="missing passive fill event lead lag scorecard columns"):
        passive_fill_event_lead_lag_scorecard(pd.DataFrame({"event_regime": ["thin"]}))


def test_passive_fill_event_window_diagnostics_labels_regime_transitions() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "long", "short"],
            "regime": ["calm", "calm", "thin", "stress", "stress", "stress"],
            "bid_fill_probability": [0.10, 0.91, 0.20, 0.10, 0.92, 0.10],
            "ask_fill_probability": [0.10, 0.20, 0.93, 0.92, 0.20, 0.10],
            "bid_adverse_fill_probability": [0.05, 0.30, 0.20, 0.20, 0.35, 0.10],
            "ask_adverse_fill_probability": [0.10, 0.20, 0.40, 0.45, 0.10, 0.10],
            "execution_adjusted_edge_ticks": [0.10, 0.80, 0.70, 0.90, 0.50, 0.20],
            "long_net_return_ticks": [0.20, 0.30, -0.20, -0.10, 0.40, 0.10],
            "short_net_return_ticks": [-0.10, -0.20, 0.50, -0.50, -0.20, 0.10],
        }
    )

    events = passive_fill_event_window_diagnostics(
        frame,
        threshold=0.90,
        window=1,
        regime_col="regime",
    )

    assert events["event_index"].tolist() == [1, 2, 3, 4]
    assert events["pre_window_regime"].tolist() == ["calm", "calm", "thin", "stress"]
    assert events["post_window_regime"].tolist() == ["thin", "stress", "stress", "stress"]
    assert events["regime_transition"].tolist() == [
        "calm->thin",
        "calm->stress",
        "thin->stress",
        "stress->stress",
    ]


def test_passive_fill_event_transition_summary_ranks_transition_toxicity() -> None:
    events = pd.DataFrame(
        {
            "event_regime": ["thin", "thin", "stress", "stress"],
            "regime_transition": ["calm->thin", "calm->thin", "thin->stress", "stress->stress"],
            "event_fill_probability": [0.90, 0.80, 0.95, 0.85],
            "event_adverse_fill_probability": [0.30, 0.20, 0.60, 0.10],
            "event_edge_ticks": [0.70, 0.50, 1.20, 0.40],
            "post_minus_pre_realized_edge": [-0.40, -0.10, -1.00, 0.20],
        }
    )

    summary = passive_fill_event_transition_summary(events)

    assert summary["regime_transition"].tolist() == ["thin->stress", "calm->thin", "stress->stress"]
    assert summary["events"].tolist() == [1, 2, 1]
    assert summary["adverse_post_edge_share"].tolist() == pytest.approx([1.00, 1.00, 0.00])
    assert summary["mean_event_adverse_fill_probability"].tolist() == pytest.approx([0.60, 0.25, 0.10])
    assert summary["mean_post_minus_pre_realized_edge"].tolist() == pytest.approx([-1.00, -0.25, 0.20])


def test_passive_fill_event_transition_summary_rejects_bad_events() -> None:
    with pytest.raises(ValueError, match="missing passive fill event transition summary columns"):
        passive_fill_event_transition_summary(pd.DataFrame({"event_regime": ["thin"]}))


def test_passive_fill_event_lifecycle_summary_ranks_full_regime_paths() -> None:
    events = pd.DataFrame(
        {
            "pre_window_regime": ["calm", "calm", "thin", "stress"],
            "event_regime": ["thin", "thin", "stress", "stress"],
            "post_window_regime": ["stress", "stress", "stress", "calm"],
            "regime_transition": ["calm->stress", "calm->stress", "thin->stress", "stress->calm"],
            "event_fill_probability": [0.90, 0.80, 0.95, 0.85],
            "event_adverse_fill_probability": [0.30, 0.20, 0.60, 0.10],
            "event_edge_ticks": [0.70, 0.50, 1.20, 0.40],
            "pre_realized_edge_sum": [0.40, 0.10, 0.20, -0.10],
            "post_realized_edge_sum": [-0.40, -0.20, -0.90, 0.30],
            "post_minus_pre_realized_edge": [-0.80, -0.30, -1.10, 0.40],
            "window_rows": [3, 3, 3, 3],
        }
    )

    summary = passive_fill_event_lifecycle_summary(events)

    assert summary["lifecycle_path"].tolist() == [
        "thin|stress|stress",
        "calm|thin|stress",
        "stress|stress|calm",
    ]
    assert summary["events"].tolist() == [1, 2, 1]
    assert summary["adverse_post_edge_share"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert summary["mean_pre_realized_edge_sum"].tolist() == pytest.approx([0.20, 0.25, -0.10])
    assert summary["mean_post_realized_edge_sum"].tolist() == pytest.approx([-0.90, -0.30, 0.30])
    assert summary["mean_post_minus_pre_realized_edge"].tolist() == pytest.approx([-1.10, -0.55, 0.40])
    assert summary["lifecycle_toxicity_label"].tolist() == [
        "toxic_transition_lifecycle",
        "toxic_transition_lifecycle",
        "benign_transition_lifecycle",
    ]


def test_passive_fill_event_lifecycle_summary_rejects_bad_events() -> None:
    with pytest.raises(ValueError, match="missing passive fill event lifecycle summary columns"):
        passive_fill_event_lifecycle_summary(pd.DataFrame({"event_regime": ["thin"]}))


def test_passive_fill_event_lifecycle_policy_curve_sweeps_full_regime_paths() -> None:
    events = pd.DataFrame(
        {
            "pre_window_regime": ["calm", "calm", "thin", "thin", "stress"],
            "event_regime": ["thin", "thin", "stress", "stress", "stress"],
            "post_window_regime": ["stress", "stress", "stress", "stress", "calm"],
            "event_fill_probability": [0.70, 0.92, 0.82, 0.95, 0.86],
            "event_adverse_fill_probability": [0.20, 0.30, 0.60, 0.55, 0.15],
            "event_edge_ticks": [0.40, 0.80, 1.10, 1.20, 0.30],
            "pre_realized_edge_sum": [0.10, 0.20, 0.30, 0.10, -0.10],
            "post_realized_edge_sum": [0.30, 0.10, -0.50, -0.30, 0.20],
            "post_minus_pre_realized_edge": [0.20, -0.10, -0.80, -0.40, 0.30],
        }
    )

    curve = passive_fill_event_lifecycle_policy_curve(
        events,
        thresholds=(0.80, 0.90),
        max_adverse_post_edge_share=0.75,
        min_mean_post_minus_pre_edge=-0.30,
    )

    assert curve["lifecycle_path"].tolist() == [
        "calm|thin|stress",
        "calm|thin|stress",
        "stress|stress|calm",
        "stress|stress|calm",
        "thin|stress|stress",
        "thin|stress|stress",
    ]
    assert curve["threshold"].tolist() == pytest.approx([0.80, 0.90, 0.80, 0.90, 0.80, 0.90])
    assert curve["candidate_events"].tolist() == [1, 1, 1, 0, 2, 1]
    assert curve["event_share"].tolist() == pytest.approx([0.50, 0.50, 1.00, 0.00, 1.00, 0.50])
    assert curve["mean_pre_realized_edge_sum"].tolist() == pytest.approx([0.20, 0.20, -0.10, 0.0, 0.20, 0.10])
    assert curve["mean_post_realized_edge_sum"].tolist() == pytest.approx([0.10, 0.10, 0.20, 0.0, -0.40, -0.30])
    assert curve["mean_post_minus_pre_realized_edge"].tolist() == pytest.approx([-0.10, -0.10, 0.30, 0.0, -0.60, -0.40])
    assert curve["policy_label"].tolist() == [
        "lifecycle_policy_review",
        "lifecycle_policy_review",
        "selective_lifecycle_policy",
        "no_lifecycle_policy_events",
        "lifecycle_policy_blocked",
        "lifecycle_policy_blocked",
    ]


def test_passive_fill_event_lifecycle_policy_curve_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="threshold values must be in"):
        passive_fill_event_lifecycle_policy_curve(pd.DataFrame(), thresholds=(-0.10,))
    with pytest.raises(ValueError, match="missing passive fill event lifecycle policy columns"):
        passive_fill_event_lifecycle_policy_curve(pd.DataFrame({"event_regime": ["thin"]}))


def test_passive_fill_event_transition_policy_curve_sweeps_fill_cutoffs_by_transition() -> None:
    events = pd.DataFrame(
        {
            "regime_transition": ["calm->thin", "calm->thin", "thin->stress", "thin->stress"],
            "event_fill_probability": [0.70, 0.92, 0.82, 0.95],
            "event_adverse_fill_probability": [0.20, 0.30, 0.60, 0.55],
            "event_edge_ticks": [0.40, 0.80, 1.10, 1.20],
            "post_minus_pre_realized_edge": [0.20, -0.10, -0.80, -0.40],
        }
    )

    curve = passive_fill_event_transition_policy_curve(
        events,
        thresholds=(0.80, 0.90),
        max_adverse_post_edge_share=0.75,
        min_mean_post_minus_pre_edge=-0.30,
    )

    assert curve["regime_transition"].tolist() == [
        "calm->thin",
        "calm->thin",
        "thin->stress",
        "thin->stress",
    ]
    assert curve["threshold"].tolist() == pytest.approx([0.80, 0.90, 0.80, 0.90])
    assert curve["candidate_events"].tolist() == [1, 1, 2, 1]
    assert curve["event_share"].tolist() == pytest.approx([0.50, 0.50, 1.00, 0.50])
    assert curve["adverse_post_edge_share"].tolist() == pytest.approx([1.00, 1.00, 1.00, 1.00])
    assert curve["mean_post_minus_pre_realized_edge"].tolist() == pytest.approx([-0.10, -0.10, -0.60, -0.40])
    assert curve["policy_label"].tolist() == [
        "transition_policy_review",
        "transition_policy_review",
        "transition_policy_blocked",
        "transition_policy_blocked",
    ]


def test_passive_fill_event_transition_policy_curve_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="threshold values must be in"):
        passive_fill_event_transition_policy_curve(pd.DataFrame(), thresholds=(1.25,))
    with pytest.raises(ValueError, match="missing passive fill event transition policy columns"):
        passive_fill_event_transition_policy_curve(pd.DataFrame({"regime_transition": ["thin->stress"]}))


def test_passive_fill_event_transition_scorecard_blocks_toxic_regime_paths() -> None:
    summary = pd.DataFrame(
        {
            "regime_transition": ["calm->thin", "thin->stress", "stress->stress"],
            "event_regimes": [1, 1, 1],
            "events": [4, 3, 2],
            "adverse_post_edge_events": [2, 3, 0],
            "adverse_post_edge_share": [0.50, 1.00, 0.00],
            "mean_event_fill_probability": [0.82, 0.93, 0.78],
            "mean_event_adverse_fill_probability": [0.22, 0.55, 0.12],
            "mean_event_edge_ticks": [0.40, 0.90, 0.30],
            "mean_post_minus_pre_realized_edge": [-0.15, -0.80, 0.20],
            "worst_post_minus_pre_realized_edge": [-0.60, -1.50, 0.05],
        }
    )

    scorecard = passive_fill_event_transition_scorecard(
        summary,
        max_adverse_post_edge_share=0.75,
        min_mean_post_minus_pre_edge=-0.30,
    )

    assert scorecard == {
        "rows": 3,
        "transitions": 3,
        "total_events": 9,
        "eligible_transitions": 3,
        "blocked_transitions": 1,
        "worst_transition": "thin->stress",
        "worst_adverse_post_edge_share": pytest.approx(1.00),
        "worst_mean_post_minus_pre_realized_edge": pytest.approx(-0.80),
        "worst_post_minus_pre_realized_edge": pytest.approx(-1.50),
        "weighted_mean_event_fill_probability": pytest.approx(0.8477777778),
        "weighted_mean_event_adverse_fill_probability": pytest.approx(0.3077777778),
        "weighted_mean_post_minus_pre_realized_edge": pytest.approx(-0.2888888889),
        "transition_toxicity_label": "transition_event_window_blocker",
    }


def test_passive_fill_event_transition_scorecard_labels_pass_and_thin_samples() -> None:
    summary = pd.DataFrame(
        {
            "regime_transition": ["stable->stable"],
            "event_regimes": [1],
            "events": [2],
            "adverse_post_edge_events": [0],
            "adverse_post_edge_share": [0.0],
            "mean_event_fill_probability": [0.70],
            "mean_event_adverse_fill_probability": [0.10],
            "mean_event_edge_ticks": [0.30],
            "mean_post_minus_pre_realized_edge": [0.20],
            "worst_post_minus_pre_realized_edge": [0.10],
        }
    )

    assert (
        passive_fill_event_transition_scorecard(summary)["transition_toxicity_label"]
        == "transition_event_window_pass"
    )
    assert (
        passive_fill_event_transition_scorecard(summary, min_events=3)["transition_toxicity_label"]
        == "insufficient_transition_event_windows"
    )


def test_passive_fill_event_transition_scorecard_rejects_bad_summary() -> None:
    with pytest.raises(ValueError, match="max_adverse_post_edge_share"):
        passive_fill_event_transition_scorecard(pd.DataFrame(), max_adverse_post_edge_share=1.5)
    with pytest.raises(ValueError, match="missing passive fill event transition toxicity columns"):
        passive_fill_event_transition_scorecard(pd.DataFrame({"regime_transition": ["thin->stress"]}))


def test_passive_fill_event_lifecycle_scorecard_blocks_toxic_full_paths() -> None:
    summary = pd.DataFrame(
        {
            "lifecycle_path": ["calm|thin|thin", "thin|stress|stress", "stress|stress|calm"],
            "events": [4, 3, 2],
            "adverse_post_edge_share": [0.50, 1.00, 0.00],
            "mean_event_fill_probability": [0.82, 0.93, 0.78],
            "mean_event_adverse_fill_probability": [0.22, 0.55, 0.12],
            "mean_post_minus_pre_realized_edge": [-0.15, -0.80, 0.20],
            "worst_post_minus_pre_realized_edge": [-0.60, -1.50, 0.05],
        }
    )

    scorecard = passive_fill_event_lifecycle_scorecard(
        summary,
        max_adverse_post_edge_share=0.75,
        min_mean_post_minus_pre_edge=-0.30,
    )

    assert scorecard == {
        "rows": 3,
        "lifecycle_paths": 3,
        "total_events": 9,
        "eligible_lifecycle_paths": 3,
        "blocked_lifecycle_paths": 1,
        "worst_lifecycle_path": "thin|stress|stress",
        "worst_adverse_post_edge_share": pytest.approx(1.00),
        "worst_mean_post_minus_pre_realized_edge": pytest.approx(-0.80),
        "worst_post_minus_pre_realized_edge": pytest.approx(-1.50),
        "weighted_mean_event_fill_probability": pytest.approx(0.8477777778),
        "weighted_mean_event_adverse_fill_probability": pytest.approx(0.3077777778),
        "weighted_mean_post_minus_pre_realized_edge": pytest.approx(-0.2888888889),
        "lifecycle_toxicity_gate_label": "lifecycle_event_window_blocker",
    }


def test_passive_fill_event_lifecycle_scorecard_labels_pass_and_thin_samples() -> None:
    summary = pd.DataFrame(
        {
            "lifecycle_path": ["stable|stable|stable"],
            "events": [2],
            "adverse_post_edge_share": [0.0],
            "mean_event_fill_probability": [0.70],
            "mean_event_adverse_fill_probability": [0.10],
            "mean_post_minus_pre_realized_edge": [0.20],
            "worst_post_minus_pre_realized_edge": [0.10],
        }
    )

    assert (
        passive_fill_event_lifecycle_scorecard(summary)["lifecycle_toxicity_gate_label"]
        == "lifecycle_event_window_pass"
    )
    assert (
        passive_fill_event_lifecycle_scorecard(summary, min_events=3)[
            "lifecycle_toxicity_gate_label"
        ]
        == "insufficient_lifecycle_event_windows"
    )


def test_passive_fill_event_lifecycle_scorecard_rejects_bad_summary() -> None:
    with pytest.raises(ValueError, match="max_adverse_post_edge_share"):
        passive_fill_event_lifecycle_scorecard(pd.DataFrame(), max_adverse_post_edge_share=1.5)
    with pytest.raises(ValueError, match="missing passive fill event lifecycle toxicity columns"):
        passive_fill_event_lifecycle_scorecard(
            pd.DataFrame({"lifecycle_path": ["thin|stress|stress"]})
        )


def test_passive_fill_event_regime_summary_ranks_adverse_execution_windows() -> None:
    events = pd.DataFrame(
        {
            "event_regime": ["thin", "stressed", "stressed"],
            "event_fill_probability": [0.90, 0.85, 0.88],
            "event_adverse_fill_probability": [0.30, 0.35, 0.40],
            "event_edge_ticks": [0.70, 0.90, 1.10],
            "post_minus_pre_realized_edge": [-0.30, 0.40, -1.00],
        }
    )

    summary = passive_fill_event_regime_summary(events)

    assert summary["event_regime"].tolist() == ["thin", "stressed"]
    assert summary["events"].tolist() == [1, 2]
    assert summary["adverse_post_edge_share"].tolist() == pytest.approx([1.00, 0.50])
    assert summary["mean_event_fill_probability"].tolist() == pytest.approx([0.90, 0.865])
    assert summary["mean_event_adverse_fill_probability"].tolist() == pytest.approx([0.30, 0.375])
    assert summary["mean_post_minus_pre_realized_edge"].tolist() == pytest.approx([-0.30, -0.30])


def test_passive_fill_event_toxicity_scorecard_blocks_adverse_event_regimes() -> None:
    summary = pd.DataFrame(
        {
            "event_regime": ["stable", "thin"],
            "events": [5, 3],
            "adverse_post_edge_events": [1, 2],
            "adverse_post_edge_share": [0.20, 2.0 / 3.0],
            "mean_event_fill_probability": [0.72, 0.91],
            "mean_event_adverse_fill_probability": [0.18, 0.42],
            "mean_event_edge_ticks": [0.40, 0.80],
            "mean_post_minus_pre_realized_edge": [0.10, -0.40],
            "worst_post_minus_pre_realized_edge": [-0.20, -0.90],
        }
    )

    scorecard = passive_fill_event_toxicity_scorecard(
        summary,
        max_adverse_post_edge_share=0.60,
        min_mean_post_minus_pre_edge=-0.25,
    )

    assert scorecard == {
        "rows": 2,
        "regimes": 2,
        "total_events": 8,
        "eligible_regimes": 2,
        "blocked_regimes": 1,
        "worst_regime": "thin",
        "worst_adverse_post_edge_share": pytest.approx(2.0 / 3.0),
        "worst_mean_post_minus_pre_realized_edge": pytest.approx(-0.40),
        "worst_post_minus_pre_realized_edge": pytest.approx(-0.90),
        "weighted_mean_event_fill_probability": pytest.approx(0.79125),
        "weighted_mean_event_adverse_fill_probability": pytest.approx(0.27),
        "weighted_mean_post_minus_pre_realized_edge": pytest.approx(-0.0875),
        "event_toxicity_label": "event_window_blocker",
    }


def test_passive_fill_event_toxicity_scorecard_labels_pass_and_thin_samples() -> None:
    summary = pd.DataFrame(
        {
            "event_regime": ["stable"],
            "events": [2],
            "adverse_post_edge_events": [0],
            "adverse_post_edge_share": [0.0],
            "mean_event_fill_probability": [0.70],
            "mean_event_adverse_fill_probability": [0.10],
            "mean_event_edge_ticks": [0.30],
            "mean_post_minus_pre_realized_edge": [0.20],
            "worst_post_minus_pre_realized_edge": [0.10],
        }
    )

    assert passive_fill_event_toxicity_scorecard(summary)["event_toxicity_label"] == "event_window_pass"
    assert (
        passive_fill_event_toxicity_scorecard(summary, min_events=3)["event_toxicity_label"]
        == "insufficient_event_windows"
    )


def test_passive_fill_event_toxicity_scorecard_rejects_bad_summary() -> None:
    with pytest.raises(ValueError, match="max_adverse_post_edge_share"):
        passive_fill_event_toxicity_scorecard(pd.DataFrame(), max_adverse_post_edge_share=1.5)
    with pytest.raises(ValueError, match="missing passive fill event toxicity columns"):
        passive_fill_event_toxicity_scorecard(pd.DataFrame({"event_regime": ["thin"]}))


def test_passive_fill_event_window_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        passive_fill_event_window_diagnostics(pd.DataFrame(), threshold=1.5)


def test_execution_publishability_review_packet_surfaces_queue_gate_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "publishable_side": ["long", "long", "short", "abstain", "short"],
            "best_execution_side": ["long", "abstain", "long", "short", "short"],
            "execution_adjusted_edge_ticks": [0.70, -0.20, 0.10, 0.30, 0.80],
            "long_fill_adjusted_edge_ticks": [0.70, -0.20, 0.10, -0.10, -0.40],
            "short_fill_adjusted_edge_ticks": [-0.30, -0.10, -0.20, 0.30, 0.80],
            "bid_fill_probability": [0.80, 0.20, 0.60, 0.20, 0.10],
            "ask_fill_probability": [0.30, 0.40, 0.20, 0.70, 0.90],
            "bid_adverse_fill_probability": [0.10, 0.40, 0.20, 0.20, 0.30],
            "ask_adverse_fill_probability": [0.30, 0.30, 0.50, 0.10, 0.10],
        }
    )

    packet = execution_publishability_review_packet(frame)

    assert packet["review_priority"].tolist()[:2] == [3, 3]
    assert packet.loc[0, "publishable_side"] == "long"
    assert packet.loc[0, "best_execution_side"] == "abstain"
    assert packet.loc[0, "review_note"] == "pre-execution long signal abstains after queue/adverse-fill adjustment"
    assert packet.loc[0, "mean_publishable_fill_probability"] == pytest.approx(0.20)
    assert packet.loc[0, "mean_best_fill_probability"] == pytest.approx(0.0)
    assert packet.loc[1, "publishable_side"] == "short"
    assert packet.loc[1, "best_execution_side"] == "long"
    assert packet.loc[1, "mean_edge_drag_ticks"] == pytest.approx(0.30)
    assert packet.loc[1, "conflict_share"] == pytest.approx(1.0)


def test_execution_publishability_review_packet_handles_empty_frames() -> None:
    packet = execution_publishability_review_packet(pd.DataFrame())

    assert list(packet.columns) == [
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
    assert packet.empty


def test_execution_publishability_release_gate_blocks_fragile_capacity() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long", "long"],
            "best_execution_side": ["long", "abstain"],
            "rows": [75, 25],
            "conflict_rows": [0, 25],
            "conflict_share": [0.0, 1.0],
            "mean_execution_adjusted_edge_ticks": [0.24, -0.02],
            "mean_best_fill_probability": [0.62, 0.0],
            "mean_best_adverse_fill_probability": [0.15, 0.0],
            "mean_publishable_fill_probability": [0.62, 0.48],
            "mean_edge_drag_ticks": [0.0, -0.12],
            "review_priority": [0, 3],
            "review_note": ["agree", "pre-execution signal abstains"],
        }
    )
    quality_gate = {
        "quality_gate_label": "queue_execution_publishable",
        "blocked_regimes": 0,
    }
    capacity_stability = {
        "capacity_stability_label": "capacity_fragile",
        "capacity_fraction_gap": -0.20,
        "capacity_edge_gap_ticks": -0.05,
        "capacity_tradable_share_gap": -0.02,
        "dominant_side_changed": False,
    }

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate=quality_gate,
        capacity_stability=capacity_stability,
        max_conflict_share=0.30,
    )

    assert gate["decision"] == "block"
    assert gate["passes"] is False
    assert gate["release_gate_label"] == "execution_release_blocked"
    assert gate["total_rows"] == 100
    assert gate["weighted_conflict_share"] == pytest.approx(0.25)
    assert gate["high_priority_conflict_rows"] == 25
    assert gate["capacity_stability_label"] == "capacity_fragile"
    assert "capacity_fragile" in gate["blocking_reasons"]


def test_execution_publishability_release_gate_passes_clean_execution_evidence() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long", "abstain"],
            "best_execution_side": ["long", "abstain"],
            "rows": [80, 20],
            "conflict_rows": [0, 0],
            "conflict_share": [0.0, 0.0],
            "mean_execution_adjusted_edge_ticks": [0.31, 0.0],
            "mean_best_fill_probability": [0.66, 0.0],
            "mean_best_adverse_fill_probability": [0.10, 0.0],
            "mean_publishable_fill_probability": [0.66, 0.0],
            "mean_edge_drag_ticks": [0.0, 0.0],
            "review_priority": [0, 0],
            "review_note": ["agree", "agree"],
        }
    )
    quality_gate = {"quality_gate_label": "queue_execution_publishable"}
    capacity_stability = {"capacity_stability_label": "capacity_stable"}

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate=quality_gate,
        capacity_stability=capacity_stability,
    )

    assert gate["decision"] == "pass"
    assert gate["passes"] is True
    assert gate["release_gate_label"] == "execution_release_publishable"
    assert gate["blocking_reasons"] == "none"
    assert gate["review_reasons"] == "none"


def test_execution_adjusted_edge_summary_handles_empty_frames() -> None:
    summary = execution_adjusted_edge_summary(pd.DataFrame())

    assert summary == {
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


def test_execution_adjusted_edge_summary_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing execution summary columns"):
        execution_adjusted_edge_summary(pd.DataFrame({"best_execution_side": ["long"]}))


def test_execution_functions_reject_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position columns"):
        add_queue_position_features(pd.DataFrame({"bid_sz_1": [1.0]}))
    with pytest.raises(ValueError, match="missing fill probability columns"):
        add_passive_fill_probabilities(pd.DataFrame({"lcri": [1.0]}))
    with pytest.raises(ValueError, match="missing execution edge columns"):
        add_execution_adjusted_edge(pd.DataFrame({"lcri": [1.0]}))
    with pytest.raises(ValueError, match="missing execution publishability review columns"):
        execution_publishability_review_packet(pd.DataFrame({"publishable_side": ["long"]}))


def test_execution_functions_reject_non_finite_inputs() -> None:
    frame = _book_frame()
    frame.loc[0, "bid_sz_1"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        add_queue_position_features(frame, levels=2)


def test_passive_fill_threshold_policy_curve_scores_actionable_cutoffs() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "abstain"],
            "bid_fill_probability": [0.82, 0.55, 0.20, 0.25, 0.90],
            "ask_fill_probability": [0.20, 0.35, 0.77, 0.45, 0.10],
            "bid_realized_fill": [1.0, 0.0, 0.0, 0.0, 1.0],
            "ask_realized_fill": [0.0, 0.0, 1.0, 0.0, 0.0],
            "long_net_return_ticks": [0.60, -0.20, 0.10, 0.20, 0.40],
            "short_net_return_ticks": [-0.10, 0.30, 0.50, -0.40, -0.20],
            "execution_adjusted_edge_ticks": [0.32, 0.05, 0.28, -0.05, 0.0],
        }
    )

    curve = passive_fill_threshold_policy_curve(frame, thresholds=(0.50, 0.75))

    assert curve["threshold"].tolist() == pytest.approx([0.50, 0.75])
    assert curve["candidate_rows"].tolist() == [3, 2]
    assert curve["trade_share"].tolist() == pytest.approx([3 / 5, 2 / 5])
    assert curve["long_rows"].tolist() == [2, 1]
    assert curve["short_rows"].tolist() == [1, 1]
    assert curve["mean_predicted_fill_probability"].tolist() == pytest.approx(
        [(0.82 + 0.55 + 0.77) / 3, (0.82 + 0.77) / 2]
    )
    assert curve["realized_fill_rate"].tolist() == pytest.approx([2 / 3, 1.0])
    assert curve["mean_realized_edge_ticks"].tolist() == pytest.approx(
        [(0.60 - 0.20 + 0.50) / 3, (0.60 + 0.50) / 2]
    )
    assert curve["positive_edge_rate"].tolist() == pytest.approx([2 / 3, 1.0])
    assert curve["mean_execution_adjusted_edge_ticks"].tolist() == pytest.approx(
        [(0.32 + 0.05 + 0.28) / 3, (0.32 + 0.28) / 2]
    )
    assert curve["policy_label"].tolist() == ["broad_execution_policy", "selective_high_quality_policy"]


def test_passive_fill_threshold_policy_curve_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="threshold values must be in"):
        passive_fill_threshold_policy_curve(pd.DataFrame(), thresholds=(1.25,))
