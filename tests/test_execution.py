import pandas as pd
import pytest

from lcri_lab.execution import (
    FillProbabilityConfig,
    add_event_level_trade_confirmed_fill_proxy,
    add_execution_adjusted_edge,
    add_passive_fill_probabilities,
    add_latency_adjusted_passive_fill_probabilities,
    add_event_level_realized_fill_proxy,
    passive_fill_proxy_disagreement,
    trade_confirmed_passive_fill_latency_summary,
    queue_position_trade_confirmation_regime_scorecard,
    queue_position_trade_confirmation_release_scorecard,
    queue_position_trade_confirmation_surface,
    queue_position_unfilled_opportunity_curve,
    queue_position_unfilled_opportunity_scorecard,
    add_passive_fill_event_window_regimes,
    add_queue_position_features,
    add_queue_position_order_size_features,
    add_queue_position_realized_fill_proxy,
    execution_adjusted_edge_component_attribution,
    execution_adjusted_edge_summary,
    execution_adjusted_lcri_absorption_attribution,
    execution_adjusted_lcri_event_window_attribution,
    execution_adjusted_lcri_event_window_release_scorecard,
    execution_adjusted_lcri_quantile_diagnostics,
    execution_adjusted_lcri_regime_attribution,
    execution_adjusted_lcri_side_release_scorecard,
    passive_fill_brier_decomposition,
    passive_fill_calibration_curve,
    passive_fill_calibration_summary,
    passive_fill_event_regime_summary,
    passive_fill_event_lead_lag_profile,
    passive_fill_event_lead_lag_scorecard,
    passive_fill_event_lifecycle_policy_curve,
    passive_fill_event_lifecycle_scorecard,
    passive_fill_event_lifecycle_summary,
    passive_fill_event_policy_stability,
    passive_fill_event_policy_stability_scorecard,
    passive_fill_event_toxicity_scorecard,
    passive_fill_event_window_sensitivity,
    passive_fill_event_window_transition_stability,
    passive_fill_event_window_transition_stability_scorecard,
    passive_fill_event_window_transition_matrix,
    passive_fill_event_window_transition_scorecard,
    passive_fill_event_transition_policy_curve,
    passive_fill_event_transition_scorecard,
    passive_fill_event_transition_summary,
    passive_fill_event_window_regime_summary,
    passive_fill_event_window_diagnostics,
    execution_publishability_release_gate,
    execution_publishability_review_packet,
    passive_fill_edge_curve,
    passive_fill_realization_hazard_curve,
    passive_fill_realization_horizon_sweep,
    passive_fill_threshold_policy_curve,
    queue_position_adverse_selection_policy_frontier,
    queue_position_adverse_selection_policy_summary,
    queue_position_capacity_frontier,
    queue_position_capacity_stability,
    queue_position_execution_readiness_scorecard,
    queue_position_execution_quality_gate,
    queue_position_regime_capacity_concentration,
    queue_position_regime_capacity_frontier,
    queue_position_regime_capacity_stability,
    queue_position_regime_capacity_stability_summary,
    queue_position_calibration_drift,
    queue_position_calibration_residual_summary,
    queue_position_calibration_reliability_scorecard,
    queue_position_calibration_stability,
    queue_position_calibration_stability_summary,
    queue_position_edge_decay,
    queue_position_fill_calibration_surface,
    queue_position_fill_monotonicity_scorecard,
    queue_position_fill_surface,
    queue_position_toxicity_surface,
    queue_position_fraction_sweep,
    queue_position_order_size_capacity_frontier,
    queue_position_order_size_sweep,
    queue_position_expected_value_frontier,
    queue_position_expected_value_policy_selection,
    queue_position_expected_value_policy_scorecard,
    queue_position_expected_value_oos_validation,
    queue_position_expected_value_policy_drift,
    queue_position_expected_value_stress_summary,
    queue_position_expected_value_stress_table,
    queue_position_latency_sensitivity,
    queue_position_latency_edge_regime_surface,
    queue_position_latency_edge_survival,
    queue_position_latency_edge_survival_scorecard,
    queue_position_latency_regime_surface,
    queue_position_latency_release_scorecard,
    queue_position_lcri_tail_adverse_selection_release_scorecard,
    queue_position_lcri_tail_adverse_selection_surface,
    queue_position_lcri_tail_fill_residuals,
    queue_position_path_drawdown_episodes,
    queue_position_path_drawdown_summary,
    queue_position_path_risk_concentration,
    queue_position_path_risk_release_gate,
    queue_position_path_risk_scorecard,
    queue_position_path_tail_loss_release_gate,
    queue_position_path_tail_loss_scorecard,
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


def test_queue_position_order_size_features_estimate_full_queue_clearance() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2, queue_position_fraction=0.25)

    output = add_queue_position_order_size_features(queued, levels=2, order_size_fraction=0.10)

    assert output["bid_child_order_size"].tolist() == pytest.approx([10.0, 40.0])
    assert output["ask_child_order_size"].tolist() == pytest.approx([20.0, 10.0])
    assert output["bid_queue_clear_size"].tolist() == pytest.approx([35.0, 140.0])
    assert output["ask_queue_clear_size"].tolist() == pytest.approx([70.0, 35.0])
    assert output["bid_order_size_share"].tolist() == pytest.approx([10.0 / 150.0, 40.0 / 450.0])
    assert output["ask_order_size_share"].tolist() == pytest.approx([20.0 / 250.0, 10.0 / 150.0])
    assert output["bid_queue_clear_share"].tolist() == pytest.approx([35.0 / 150.0, 140.0 / 450.0])
    assert output["ask_queue_clear_share"].tolist() == pytest.approx([70.0 / 250.0, 35.0 / 150.0])
    assert output["queue_clear_size_imbalance"].tolist() == pytest.approx([-35.0, 105.0])


def test_queue_position_order_size_features_accept_explicit_child_order_columns() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2, queue_position_fraction=0.25).assign(
        bid_order=[5.0, 60.0],
        ask_order=[15.0, 20.0],
    )

    output = add_queue_position_order_size_features(
        queued,
        levels=2,
        order_size_fraction=0.99,
        bid_order_size_col="bid_order",
        ask_order_size_col="ask_order",
    )

    assert output["bid_child_order_size"].tolist() == pytest.approx([5.0, 60.0])
    assert output["ask_child_order_size"].tolist() == pytest.approx([15.0, 20.0])
    assert output["bid_queue_clear_size"].tolist() == pytest.approx([30.0, 160.0])
    assert output["ask_queue_clear_size"].tolist() == pytest.approx([65.0, 45.0])


def test_queue_position_order_size_features_rejects_negative_child_size() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2).assign(bid_order=[1.0, -1.0])

    with pytest.raises(ValueError, match="order sizes must be non-negative"):
        add_queue_position_order_size_features(queued, levels=2, bid_order_size_col="bid_order")


def test_latency_adjusted_passive_fill_probabilities_discount_stale_queue_state() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2, queue_position_fraction=0.25)
    sized = add_queue_position_order_size_features(queued, levels=2, order_size_fraction=0.10)
    probabilities = add_passive_fill_probabilities(sized)

    output = add_latency_adjusted_passive_fill_probabilities(probabilities, latency_steps=3.0)

    assert (output["bid_latency_adjusted_fill_probability"] <= output["bid_fill_probability"]).all()
    assert (output["ask_latency_adjusted_fill_probability"] <= output["ask_fill_probability"]).all()
    assert (output["bid_latency_adjusted_adverse_fill_probability"] >= output["bid_adverse_fill_probability"]).all()
    assert (output["ask_latency_adjusted_adverse_fill_probability"] >= output["ask_adverse_fill_probability"]).all()
    assert output["bid_latency_adjusted_fill_probability"].between(0.0, 1.0).all()
    assert output["ask_latency_adjusted_adverse_fill_probability"].between(0.0, 1.0).all()
    assert output["bid_latency_risk"].iloc[1] > output["bid_latency_risk"].iloc[0]


def test_latency_adjusted_passive_fill_probabilities_preserve_zero_latency_base_probabilities() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2, queue_position_fraction=0.25)
    probabilities = add_passive_fill_probabilities(queued)

    output = add_latency_adjusted_passive_fill_probabilities(probabilities, latency_steps=0.0)

    assert output["bid_latency_adjusted_fill_probability"].tolist() == pytest.approx(
        probabilities["bid_fill_probability"].tolist()
    )
    assert output["ask_latency_adjusted_fill_probability"].tolist() == pytest.approx(
        probabilities["ask_fill_probability"].tolist()
    )
    assert output["bid_latency_adjusted_adverse_fill_probability"].tolist() == pytest.approx(
        probabilities["bid_adverse_fill_probability"].tolist()
    )
    assert output["ask_latency_adjusted_adverse_fill_probability"].tolist() == pytest.approx(
        probabilities["ask_adverse_fill_probability"].tolist()
    )


def test_latency_adjusted_passive_fill_probabilities_accept_latency_column_and_reject_negative() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2, queue_position_fraction=0.25)
    probabilities = add_passive_fill_probabilities(queued).assign(decision_latency=[0.0, 2.0])

    output = add_latency_adjusted_passive_fill_probabilities(probabilities, latency_col="decision_latency")

    assert output["latency_steps"].tolist() == pytest.approx([0.0, 2.0])
    assert output["ask_latency_adjusted_fill_probability"].iloc[0] == pytest.approx(
        probabilities["ask_fill_probability"].iloc[0]
    )
    with pytest.raises(ValueError, match="latency values must be non-negative"):
        add_latency_adjusted_passive_fill_probabilities(
            probabilities.assign(decision_latency=[0.0, -1.0]), latency_col="decision_latency"
        )


def test_execution_adjusted_edge_component_attribution_decomposes_fill_and_adverse_drag() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "short", "abstain"],
            "long_net_return_ticks": [2.0, -1.0, 3.0],
            "short_net_return_ticks": [-2.0, 4.0, 1.0],
            "bid_fill_probability": [0.50, 0.90, 0.20],
            "ask_fill_probability": [0.10, 0.25, 0.30],
            "bid_adverse_fill_probability": [0.25, 0.20, 0.10],
            "ask_adverse_fill_probability": [0.05, 0.10, 0.20],
            "long_fill_adjusted_edge_ticks": [0.50, -1.10, 0.30],
            "short_fill_adjusted_edge_ticks": [-0.30, 0.60, 0.10],
            "execution_adjusted_edge_ticks": [0.50, 0.60, 0.0],
        }
    )

    attribution = execution_adjusted_edge_component_attribution(frame)

    assert attribution["best_execution_side"].tolist() == ["long", "short", "abstain"]
    assert attribution["rows"].tolist() == [1, 1, 1]
    assert attribution["mean_raw_edge_ticks"].tolist() == pytest.approx([2.0, 4.0, 0.0])
    assert attribution["mean_fill_captured_edge_ticks"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert attribution["mean_adverse_selection_cost_ticks"].tolist() == pytest.approx([0.5, 0.4, 0.0])
    assert attribution["mean_execution_adjusted_edge_ticks"].tolist() == pytest.approx([0.5, 0.6, 0.0])
    assert attribution["mean_fill_shortfall_ticks"].tolist() == pytest.approx([1.0, 3.0, 0.0])
    assert attribution["fill_capture_ratio"].tolist() == pytest.approx([0.5, 0.25, 0.0])
    assert attribution["adverse_drag_ratio"].tolist() == pytest.approx([0.25, 0.10, 0.0])


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


def test_queue_position_realized_fill_proxy_requires_child_order_clearance_when_present() -> None:
    frame = pd.DataFrame(
        {
            "bid_px_1": [100.0, 100.0],
            "ask_px_1": [101.0, 101.0],
            "bid_sz_1": [100.0, 55.0],
            "ask_sz_1": [80.0, 45.0],
            "bid_queue_ahead": [40.0, 40.0],
            "ask_queue_ahead": [30.0, 30.0],
            "bid_queue_clear_size": [55.0, 55.0],
            "ask_queue_clear_size": [45.0, 45.0],
        }
    )

    output = add_queue_position_realized_fill_proxy(frame)

    assert output["bid_queue_depletion_ratio"].tolist() == pytest.approx([45.0 / 55.0, 0.0])
    assert output["ask_queue_depletion_ratio"].tolist() == pytest.approx([35.0 / 45.0, 0.0])
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


def test_event_level_realized_fill_proxy_requires_child_order_clearance_when_present() -> None:
    snapshots = pd.DataFrame(
        {
            "timestamp": [0.0],
            "bid_px_1": [100.0],
            "ask_px_1": [101.0],
            "bid_queue_ahead": [40.0],
            "ask_queue_ahead": [30.0],
            "bid_queue_clear_size": [60.0],
            "ask_queue_clear_size": [50.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [0.25, 0.50],
            "event_type": ["trade", "trade"],
            "side": ["sell", "buy"],
            "price": [100.0, 101.0],
            "size": [45.0, 35.0],
        }
    )

    output = add_event_level_realized_fill_proxy(snapshots, events, horizon=1.0)

    assert output["bid_event_depletion_ratio"].tolist() == pytest.approx([45.0 / 60.0])
    assert output["ask_event_depletion_ratio"].tolist() == pytest.approx([35.0 / 50.0])
    assert output["bid_realized_fill"].tolist() == pytest.approx([0.0])
    assert output["ask_realized_fill"].tolist() == pytest.approx([0.0])


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


def test_event_level_trade_confirmed_fill_proxy_flags_cancel_only_false_fill() -> None:
    snapshots = pd.DataFrame(
        {
            "timestamp": [0.0],
            "bid_px_1": [100.0],
            "ask_px_1": [101.0],
            "bid_queue_ahead": [30.0],
            "ask_queue_ahead": [30.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [0.25],
            "event_type": ["cancel"],
            "side": ["bid"],
            "price": [100.0],
            "size": [35.0],
        }
    )

    output = add_event_level_trade_confirmed_fill_proxy(snapshots, events, horizon=1.0)

    assert output["bid_event_trade_depletion"].tolist() == pytest.approx([0.0])
    assert output["bid_event_cancel_depletion"].tolist() == pytest.approx([35.0])
    assert output["bid_event_total_queue_advance"].tolist() == pytest.approx([35.0])
    assert output["bid_trade_confirmed_fill"].tolist() == pytest.approx([0.0])
    assert output["bid_queue_advance_without_trade"].tolist() == pytest.approx([1.0])
    assert pd.isna(output.loc[0, "bid_trade_confirmed_fill_latency"])


def test_event_level_trade_confirmed_fill_proxy_requires_trade_after_queue_clearance() -> None:
    snapshots = pd.DataFrame(
        {
            "timestamp": [0.0, 10.0],
            "bid_px_1": [100.0, 100.0],
            "ask_px_1": [101.0, 101.0],
            "bid_queue_ahead": [30.0, 30.0],
            "ask_queue_ahead": [30.0, 30.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [0.10, 0.20, 0.40, 10.10, 10.20],
            "event_type": ["cancel", "cancel", "trade", "trade", "cancel"],
            "side": ["bid", "bid", "sell", "sell", "bid"],
            "price": [100.0, 100.0, 100.0, 100.0, 100.0],
            "size": [20.0, 10.0, 1.0, 1.0, 30.0],
        }
    )

    output = add_event_level_trade_confirmed_fill_proxy(snapshots, events, horizon=1.0)

    assert output["bid_trade_confirmed_fill"].tolist() == pytest.approx([1.0, 0.0])
    assert output["bid_trade_confirmed_fill_latency"].tolist()[0] == pytest.approx(0.40)
    assert pd.isna(output.loc[1, "bid_trade_confirmed_fill_latency"])
    assert output["bid_queue_advance_without_trade"].tolist() == pytest.approx([0.0, 1.0])


def test_event_level_trade_confirmed_fill_proxy_respects_child_order_clearance_and_groups() -> None:
    snapshots = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": [0.0, 0.0],
            "bid_px_1": [100.0, 50.0],
            "ask_px_1": [101.0, 51.0],
            "bid_queue_ahead": [20.0, 20.0],
            "ask_queue_ahead": [20.0, 20.0],
            "bid_queue_clear_size": [40.0, 40.0],
            "ask_queue_clear_size": [30.0, 30.0],
        }
    )
    events = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "timestamp": [0.10, 0.20, 0.10, 0.20],
            "event_type": ["trade", "trade", "cancel", "trade"],
            "side": ["sell", "sell", "bid", "sell"],
            "price": [100.0, 100.0, 50.0, 50.0],
            "size": [20.0, 19.0, 40.0, 1.0],
        }
    )

    output = add_event_level_trade_confirmed_fill_proxy(snapshots, events, horizon=1.0, group_cols="symbol")

    assert output["bid_event_total_queue_advance"].tolist() == pytest.approx([39.0, 41.0])
    assert output["bid_trade_confirmed_fill"].tolist() == pytest.approx([0.0, 1.0])
    assert output["bid_trade_confirmed_fill_latency"].tolist()[1] == pytest.approx(0.20)


def test_trade_confirmed_passive_fill_latency_summary_surfaces_latency_and_cancel_only_risk() -> None:
    frame = pd.DataFrame(
        {
            "bid_trade_confirmed_fill": [1.0, 0.0, 1.0, 0.0],
            "ask_trade_confirmed_fill": [0.0, 1.0, 0.0, 0.0],
            "bid_trade_confirmed_fill_latency": [0.20, pd.NA, 0.50, pd.NA],
            "ask_trade_confirmed_fill_latency": [pd.NA, 0.40, pd.NA, pd.NA],
            "bid_queue_advance_without_trade": [0.0, 1.0, 0.0, 0.0],
            "ask_queue_advance_without_trade": [0.0, 0.0, 1.0, 0.0],
            "bid_event_trade_depletion": [30.0, 0.0, 10.0, 0.0],
            "ask_event_trade_depletion": [0.0, 20.0, 0.0, 0.0],
            "bid_event_cancel_depletion": [0.0, 35.0, 5.0, 0.0],
            "ask_event_cancel_depletion": [0.0, 0.0, 40.0, 0.0],
        }
    )

    summary = trade_confirmed_passive_fill_latency_summary(
        frame,
        max_mean_latency=0.30,
        max_cancel_only_clear_rate=0.20,
    )
    by_side = summary.set_index("side")

    assert summary.columns.tolist() == [
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
    assert by_side.loc["bid", "trade_confirmed_fill_rate"] == pytest.approx(0.50)
    assert by_side.loc["bid", "cancel_only_clear_rate"] == pytest.approx(0.25)
    assert by_side.loc["bid", "mean_fill_latency"] == pytest.approx(0.35)
    assert by_side.loc["bid", "p95_fill_latency"] == pytest.approx(0.485)
    assert by_side.loc["bid", "review_label"] == "cancel_only_and_latency_risk"
    assert by_side.loc["ask", "trade_confirmed_fill_rate"] == pytest.approx(0.25)
    assert by_side.loc["ask", "mean_fill_latency"] == pytest.approx(0.40)
    assert by_side.loc["ask", "review_label"] == "cancel_only_and_latency_risk"
    assert by_side.loc["all", "rows"] == 8
    assert by_side.loc["all", "cancel_only_clear_rate"] == pytest.approx(0.25)


def test_trade_confirmed_passive_fill_latency_summary_rejects_invalid_fill_flags() -> None:
    with pytest.raises(ValueError, match=r"fill flags must be in \[0, 1\]"):
        trade_confirmed_passive_fill_latency_summary(
            pd.DataFrame(
                {
                    "bid_trade_confirmed_fill": [1.2],
                    "ask_trade_confirmed_fill": [0.0],
                    "bid_trade_confirmed_fill_latency": [0.1],
                    "ask_trade_confirmed_fill_latency": [pd.NA],
                    "bid_queue_advance_without_trade": [0.0],
                    "ask_queue_advance_without_trade": [0.0],
                    "bid_event_trade_depletion": [1.0],
                    "ask_event_trade_depletion": [0.0],
                    "bid_event_cancel_depletion": [0.0],
                    "ask_event_cancel_depletion": [0.0],
                }
            )
        )


def test_passive_fill_proxy_disagreement_audits_snapshot_vs_event_labels() -> None:
    frame = pd.DataFrame(
        {
            "bid_snapshot_fill": [1.0, 1.0, 0.0, 0.0],
            "bid_event_fill": [1.0, 0.0, 1.0, 0.0],
            "ask_snapshot_fill": [0.0, 1.0, 0.0, 1.0],
            "ask_event_fill": [0.0, 1.0, 0.0, 0.0],
        }
    )

    audit = passive_fill_proxy_disagreement(
        frame,
        snapshot_cols=("bid_snapshot_fill", "ask_snapshot_fill"),
        event_cols=("bid_event_fill", "ask_event_fill"),
        max_disagreement_rate=0.20,
    )
    by_side = audit.set_index("side")

    assert audit.columns.tolist() == [
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
    assert by_side.loc["bid", "disagreement_rate"] == pytest.approx(0.50)
    assert by_side.loc["bid", "false_positive_rate"] == pytest.approx(0.25)
    assert by_side.loc["bid", "false_negative_rate"] == pytest.approx(0.25)
    assert by_side.loc["bid", "precision"] == pytest.approx(0.50)
    assert by_side.loc["bid", "recall"] == pytest.approx(0.50)
    assert by_side.loc["bid", "review_label"] == "proxy_event_disagreement"
    assert by_side.loc["ask", "snapshot_event_fill_bias"] == pytest.approx(0.25)
    assert by_side.loc["ask", "review_label"] == "proxy_event_false_positive_bias"
    assert by_side.loc["all", "rows"] == 8
    assert by_side.loc["all", "disagreement_rate"] == pytest.approx(3 / 8)


def test_passive_fill_proxy_disagreement_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match=r"fill labels must be in \[0, 1\]"):
        passive_fill_proxy_disagreement(
            pd.DataFrame({"snapshot": [0.0, 1.2], "event": [0.0, 1.0]}),
            snapshot_cols=("snapshot",),
            event_cols=("event",),
            sides=("bid",),
        )


def test_passive_fill_probabilities_move_with_pressure_and_queue() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2, queue_position_fraction=0.50)
    output = add_passive_fill_probabilities(queued)

    assert output.loc[0, "bid_fill_probability"] > output.loc[1, "bid_fill_probability"]
    assert output.loc[1, "ask_fill_probability"] > output.loc[0, "ask_fill_probability"]
    assert output.loc[0, "bid_fill_probability"] > output.loc[0, "ask_fill_probability"]
    assert output.loc[1, "ask_fill_probability"] > output.loc[1, "bid_fill_probability"]
    assert output["passive_fill_regime"].tolist() == ["bid_depletion", "ask_depletion"]


def test_passive_fill_probabilities_penalize_child_order_clearance_size() -> None:
    queued = add_queue_position_features(_book_frame(), levels=2, queue_position_fraction=0.25)
    small_order = add_queue_position_order_size_features(queued, levels=2, order_size_fraction=0.05)
    large_order = add_queue_position_order_size_features(queued, levels=2, order_size_fraction=0.50)

    small_output = add_passive_fill_probabilities(small_order)
    large_output = add_passive_fill_probabilities(large_order)

    assert small_output["bid_fill_probability"].tolist()[0] > large_output["bid_fill_probability"].tolist()[0]
    assert small_output["ask_fill_probability"].tolist()[1] > large_output["ask_fill_probability"].tolist()[1]


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


def test_execution_adjusted_lcri_side_release_scorecard_flags_side_inversions() -> None:
    attribution = pd.DataFrame(
        {
            "lcri_side": ["long", "short", "neutral"],
            "rows": [40, 30, 10],
            "tradable_rows": [32, 18, 0],
            "execution_conflict_rows": [4, 15, 0],
            "execution_conflict_share": [0.10, 0.50, 0.0],
            "mean_signal_confidence": [0.76, 0.72, 0.50],
            "mean_execution_adjusted_edge_ticks": [0.45, -0.08, 0.0],
            "mean_fill_probability_advantage": [0.20, -0.04, 0.0],
            "mean_adverse_fill_probability_advantage": [0.12, 0.35, 0.0],
            "dominant_execution_side": ["long", "long", "none"],
            "review_label": [
                "execution_side_preserved",
                "execution_side_inversion_review",
                "neutral_signal",
            ],
        }
    )

    scorecard = execution_adjusted_lcri_side_release_scorecard(
        attribution,
        min_tradable_share=0.60,
        max_conflict_share=0.25,
        min_mean_edge_ticks=0.05,
    )

    assert scorecard == {
        "side_rows": 3,
        "directional_rows": 70,
        "directional_tradable_rows": 50,
        "directional_tradable_share": pytest.approx(50 / 70),
        "max_directional_conflict_share": pytest.approx(0.50),
        "inverted_side_count": 1,
        "negative_edge_side_count": 1,
        "weak_fill_advantage_side_count": 1,
        "worst_side": "short",
        "release_decision": "block",
        "review_note": "execution_lcri_side_inversion_blocked",
    }


def test_execution_adjusted_lcri_side_release_scorecard_passes_preserved_sides() -> None:
    attribution = pd.DataFrame(
        {
            "lcri_side": ["long", "short"],
            "rows": [50, 50],
            "tradable_rows": [45, 44],
            "execution_conflict_rows": [2, 3],
            "execution_conflict_share": [0.04, 0.06],
            "mean_signal_confidence": [0.80, 0.78],
            "mean_execution_adjusted_edge_ticks": [0.35, 0.28],
            "mean_fill_probability_advantage": [0.18, 0.14],
            "mean_adverse_fill_probability_advantage": [0.10, 0.12],
            "dominant_execution_side": ["long", "short"],
            "review_label": ["execution_side_preserved", "execution_side_preserved"],
        }
    )

    scorecard = execution_adjusted_lcri_side_release_scorecard(attribution)

    assert scorecard["release_decision"] == "pass"
    assert scorecard["review_note"] == "execution_lcri_side_supported"
    assert scorecard["worst_side"] == "short"


def test_execution_adjusted_lcri_side_release_scorecard_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="missing execution-adjusted LCRI side scorecard columns"):
        execution_adjusted_lcri_side_release_scorecard(pd.DataFrame({"lcri_side": ["long"]}))


def test_execution_adjusted_lcri_quantile_diagnostics_measures_signal_survival() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "execution_adjusted_lcri_score": [-3.0, 0.0, -1.0, 0.0, 2.0, 3.0],
            "execution_adjusted_edge_ticks": [1.2, -0.2, 0.3, -0.1, 0.5, 1.4],
            "best_execution_side": ["short", "abstain", "short", "abstain", "long", "long"],
            "bid_fill_probability": [0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
            "ask_fill_probability": [0.80, 0.20, 0.60, 0.30, 0.50, 0.40],
            "bid_adverse_fill_probability": [0.10, 0.20, 0.30, 0.40, 0.10, 0.20],
            "ask_adverse_fill_probability": [0.20, 0.30, 0.10, 0.50, 0.40, 0.30],
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
        "mean_selected_fill_probability",
        "mean_selected_adverse_fill_probability",
        "fill_minus_adverse_probability_spread",
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
    assert diagnostics["mean_selected_fill_probability"].tolist() == pytest.approx([0.30, 0.35, 0.85])
    assert diagnostics["mean_selected_adverse_fill_probability"].tolist() == pytest.approx([0.05, 0.05, 0.20])
    assert diagnostics["fill_minus_adverse_probability_spread"].tolist() == pytest.approx([0.25, 0.30, 0.65])
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


def test_passive_fill_realization_hazard_curve_attributes_immediate_vs_delayed_fills() -> None:
    frame = pd.DataFrame(
        {
            "session": ["am", "am", "am", "pm", "pm", "pm"],
            "bid_px_1": [100.0, 100.0, 99.0, 200.0, 200.0, 200.0],
            "ask_px_1": [101.0, 101.0, 102.0, 201.0, 201.0, 201.0],
            "bid_sz_1": [100.0, 85.0, 120.0, 90.0, 75.0, 75.0],
            "ask_sz_1": [80.0, 70.0, 90.0, 70.0, 68.0, 68.0],
            "bid_queue_ahead": [40.0, 40.0, 40.0, 30.0, 30.0, 30.0],
            "ask_queue_ahead": [30.0, 30.0, 30.0, 20.0, 20.0, 20.0],
            "best_execution_side": ["long", "short", "long", "long", "short", "abstain"],
            "bid_fill_probability": [0.60, 0.20, 0.80, 0.50, 0.10, 0.40],
            "ask_fill_probability": [0.30, 0.70, 0.40, 0.20, 0.90, 0.40],
        }
    )

    curve = passive_fill_realization_hazard_curve(
        frame,
        horizons=[1, 2],
        group_cols="session",
        regime_col="session",
    )

    assert curve["session"].tolist() == ["am", "am", "pm", "pm"]
    assert curve["horizon"].tolist() == [1, 2, 1, 2]
    assert curve["eligible_rows"].tolist() == [3, 3, 2, 2]
    assert curve["cumulative_realized_fill_rate"].tolist() == pytest.approx([1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0])
    assert curve["incremental_realized_fill_rate"].tolist() == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0])
    assert curve["conditional_fill_hazard"].tolist() == pytest.approx([1.0 / 3.0, 0.5, 0.0, 0.0])
    assert curve["mean_selected_fill_probability"].tolist() == pytest.approx([0.70, 0.70, 0.70, 0.70])
    assert curve["timing_slippage_vs_prediction"].tolist() == pytest.approx([-0.3666666667, -0.0333333333, -0.70, -0.70])
    assert curve["horizon_timing_label"].tolist() == [
        "under_realized",
        "near_prediction",
        "under_realized",
        "under_realized",
    ]


def test_passive_fill_realization_hazard_curve_rejects_invalid_horizons() -> None:
    with pytest.raises(ValueError, match="horizons must be a non-empty sequence"):
        passive_fill_realization_hazard_curve(pd.DataFrame(), horizons=[])
    with pytest.raises(ValueError, match="horizon values must be positive integers"):
        passive_fill_realization_hazard_curve(pd.DataFrame(), horizons=[1, 0])


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


def test_queue_position_adverse_selection_policy_frontier_scores_fill_and_toxicity_cutoffs() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "abstain"],
            "bid_fill_probability": [0.80, 0.65, 0.20, 0.30, 0.95],
            "ask_fill_probability": [0.40, 0.35, 0.72, 0.88, 0.95],
            "bid_adverse_fill_probability": [0.15, 0.35, 0.10, 0.10, 0.05],
            "ask_adverse_fill_probability": [0.10, 0.20, 0.25, 0.45, 0.05],
            "bid_realized_fill": [1.0, 0.0, 0.0, 0.0, 1.0],
            "ask_realized_fill": [0.0, 0.0, 1.0, 1.0, 1.0],
            "long_net_return_ticks": [2.0, -1.0, 0.0, 0.0, 3.0],
            "short_net_return_ticks": [0.0, 0.0, 1.5, -2.0, 3.0],
            "execution_adjusted_edge_ticks": [1.6, 0.2, 1.1, -0.4, 2.0],
        }
    )

    frontier = queue_position_adverse_selection_policy_frontier(
        frame,
        fill_thresholds=(0.60, 0.75),
        adverse_thresholds=(0.30, 0.40),
    )

    assert frontier["policy_label"].tolist() == [
        "balanced_execution_policy",
        "edge_positive_fill_uncertain_policy",
        "selective_toxicity_control_policy",
        "selective_toxicity_control_policy",
    ]
    first = frontier.iloc[0]
    assert first["candidate_rows"] == 2
    assert first["trade_share"] == pytest.approx(0.40)
    assert first["realized_fill_rate"] == pytest.approx(1.0)
    assert first["mean_adverse_fill_probability"] == pytest.approx(0.20)
    assert first["mean_realized_edge_ticks"] == pytest.approx(1.75)
    assert first["toxicity_filtered_rows"] == 2
    assert first["toxicity_filtered_share"] == pytest.approx(0.50)


def test_queue_position_adverse_selection_policy_frontier_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError, match="fill_thresholds must be a non-empty sequence"):
        queue_position_adverse_selection_policy_frontier(pd.DataFrame(), fill_thresholds=())
    with pytest.raises(ValueError, match="adverse threshold values must be in"):
        queue_position_adverse_selection_policy_frontier(pd.DataFrame(), adverse_thresholds=(1.20,))


def test_queue_position_adverse_selection_policy_summary_selects_best_publishable_policy() -> None:
    frontier = pd.DataFrame(
        {
            "fill_threshold": [0.55, 0.65, 0.75],
            "adverse_threshold": [0.35, 0.30, 0.25],
            "candidate_rows": [40, 20, 8],
            "trade_share": [0.40, 0.20, 0.08],
            "long_rows": [25, 15, 8],
            "short_rows": [15, 5, 0],
            "mean_predicted_fill_probability": [0.70, 0.76, 0.82],
            "mean_adverse_fill_probability": [0.32, 0.22, 0.18],
            "realized_fill_rate": [0.72, 0.80, 0.88],
            "mean_realized_edge_ticks": [0.20, 0.55, 0.90],
            "positive_edge_rate": [0.58, 0.70, 0.90],
            "mean_execution_adjusted_edge_ticks": [0.18, 0.48, 0.80],
            "toxicity_filtered_rows": [10, 15, 22],
            "toxicity_filtered_share": [0.10, 0.15, 0.22],
            "policy_label": [
                "edge_positive_fill_uncertain_policy",
                "selective_toxicity_control_policy",
                "high_quality_capacity_constrained_policy",
            ],
        }
    )

    summary = queue_position_adverse_selection_policy_summary(
        frontier,
        min_trade_share=0.10,
        min_realized_fill_rate=0.75,
        min_mean_realized_edge_ticks=0.25,
        max_mean_adverse_fill_probability=0.30,
    )

    assert summary["policies"] == 3
    assert summary["publishable_policies"] == 1
    assert summary["best_fill_threshold"] == pytest.approx(0.65)
    assert summary["best_adverse_threshold"] == pytest.approx(0.30)
    assert summary["best_policy_label"] == "selective_toxicity_control_policy"
    assert summary["dominant_side"] == "long"
    assert summary["policy_summary_label"] == "publishable_toxicity_control_policy"


def test_queue_position_adverse_selection_policy_summary_flags_no_viable_policy() -> None:
    frontier = pd.DataFrame(
        {
            "fill_threshold": [0.55],
            "adverse_threshold": [0.35],
            "candidate_rows": [5],
            "trade_share": [0.05],
            "long_rows": [3],
            "short_rows": [2],
            "mean_predicted_fill_probability": [0.65],
            "mean_adverse_fill_probability": [0.45],
            "realized_fill_rate": [0.40],
            "mean_realized_edge_ticks": [-0.10],
            "positive_edge_rate": [0.40],
            "mean_execution_adjusted_edge_ticks": [-0.20],
            "toxicity_filtered_rows": [12],
            "toxicity_filtered_share": [0.60],
            "policy_label": ["execution_policy_rejected"],
        }
    )

    summary = queue_position_adverse_selection_policy_summary(frontier)

    assert summary["publishable_policies"] == 0
    assert summary["best_policy_label"] == "none"
    assert summary["policy_summary_label"] == "no_publishable_toxicity_control_policy"


def test_queue_position_adverse_selection_policy_summary_rejects_invalid_frontier() -> None:
    with pytest.raises(ValueError, match="min_trade_share"):
        queue_position_adverse_selection_policy_summary(pd.DataFrame(), min_trade_share=1.5)
    with pytest.raises(ValueError, match="missing queue position adverse-selection policy summary columns"):
        queue_position_adverse_selection_policy_summary(pd.DataFrame({"fill_threshold": [0.5]}))


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


def test_queue_position_toxicity_surface_flags_adverse_deep_queue_cells() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "abstain", "long"],
            "regime": ["open", "open", "open", "thin", "open", "thin"],
            "bid_queue_share": [0.10, 0.85, 0.20, 0.40, 0.50, 0.90],
            "ask_queue_share": [0.30, 0.50, 0.20, 0.80, 0.50, 0.25],
            "bid_fill_probability": [0.75, 0.80, 0.20, 0.30, 0.50, 0.90],
            "ask_fill_probability": [0.20, 0.40, 0.70, 0.85, 0.50, 0.30],
            "bid_adverse_fill_probability": [0.10, 0.65, 0.20, 0.30, 0.50, 0.70],
            "ask_adverse_fill_probability": [0.20, 0.40, 0.15, 0.75, 0.50, 0.30],
            "bid_realized_fill": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "ask_realized_fill": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            "long_net_return_ticks": [0.40, -0.20, 0.0, 0.0, 0.0, -0.50],
            "short_net_return_ticks": [0.0, 0.0, 0.30, -0.40, 0.0, 0.0],
            "execution_adjusted_edge_ticks": [0.30, -0.10, 0.20, -0.30, 0.0, -0.40],
        }
    )

    surface = queue_position_toxicity_surface(frame, queue_bins=2, regime_col="regime")

    assert surface["regime"].tolist() == ["open", "open", "open", "thin", "thin"]
    assert surface["best_execution_side"].tolist() == ["long", "long", "short", "long", "short"]
    assert surface["queue_toxicity_label"].tolist() == [
        "benign_queue_fill",
        "toxic_queue_fill",
        "benign_queue_fill",
        "toxic_queue_fill",
        "toxic_queue_fill",
    ]
    assert surface["adverse_to_fill_ratio"].tolist() == pytest.approx(
        [0.10 / 0.75, 0.65 / 0.80, 0.15 / 0.70, 0.70 / 0.90, 0.75 / 0.85]
    )
    assert surface["realized_loss_rate"].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0, 1.0])


def test_queue_position_toxicity_surface_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="queue_bins must be at least 1"):
        queue_position_toxicity_surface(pd.DataFrame(), queue_bins=0)
    with pytest.raises(ValueError, match="missing queue position toxicity surface columns"):
        queue_position_toxicity_surface(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_lcri_tail_fill_residuals_audits_tail_execution_calibration() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["open", "open", "open", "open", "thin", "thin"],
            "best_execution_side": ["long", "long", "long", "short", "short", "abstain"],
            "lcri": [-0.2, -1.5, -3.0, 2.0, 4.0, 0.1],
            "bid_fill_probability": [0.30, 0.80, 0.90, 0.20, 0.10, 0.40],
            "ask_fill_probability": [0.20, 0.20, 0.10, 0.70, 0.95, 0.50],
            "bid_realized_fill": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "ask_realized_fill": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            "execution_adjusted_edge_ticks": [0.1, 1.2, 1.5, 0.8, -0.4, 0.0],
        }
    )

    residuals = queue_position_lcri_tail_fill_residuals(
        frame,
        lcri_bins=2,
        max_abs_fill_residual=0.30,
    )

    assert residuals["regime"].tolist() == ["open", "open", "open", "thin"]
    assert residuals["best_execution_side"].tolist() == ["long", "long", "short", "short"]
    assert residuals["lcri_tail_bin"].tolist() == [1, 2, 1, 1]
    assert residuals["rows"].tolist() == [2, 1, 1, 1]
    assert residuals["mean_abs_lcri"].tolist() == pytest.approx([0.85, 3.0, 2.0, 4.0])
    assert residuals["mean_predicted_fill_probability"].tolist() == pytest.approx([0.55, 0.90, 0.70, 0.95])
    assert residuals["realized_fill_rate"].tolist() == pytest.approx([0.50, 0.00, 1.00, 0.00])
    assert residuals["fill_residual"].tolist() == pytest.approx([-0.05, -0.90, 0.30, -0.95])
    assert residuals["mean_execution_adjusted_edge_ticks"].tolist() == pytest.approx([0.65, 1.50, 0.80, -0.40])
    assert residuals["tail_fill_residual_label"].tolist() == [
        "tail_fill_calibrated",
        "tail_fill_overstated",
        "tail_fill_calibrated",
        "tail_fill_overstated",
    ]


def test_queue_position_lcri_tail_fill_residuals_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="lcri_bins must be at least 1"):
        queue_position_lcri_tail_fill_residuals(pd.DataFrame(), lcri_bins=0)
    with pytest.raises(ValueError, match="missing queue position LCRI tail fill residual columns"):
        queue_position_lcri_tail_fill_residuals(pd.DataFrame({"best_execution_side": ["long"]}))


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


def test_queue_position_order_size_sweep_quantifies_child_order_capacity_decay() -> None:
    frame = add_queue_position_features(
        _book_frame().assign(
            lcri_probability=[0.25, 0.75],
            long_net_return_ticks=[1.0, 1.5],
            short_net_return_ticks=[1.5, 1.0],
        ),
        levels=2,
        queue_position_fraction=0.25,
    )

    sweep = queue_position_order_size_sweep(
        frame,
        order_size_fractions=[0.0, 0.2, 0.4],
        levels=2,
        fill_config=FillProbabilityConfig(adverse_selection_scale=0.25),
    )

    assert sweep["order_size_fraction"].tolist() == pytest.approx([0.0, 0.2, 0.4])
    assert sweep["rows"].tolist() == [2, 2, 2]
    assert sweep["mean_bid_child_order_size"].is_monotonic_increasing
    assert sweep["mean_ask_child_order_size"].is_monotonic_increasing
    assert sweep["mean_bid_queue_clear_share"].is_monotonic_increasing
    assert sweep["mean_ask_queue_clear_share"].is_monotonic_increasing
    assert sweep["mean_bid_fill_probability"].is_monotonic_decreasing
    assert sweep["mean_ask_fill_probability"].is_monotonic_decreasing
    assert sweep["mean_execution_adjusted_edge_ticks"].iloc[0] > sweep[
        "mean_execution_adjusted_edge_ticks"
    ].iloc[-1]
    assert sweep["tradable_share"].between(0.0, 1.0).all()
    assert set(sweep["dominant_execution_side"]).issubset({"long", "short", "none"})


def test_queue_position_order_size_sweep_rejects_invalid_order_size_fractions() -> None:
    with pytest.raises(ValueError, match="order_size_fractions"):
        queue_position_order_size_sweep(_book_frame(), order_size_fractions=[])
    with pytest.raises(ValueError, match="order_size_fraction"):
        queue_position_order_size_sweep(_book_frame(), order_size_fractions=[-0.1])


def test_queue_position_order_size_capacity_frontier_identifies_max_viable_child_size() -> None:
    sweep = pd.DataFrame(
        {
            "order_size_fraction": [0.0, 0.1, 0.25, 0.5],
            "rows": [20, 20, 20, 20],
            "mean_execution_adjusted_edge_ticks": [0.42, 0.31, 0.12, -0.04],
            "tradable_share": [0.90, 0.76, 0.54, 0.30],
            "dominant_execution_side": ["long", "long", "short", "short"],
        }
    )

    frontier = queue_position_order_size_capacity_frontier(
        sweep,
        min_edge_ticks=0.10,
        min_tradable_share=0.50,
    )

    assert frontier["rows"] == 4
    assert frontier["viable_rows"] == 3
    assert frontier["minimum_order_size_fraction"] == pytest.approx(0.0)
    assert frontier["max_viable_order_size_fraction"] == pytest.approx(0.25)
    assert frontier["minimum_size_mean_execution_adjusted_edge_ticks"] == pytest.approx(0.42)
    assert frontier["max_viable_mean_execution_adjusted_edge_ticks"] == pytest.approx(0.12)
    assert frontier["edge_decay_to_capacity_ticks"] == pytest.approx(0.30)
    assert frontier["minimum_size_tradable_share"] == pytest.approx(0.90)
    assert frontier["max_viable_tradable_share"] == pytest.approx(0.54)
    assert frontier["tradable_share_decay_to_capacity"] == pytest.approx(0.36)
    assert frontier["dominant_execution_side_at_capacity"] == "short"
    assert frontier["order_size_capacity_label"] == "child_order_capacity_constrained"


def test_queue_position_order_size_capacity_frontier_marks_no_viable_capacity() -> None:
    frontier = queue_position_order_size_capacity_frontier(
        pd.DataFrame(
            {
                "order_size_fraction": [0.1, 0.2],
                "rows": [10, 10],
                "mean_execution_adjusted_edge_ticks": [-0.02, -0.05],
                "tradable_share": [0.20, 0.10],
                "dominant_execution_side": ["none", "none"],
            }
        ),
        min_edge_ticks=0.0,
        min_tradable_share=0.50,
    )

    assert frontier["rows"] == 2
    assert frontier["viable_rows"] == 0
    assert frontier["minimum_order_size_fraction"] == pytest.approx(0.1)
    assert frontier["minimum_size_mean_execution_adjusted_edge_ticks"] == pytest.approx(-0.02)
    assert frontier["minimum_size_tradable_share"] == pytest.approx(0.20)
    assert frontier["order_size_capacity_label"] == "no_viable_child_order_capacity"


def test_queue_position_order_size_capacity_frontier_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="order size capacity frontier fractions"):
        queue_position_order_size_capacity_frontier(
            pd.DataFrame(
                {
                    "order_size_fraction": [-0.1],
                    "rows": [1],
                    "mean_execution_adjusted_edge_ticks": [0.1],
                    "tradable_share": [1.0],
                    "dominant_execution_side": ["long"],
                }
            )
        )
    with pytest.raises(ValueError, match="min_tradable_share"):
        queue_position_order_size_capacity_frontier(pd.DataFrame(), min_tradable_share=1.1)


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


def test_queue_position_regime_capacity_stability_flags_heldout_state_fragility() -> None:
    research = pd.DataFrame(
        {
            "regime": ["open", "stress", "thin"],
            "viable_rows": [3, 1, 0],
            "max_viable_queue_position_fraction": [0.75, 0.25, 0.0],
            "max_viable_mean_execution_adjusted_edge_ticks": [0.42, 0.20, 0.0],
            "max_viable_tradable_share": [0.80, 0.62, 0.0],
            "dominant_execution_side_at_capacity": ["long", "short", "none"],
            "capacity_label": [
                "queue_capacity_constrained",
                "queue_capacity_front_only",
                "no_viable_passive_capacity",
            ],
        }
    )
    heldout = pd.DataFrame(
        {
            "regime": ["open", "stress", "auction"],
            "viable_rows": [3, 0, 2],
            "max_viable_queue_position_fraction": [0.50, 0.0, 0.50],
            "max_viable_mean_execution_adjusted_edge_ticks": [0.30, 0.0, 0.25],
            "max_viable_tradable_share": [0.70, 0.0, 0.65],
            "dominant_execution_side_at_capacity": ["long", "none", "short"],
            "capacity_label": [
                "queue_capacity_constrained",
                "no_viable_passive_capacity",
                "queue_capacity_constrained",
            ],
        }
    )

    stability = queue_position_regime_capacity_stability(research, heldout)

    rows = stability.set_index("regime")
    assert rows.loc["open", "capacity_fraction_gap"] == pytest.approx(-0.25)
    assert rows.loc["open", "regime_capacity_stability_label"] == "regime_capacity_fragile"
    assert rows.loc["stress", "lost_capacity"] is True
    assert rows.loc["stress", "regime_capacity_stability_label"] == "regime_capacity_lost"
    assert rows.loc["thin", "heldout_missing"] is True
    assert rows.loc["auction", "research_missing"] is True

    summary = queue_position_regime_capacity_stability_summary(stability)
    assert summary == {
        "regimes": 4,
        "common_regimes": 2,
        "missing_research_regimes": 1,
        "missing_heldout_regimes": 1,
        "stable_regimes": 0,
        "fragile_regimes": 1,
        "lost_capacity_regimes": 1,
        "gained_capacity_regimes": 1,
        "stable_regime_share": pytest.approx(0.0),
        "lost_capacity_share": pytest.approx(0.25),
        "mean_capacity_fraction_gap": pytest.approx(-0.25),
        "worst_regime": "stress",
        "worst_regime_capacity_stability_label": "regime_capacity_lost",
        "regime_capacity_stability_label": "regime_capacity_not_replicated",
    }


def test_queue_position_regime_capacity_stability_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="regime_col"):
        queue_position_regime_capacity_stability(pd.DataFrame(), pd.DataFrame(), regime_col="")
    with pytest.raises(ValueError, match="missing research regime capacity frontier columns"):
        queue_position_regime_capacity_stability(pd.DataFrame({"regime": ["open"]}), pd.DataFrame())
    with pytest.raises(ValueError, match="missing regime capacity stability columns"):
        queue_position_regime_capacity_stability_summary(pd.DataFrame({"regime": ["open"]}))


def test_queue_position_calibration_drift_flags_regime_instability() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["open", "stress", "open", "stress", "open"],
            "best_execution_side": ["long", "long", "short", "short", "long"],
            "queue_share_bin": [1, 1, 2, 2, 2],
            "fill_probability_bin": [1, 1, 1, 1, 2],
            "rows": [40, 20, 30, 30, 10],
            "mean_queue_share": [0.20, 0.25, 0.60, 0.65, 0.80],
            "mean_predicted_fill_probability": [0.70, 0.68, 0.50, 0.52, 0.30],
            "realized_fill_rate": [0.66, 0.32, 0.45, 0.40, 0.25],
            "calibration_error": [-0.04, -0.36, -0.05, -0.12, -0.05],
            "absolute_calibration_error": [0.04, 0.36, 0.05, 0.12, 0.05],
            "brier_score": [0.10, 0.30, 0.08, 0.12, 0.10],
            "mean_execution_adjusted_edge_ticks": [0.50, -0.20, 0.30, 0.20, 0.10],
        }
    )

    drift = queue_position_calibration_drift(surface)

    rows = drift.set_index(["best_execution_side", "queue_share_bin", "fill_probability_bin"])
    long_front = rows.loc[("long", 1, 1)]
    assert long_front["regimes"] == 2
    assert long_front["rows"] == 60
    assert long_front["fill_rate_range"] == pytest.approx(0.34)
    assert long_front["calibration_error_range"] == pytest.approx(0.32)
    assert long_front["weighted_mean_absolute_calibration_error"] == pytest.approx(
        ((40 * 0.04) + (20 * 0.36)) / 60
    )
    assert long_front["worst_regime"] == "stress"
    assert long_front["drift_label"] == "calibration_unstable"

    short_mid = rows.loc[("short", 2, 1)]
    assert short_mid["drift_label"] == "calibration_watch"
    assert ("long", 2, 2) not in rows.index


def test_queue_position_calibration_drift_rejects_bad_surface() -> None:
    with pytest.raises(ValueError, match="min_regimes"):
        queue_position_calibration_drift(pd.DataFrame(), min_regimes=1)
    with pytest.raises(ValueError, match="missing queue position calibration drift columns"):
        queue_position_calibration_drift(pd.DataFrame({"regime": ["open"]}))


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


def test_queue_position_execution_quality_gate_blocks_calibration_drift() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["open", "stress"],
            "rows": [80, 80],
            "absolute_calibration_error": [0.04, 0.05],
            "brier_score": [0.03, 0.04],
        }
    )
    decay = pd.DataFrame(
        {
            "regime": ["open", "stress"],
            "rows": [80, 80],
            "edge_decay_ticks": [0.30, 0.25],
            "calibration_error_widening": [0.01, 0.02],
            "monotonic_edge_decay": [True, True],
        }
    )
    drift = pd.DataFrame(
        {
            "best_execution_side": ["long", "short"],
            "queue_share_bin": [1, 2],
            "fill_probability_bin": [1, 1],
            "regimes": [2, 2],
            "rows": [100, 60],
            "fill_rate_range": [0.32, 0.08],
            "calibration_error_range": [0.22, 0.04],
            "weighted_mean_absolute_calibration_error": [0.11, 0.05],
            "worst_regime": ["stress", "open"],
            "drift_label": ["calibration_unstable", "calibration_stable"],
        }
    )

    gate = queue_position_execution_quality_gate(
        surface,
        decay,
        drift=drift,
        max_drift_fill_rate_range=0.25,
        max_drift_calibration_error_range=0.15,
    )

    assert gate["drift_rows"] == 160
    assert gate["unstable_drift_bins"] == 1
    assert gate["watch_drift_bins"] == 0
    assert gate["worst_drift_regime"] == "stress"
    assert gate["max_drift_fill_rate_range"] == pytest.approx(0.32)
    assert gate["max_drift_calibration_error_range"] == pytest.approx(0.22)
    assert gate["quality_gate_label"] == "queue_execution_blocked"


def test_queue_position_execution_readiness_scorecard_blocks_quality_and_capacity_risks() -> None:
    quality_gate = {
        "quality_gate_label": "queue_execution_blocked",
        "blocked_regimes": 2,
        "eligible_regimes": 3,
        "weighted_absolute_calibration_error": 0.18,
        "weighted_brier_score": 0.12,
        "max_regime_absolute_calibration_error": 0.31,
        "worst_calibration_regime": "stress",
        "worst_decay_regime": "thin",
    }
    capacity_stability = {
        "capacity_stability_label": "capacity_fragile",
        "capacity_fraction_gap": -0.25,
        "capacity_edge_gap_ticks": -0.18,
        "capacity_tradable_share_gap": -0.09,
        "dominant_side_changed": True,
    }
    concentration = {
        "capacity_concentration_label": "capacity_regime_concentrated",
        "front_only_or_no_capacity_share": 0.50,
        "worst_capacity_regime": "stress",
    }

    scorecard = queue_position_execution_readiness_scorecard(
        quality_gate, capacity_stability, concentration
    )

    assert scorecard == {
        "quality_gate_label": "queue_execution_blocked",
        "capacity_stability_label": "capacity_fragile",
        "capacity_concentration_label": "capacity_regime_concentrated",
        "queue_toxicity_label": "not_supplied",
        "blocked_regimes": 2,
        "eligible_regimes": 3,
        "execution_blocker_count": 3,
        "worst_calibration_regime": "stress",
        "worst_decay_regime": "thin",
        "worst_capacity_regime": "stress",
        "worst_toxicity_regime": "none",
        "weighted_absolute_calibration_error": pytest.approx(0.18),
        "weighted_brier_score": pytest.approx(0.12),
        "max_regime_absolute_calibration_error": pytest.approx(0.31),
        "capacity_fraction_gap": pytest.approx(-0.25),
        "capacity_edge_gap_ticks": pytest.approx(-0.18),
        "capacity_tradable_share_gap": pytest.approx(-0.09),
        "front_only_or_no_capacity_share": pytest.approx(0.50),
        "toxic_queue_row_share": pytest.approx(0.0),
        "toxic_queue_regimes": 0,
        "worst_toxicity_adverse_to_fill_ratio": pytest.approx(0.0),
        "worst_toxicity_realized_loss_rate": pytest.approx(0.0),
        "toxic_queue_mean_edge_ticks": pytest.approx(0.0),
        "dominant_side_changed": True,
        "execution_readiness_label": "execution_not_publishable",
    }


def test_queue_position_execution_readiness_scorecard_labels_publishable() -> None:
    scorecard = queue_position_execution_readiness_scorecard(
        {
            "quality_gate_label": "queue_execution_publishable",
            "blocked_regimes": 0,
            "eligible_regimes": 2,
            "weighted_absolute_calibration_error": 0.04,
            "weighted_brier_score": 0.05,
            "max_regime_absolute_calibration_error": 0.06,
            "worst_calibration_regime": "open",
            "worst_decay_regime": "open",
        },
        {
            "capacity_stability_label": "capacity_stable",
            "capacity_fraction_gap": -0.02,
            "capacity_edge_gap_ticks": -0.01,
            "capacity_tradable_share_gap": -0.01,
            "dominant_side_changed": False,
        },
    )

    assert scorecard["execution_blocker_count"] == 0
    assert scorecard["capacity_concentration_label"] == "not_supplied"
    assert scorecard["queue_toxicity_label"] == "not_supplied"
    assert scorecard["execution_readiness_label"] == "execution_publishable"


def test_queue_position_execution_readiness_scorecard_blocks_toxic_queue_fills() -> None:
    quality_gate = {
        "quality_gate_label": "queue_execution_publishable",
        "blocked_regimes": 0,
        "eligible_regimes": 2,
        "weighted_absolute_calibration_error": 0.04,
        "weighted_brier_score": 0.05,
        "max_regime_absolute_calibration_error": 0.06,
        "worst_calibration_regime": "open",
        "worst_decay_regime": "open",
    }
    capacity_stability = {
        "capacity_stability_label": "capacity_stable",
        "capacity_fraction_gap": -0.02,
        "capacity_edge_gap_ticks": -0.01,
        "capacity_tradable_share_gap": -0.01,
        "dominant_side_changed": False,
    }
    toxicity_surface = pd.DataFrame(
        {
            "regime": ["open", "stress", "stress"],
            "rows": [80, 15, 5],
            "adverse_to_fill_ratio": [0.20, 1.40, 0.90],
            "realized_loss_rate": [0.10, 0.80, 0.65],
            "mean_execution_adjusted_edge_ticks": [0.20, -0.30, -0.05],
            "queue_toxicity_label": [
                "benign_queue_fill",
                "toxic_queue_fill",
                "toxic_queue_fill",
            ],
        }
    )

    scorecard = queue_position_execution_readiness_scorecard(
        quality_gate,
        capacity_stability,
        toxicity_surface=toxicity_surface,
        max_toxic_queue_row_share=0.10,
    )

    assert scorecard["execution_blocker_count"] == 1
    assert scorecard["queue_toxicity_label"] == "toxic_queue_blocked"
    assert scorecard["toxic_queue_row_share"] == pytest.approx(0.20)
    assert scorecard["toxic_queue_regimes"] == 1
    assert scorecard["worst_toxicity_regime"] == "stress"
    assert scorecard["worst_toxicity_adverse_to_fill_ratio"] == pytest.approx(1.40)
    assert scorecard["worst_toxicity_realized_loss_rate"] == pytest.approx(0.80)
    assert scorecard["toxic_queue_mean_edge_ticks"] == pytest.approx(-0.2375)
    assert scorecard["execution_readiness_label"] == "execution_not_publishable"


def test_queue_position_execution_readiness_scorecard_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing queue execution readiness quality keys"):
        queue_position_execution_readiness_scorecard({"quality_gate_label": "x"}, {})


def test_execution_adjusted_lcri_regime_attribution_exposes_regime_side_survival() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["open", "open", "open", "stress", "stress"],
            "lcri": [2.0, 1.0, -2.0, 3.0, -1.0],
            "lcri_probability": [0.90, 0.80, 0.20, 0.95, 0.30],
            "best_execution_side": ["long", "abstain", "long", "short", "short"],
            "execution_adjusted_edge_ticks": [0.50, -0.10, 0.20, -0.40, 0.30],
            "bid_fill_probability": [0.80, 0.20, 0.70, 0.10, 0.20],
            "ask_fill_probability": [0.30, 0.40, 0.20, 0.90, 0.60],
            "bid_adverse_fill_probability": [0.10, 0.20, 0.30, 0.40, 0.20],
            "ask_adverse_fill_probability": [0.20, 0.30, 0.10, 0.50, 0.10],
        }
    )

    attribution = execution_adjusted_lcri_regime_attribution(frame)

    rows = attribution.set_index(["regime", "lcri_side"])
    open_long = rows.loc[("open", "long")]
    assert open_long["rows"] == 2
    assert open_long["tradable_rows"] == 1
    assert open_long["execution_conflict_rows"] == 1
    assert open_long["execution_survival_share"] == pytest.approx(0.50)
    assert open_long["mean_fill_probability_advantage"] == pytest.approx(0.15)
    assert open_long["review_label"] == "execution_friction_review"

    open_short = rows.loc[("open", "short")]
    assert open_short["dominant_execution_side"] == "long"
    assert open_short["execution_conflict_share"] == pytest.approx(1.00)
    assert open_short["review_label"] == "execution_side_inversion_review"

    stress_long = rows.loc[("stress", "long")]
    assert stress_long["review_label"] == "execution_side_inversion_review"
    assert stress_long["mean_execution_adjusted_edge_ticks"] == pytest.approx(-0.40)


def test_execution_adjusted_lcri_regime_attribution_rejects_missing_regime() -> None:
    with pytest.raises(ValueError, match="missing execution-adjusted LCRI regime attribution columns"):
        execution_adjusted_lcri_regime_attribution(pd.DataFrame({"lcri": [1.0]}))


def test_execution_adjusted_lcri_absorption_attribution_quantifies_absorbed_vs_transmitted_execution() -> None:
    frame = pd.DataFrame(
        {
            "absorption_regime": ["absorbed", "absorbed", "transmitted", "transmitted"],
            "publishable_side": ["long", "short", "long", "short"],
            "best_execution_side": ["abstain", "long", "long", "short"],
            "execution_adjusted_edge_ticks": [-0.20, 0.40, 0.60, 0.90],
            "bid_fill_probability": [0.20, 0.60, 0.70, 0.30],
            "ask_fill_probability": [0.50, 0.40, 0.20, 0.80],
            "bid_adverse_fill_probability": [0.10, 0.20, 0.10, 0.30],
            "ask_adverse_fill_probability": [0.30, 0.40, 0.20, 0.20],
        }
    )

    attribution = execution_adjusted_lcri_absorption_attribution(frame)

    rows = attribution.set_index("absorption_regime")
    absorbed = rows.loc["absorbed"]
    assert absorbed["rows"] == 2
    assert absorbed["publishable_rows"] == 2
    assert absorbed["executable_rows"] == 1
    assert absorbed["conflict_rows"] == 2
    assert absorbed["conflict_share"] == pytest.approx(1.00)
    assert absorbed["negative_edge_share"] == pytest.approx(0.50)
    assert absorbed["mean_execution_adjusted_edge_ticks"] == pytest.approx(0.10)
    assert absorbed["mean_selected_fill_probability"] == pytest.approx(0.30)
    assert absorbed["mean_selected_adverse_fill_probability"] == pytest.approx(0.10)
    assert absorbed["mean_fill_minus_adverse_probability"] == pytest.approx(0.20)
    assert absorbed["absorption_execution_label"] == "absorption_execution_conflicted"

    transmitted = rows.loc["transmitted"]
    assert transmitted["executable_rows"] == 2
    assert transmitted["conflict_share"] == pytest.approx(0.00)
    assert transmitted["negative_edge_share"] == pytest.approx(0.00)
    assert transmitted["mean_selected_fill_probability"] == pytest.approx(0.75)
    assert transmitted["mean_selected_adverse_fill_probability"] == pytest.approx(0.15)
    assert transmitted["absorption_execution_label"] == "absorption_execution_publishable"


def test_execution_adjusted_lcri_absorption_attribution_flags_absorbed_toxicity() -> None:
    frame = pd.DataFrame(
        {
            "absorption_regime": ["absorbed", "absorbed", "absorbed", "absorbed"],
            "publishable_side": ["long", "long", "short", "short"],
            "best_execution_side": ["long", "short", "short", "long"],
            "execution_adjusted_edge_ticks": [-0.50, -0.10, 0.20, -0.20],
            "bid_fill_probability": [0.70, 0.60, 0.20, 0.50],
            "ask_fill_probability": [0.20, 0.50, 0.60, 0.30],
            "bid_adverse_fill_probability": [0.80, 0.70, 0.20, 0.60],
            "ask_adverse_fill_probability": [0.20, 0.80, 0.70, 0.30],
        }
    )

    attribution = execution_adjusted_lcri_absorption_attribution(
        frame,
        max_negative_edge_share=0.50,
        min_fill_minus_adverse_probability=0.0,
    )

    assert attribution.loc[0, "negative_edge_share"] == pytest.approx(0.75)
    assert attribution.loc[0, "mean_fill_minus_adverse_probability"] == pytest.approx(-0.15)
    assert attribution.loc[0, "absorption_execution_label"] == "absorption_execution_toxic"


def test_execution_adjusted_lcri_absorption_attribution_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing execution-adjusted LCRI absorption attribution columns"):
        execution_adjusted_lcri_absorption_attribution(pd.DataFrame({"absorption_regime": ["absorbed"]}))


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


def test_passive_fill_brier_decomposition_separates_reliability_and_resolution() -> None:
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

    decomposition = passive_fill_brier_decomposition(curve)

    assert decomposition == {
        "rows": 10,
        "bins": 3,
        "base_fill_rate": pytest.approx(0.50),
        "weighted_brier_score": pytest.approx(0.152),
        "uncertainty": pytest.approx(0.25),
        "reliability": pytest.approx(0.05),
        "resolution": pytest.approx(0.13),
        "brier_skill_score": pytest.approx(1.0 - 0.152 / 0.25),
        "brier_decomposition_error": pytest.approx(0.152 - (0.05 - 0.13 + 0.25)),
        "calibration_quality_label": "resolved_but_needs_calibration",
    }


def test_passive_fill_brier_decomposition_labels_resolved_skill() -> None:
    curve = pd.DataFrame(
        {
            "regime": ["front", "back"],
            "bin": [1, 2],
            "rows": [5, 5],
            "mean_predicted_fill_probability": [0.10, 0.90],
            "realized_fill_rate": [0.10, 0.90],
            "calibration_error": [0.00, 0.00],
            "absolute_calibration_error": [0.00, 0.00],
            "brier_score": [0.09, 0.09],
        }
    )

    decomposition = passive_fill_brier_decomposition(curve)

    assert decomposition["base_fill_rate"] == pytest.approx(0.50)
    assert decomposition["reliability"] == pytest.approx(0.0)
    assert decomposition["resolution"] == pytest.approx(0.16)
    assert decomposition["brier_skill_score"] == pytest.approx(0.64)
    assert decomposition["calibration_quality_label"] == "resolved_calibrated_skill"


def test_passive_fill_brier_decomposition_rejects_negative_rows() -> None:
    curve = pd.DataFrame(
        {
            "regime": ["thin"],
            "bin": [1],
            "rows": [-1],
            "mean_predicted_fill_probability": [0.30],
            "realized_fill_rate": [0.00],
            "calibration_error": [-0.30],
            "absolute_calibration_error": [0.30],
            "brier_score": [0.10],
        }
    )

    with pytest.raises(ValueError, match="brier decomposition rows must be non-negative"):
        passive_fill_brier_decomposition(curve)


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


def test_queue_position_calibration_residual_summary_ranks_underfilled_queue_slices() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["thin", "thin", "thin", "stress"],
            "best_execution_side": ["long", "long", "short", "long"],
            "queue_share_bin": [1, 2, 1, 1],
            "fill_probability_bin": [2, 1, 2, 2],
            "rows": [10, 30, 20, 5],
            "mean_queue_share": [0.20, 0.80, 0.30, 0.25],
            "mean_predicted_fill_probability": [0.80, 0.60, 0.30, 0.70],
            "realized_fill_rate": [0.70, 0.20, 0.55, 0.65],
            "calibration_error": [-0.10, -0.40, 0.25, -0.05],
            "absolute_calibration_error": [0.10, 0.40, 0.25, 0.05],
            "brier_score": [0.18, 0.30, 0.20, 0.12],
            "mean_execution_adjusted_edge_ticks": [0.40, -0.30, 0.20, 0.10],
        }
    )

    summary = queue_position_calibration_residual_summary(surface, error_threshold=0.15)

    assert summary["regime"].tolist() == ["thin", "thin", "stress"]
    assert summary["best_execution_side"].tolist() == ["long", "short", "long"]
    assert summary["rows"].tolist() == [40, 20, 5]
    assert summary["underfilled_bins"].tolist() == [1, 0, 0]
    assert summary["overfilled_bins"].tolist() == [0, 1, 0]
    assert summary["weighted_calibration_error"].tolist() == pytest.approx([-0.325, 0.25, -0.05])
    assert summary["weighted_absolute_calibration_error"].tolist() == pytest.approx([0.325, 0.25, 0.05])
    assert summary["weighted_mean_execution_adjusted_edge_ticks"].tolist() == pytest.approx(
        [-0.125, 0.20, 0.10]
    )
    assert summary["worst_queue_share_bin"].tolist() == [2, 1, 1]
    assert summary["residual_label"].tolist() == [
        "underfilled_execution_drag",
        "overfilled_execution_opportunity",
        "calibration_residual_controlled",
    ]


def test_queue_position_calibration_residual_summary_rejects_bad_surface() -> None:
    with pytest.raises(ValueError, match="error_threshold"):
        queue_position_calibration_residual_summary(pd.DataFrame(), error_threshold=-0.1)
    with pytest.raises(ValueError, match="missing queue position calibration residual summary columns"):
        queue_position_calibration_residual_summary(pd.DataFrame({"regime": ["thin"]}))


def test_queue_position_calibration_stability_joins_holdout_queue_cells() -> None:
    research = pd.DataFrame(
        {
            "regime": ["calm", "thin"],
            "best_execution_side": ["long", "short"],
            "queue_share_bin": [1, 2],
            "fill_probability_bin": [1, 2],
            "rows": [20, 10],
            "realized_fill_rate": [0.70, 0.40],
            "calibration_error": [0.05, -0.10],
            "absolute_calibration_error": [0.05, 0.10],
            "brier_score": [0.10, 0.20],
            "mean_execution_adjusted_edge_ticks": [0.30, 0.10],
        }
    )
    heldout = pd.DataFrame(
        {
            "regime": ["calm", "thin", "stress"],
            "best_execution_side": ["long", "short", "long"],
            "queue_share_bin": [1, 2, 1],
            "fill_probability_bin": [1, 2, 1],
            "rows": [15, 12, 8],
            "realized_fill_rate": [0.68, 0.15, 0.50],
            "calibration_error": [0.06, -0.35, -0.05],
            "absolute_calibration_error": [0.06, 0.35, 0.05],
            "brier_score": [0.11, 0.45, 0.16],
            "mean_execution_adjusted_edge_ticks": [0.28, -0.20, 0.05],
        }
    )

    stability = queue_position_calibration_stability(research, heldout, max_error_gap=0.10)

    assert stability["calibration_stability_label"].tolist() == [
        "calibration_replicated",
        "calibration_cell_gained",
        "calibration_degraded",
    ]
    degraded = stability[stability["regime"] == "thin"].iloc[0]
    assert degraded["absolute_calibration_error_gap"] == pytest.approx(0.25)
    assert degraded["heldout_mean_execution_adjusted_edge_ticks"] == pytest.approx(-0.20)


def test_queue_position_calibration_stability_summary_flags_degraded_holdout_cells() -> None:
    stability = pd.DataFrame(
        {
            "regime": ["calm", "thin", "stress"],
            "best_execution_side": ["long", "short", "long"],
            "queue_share_bin": [1, 2, 1],
            "fill_probability_bin": [1, 2, 1],
            "research_rows": [20, 10, 0],
            "heldout_rows": [15, 12, 8],
            "realized_fill_rate_gap": [-0.02, -0.25, 0.0],
            "calibration_error_gap": [0.01, -0.25, 0.0],
            "absolute_calibration_error_gap": [0.01, 0.25, 0.05],
            "brier_score_gap": [0.01, 0.25, 0.16],
            "execution_adjusted_edge_gap_ticks": [-0.02, -0.30, 0.05],
            "calibration_stability_label": [
                "calibration_replicated",
                "calibration_degraded",
                "calibration_cell_gained",
            ],
        }
    )

    summary = queue_position_calibration_stability_summary(stability)

    assert summary["common_cells"] == 2
    assert summary["degraded_cells"] == 1
    assert summary["degraded_cell_share"] == pytest.approx(1 / 3)
    assert summary["worst_regime"] == "thin"
    assert summary["queue_calibration_stability_label"] == "queue_calibration_degraded"


def test_queue_position_calibration_reliability_scorecard_blocks_fragile_fill_model() -> None:
    residual = pd.DataFrame(
        {
            "regime": ["thin", "calm"],
            "best_execution_side": ["long", "short"],
            "bins": [2, 1],
            "rows": [40, 20],
            "underfilled_bins": [1, 0],
            "overfilled_bins": [0, 0],
            "weighted_calibration_error": [-0.32, 0.04],
            "weighted_absolute_calibration_error": [0.32, 0.04],
            "weighted_mean_execution_adjusted_edge_ticks": [-0.15, 0.20],
            "residual_label": ["underfilled_execution_drag", "calibration_residual_controlled"],
        }
    )
    drift = pd.DataFrame(
        {
            "best_execution_side": ["long", "short"],
            "queue_share_bin": [2, 1],
            "fill_probability_bin": [1, 2],
            "regimes": [3, 2],
            "rows": [80, 20],
            "fill_rate_range": [0.31, 0.04],
            "calibration_error_range": [0.20, 0.02],
            "weighted_mean_absolute_calibration_error": [0.22, 0.03],
            "worst_regime": ["thin", "calm"],
            "drift_label": ["calibration_unstable", "calibration_stable"],
        }
    )
    stability_summary = {
        "cells": 4,
        "common_cells": 3,
        "replicated_cells": 2,
        "degraded_cells": 1,
        "lost_cells": 0,
        "gained_cells": 1,
        "degraded_cell_share": 0.25,
        "mean_absolute_calibration_error_gap": 0.12,
        "worst_regime": "thin",
        "worst_best_execution_side": "long",
        "worst_queue_share_bin": 2,
        "worst_fill_probability_bin": 1,
        "worst_calibration_stability_label": "calibration_degraded",
        "queue_calibration_stability_label": "queue_calibration_degraded",
    }

    scorecard = queue_position_calibration_reliability_scorecard(
        residual,
        drift,
        stability_summary=stability_summary,
        max_weighted_abs_error=0.20,
        max_unstable_drift_share=0.25,
    )

    assert scorecard["residual_slices"] == 2
    assert scorecard["underfilled_execution_drag_slices"] == 1
    assert scorecard["unstable_drift_bins"] == 1
    assert scorecard["unstable_drift_share"] == pytest.approx(0.50)
    assert scorecard["degraded_stability_cells"] == 1
    assert scorecard["worst_residual_regime"] == "thin"
    assert scorecard["worst_drift_label"] == "calibration_unstable"
    assert scorecard["queue_calibration_reliability_label"] == "queue_calibration_underfill_block"


def test_queue_position_calibration_reliability_scorecard_passes_clean_evidence() -> None:
    residual = pd.DataFrame(
        {
            "regime": ["calm"],
            "best_execution_side": ["long"],
            "bins": [2],
            "rows": [50],
            "underfilled_bins": [0],
            "overfilled_bins": [0],
            "weighted_calibration_error": [0.01],
            "weighted_absolute_calibration_error": [0.04],
            "weighted_mean_execution_adjusted_edge_ticks": [0.30],
            "residual_label": ["calibration_residual_controlled"],
        }
    )
    drift = pd.DataFrame(
        {
            "best_execution_side": ["long"],
            "queue_share_bin": [1],
            "fill_probability_bin": [1],
            "regimes": [2],
            "rows": [50],
            "fill_rate_range": [0.03],
            "calibration_error_range": [0.02],
            "weighted_mean_absolute_calibration_error": [0.04],
            "worst_regime": ["calm"],
            "drift_label": ["calibration_stable"],
        }
    )

    scorecard = queue_position_calibration_reliability_scorecard(
        residual,
        drift,
        stability_summary={"queue_calibration_stability_label": "queue_calibration_replicated"},
    )

    assert scorecard["queue_calibration_reliability_label"] == "queue_calibration_release_ready"
    assert scorecard["max_weighted_absolute_calibration_error"] == pytest.approx(0.04)
    assert scorecard["max_fill_rate_range"] == pytest.approx(0.03)


def test_queue_position_calibration_reliability_scorecard_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="max_weighted_abs_error"):
        queue_position_calibration_reliability_scorecard(
            pd.DataFrame(), pd.DataFrame(), max_weighted_abs_error=-0.1
        )
    with pytest.raises(ValueError, match="missing queue position calibration reliability residual columns"):
        queue_position_calibration_reliability_scorecard(pd.DataFrame({"regime": ["thin"]}), pd.DataFrame())


def test_queue_position_calibration_stability_rejects_bad_surface() -> None:
    with pytest.raises(ValueError, match="max_error_gap"):
        queue_position_calibration_stability(pd.DataFrame(), pd.DataFrame(), max_error_gap=-0.1)
    with pytest.raises(ValueError, match="missing research queue position calibration stability columns"):
        queue_position_calibration_stability(pd.DataFrame({"regime": ["thin"]}), pd.DataFrame())


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


def test_passive_fill_event_window_diagnostics_respects_group_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "best_execution_side": ["long", "long", "long", "long"],
            "regime": ["calm", "calm", "stress", "stress"],
            "bid_fill_probability": [0.10, 0.95, 0.95, 0.10],
            "ask_fill_probability": [0.10, 0.10, 0.10, 0.10],
            "bid_adverse_fill_probability": [0.05, 0.30, 0.40, 0.05],
            "ask_adverse_fill_probability": [0.05, 0.05, 0.05, 0.05],
            "execution_adjusted_edge_ticks": [0.10, 0.80, 0.90, 0.20],
            "long_net_return_ticks": [0.50, 0.60, -1.00, -1.20],
            "short_net_return_ticks": [0.0, 0.0, 0.0, 0.0],
        }
    )

    events = passive_fill_event_window_diagnostics(
        frame,
        threshold=0.90,
        window=1,
        regime_col="regime",
        group_cols="symbol",
    )

    assert events["event_index"].tolist() == [1, 2]
    assert events["window_rows"].tolist() == [2, 2]
    assert events["pre_realized_edge_sum"].tolist() == pytest.approx([0.50, 0.0])
    assert events["post_realized_edge_sum"].tolist() == pytest.approx([0.0, -1.20])
    assert events["regime_transition"].tolist() == ["calm->calm", "stress->stress"]


def test_add_passive_fill_event_window_regimes_labels_execution_toxicity_neighborhoods() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "long", "short", "short", "long", "long"],
            "bid_fill_probability": [0.20, 0.92, 0.10, 0.20, 0.91, 0.10],
            "ask_fill_probability": [0.10, 0.15, 0.88, 0.10, 0.20, 0.20],
            "bid_adverse_fill_probability": [0.05, 0.55, 0.10, 0.10, 0.20, 0.05],
            "ask_adverse_fill_probability": [0.05, 0.10, 0.52, 0.10, 0.10, 0.05],
        }
    )

    output = add_passive_fill_event_window_regimes(frame, threshold=0.85, window=1)

    assert output["passive_fill_event_window_regime"].tolist() == [
        "pre_event",
        "event",
        "event",
        "post_event",
        "event",
        "post_event",
    ]
    assert output["passive_fill_event_distance"].tolist() == pytest.approx([-1, 0, 0, 1, 0, 1])
    assert output["passive_fill_event_side"].tolist() == ["long", "long", "short", "short", "long", "long"]
    assert output["passive_fill_event_fill_probability"].tolist() == pytest.approx(
        [0.92, 0.92, 0.88, 0.88, 0.91, 0.91]
    )
    assert output["passive_fill_event_toxicity_probability"].tolist() == pytest.approx(
        [0.55, 0.55, 0.52, 0.52, 0.20, 0.20]
    )


def test_add_passive_fill_event_window_regimes_respects_group_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "best_execution_side": ["long", "long", "long", "long"],
            "bid_fill_probability": [0.10, 0.95, 0.95, 0.10],
            "ask_fill_probability": [0.10, 0.10, 0.10, 0.10],
            "bid_adverse_fill_probability": [0.05, 0.40, 0.60, 0.05],
            "ask_adverse_fill_probability": [0.05, 0.05, 0.05, 0.05],
        }
    )

    output = add_passive_fill_event_window_regimes(
        frame, threshold=0.90, window=1, group_cols="symbol"
    )

    assert output["passive_fill_event_window_regime"].tolist() == [
        "pre_event",
        "event",
        "event",
        "post_event",
    ]
    assert output["passive_fill_event_distance"].tolist() == pytest.approx([-1, 0, 0, 1])
    assert output["passive_fill_event_fill_probability"].tolist() == pytest.approx(
        [0.95, 0.95, 0.95, 0.95]
    )


def test_passive_fill_event_window_regime_summary_ranks_toxic_executable_neighborhoods() -> None:
    frame = pd.DataFrame(
        {
            "passive_fill_event_window_regime": [
                "pre_event",
                "event",
                "post_event",
                "post_event",
                "calm",
            ],
            "passive_fill_event_side": ["long", "long", "long", "short", "none"],
            "passive_fill_event_fill_probability": [0.90, 0.90, 0.90, 0.88, 0.0],
            "passive_fill_event_toxicity_probability": [0.40, 0.40, 0.40, 0.70, 0.0],
            "execution_adjusted_edge_ticks": [0.10, 0.25, -0.50, -0.70, 0.05],
        }
    )

    summary = passive_fill_event_window_regime_summary(frame)

    assert summary["passive_fill_event_window_regime"].tolist() == [
        "post_event",
        "event",
        "pre_event",
        "calm",
    ]
    post_event = summary.iloc[0]
    assert post_event["rows"] == 2
    assert post_event["event_rows"] == 0
    assert post_event["row_share"] == pytest.approx(0.40)
    assert post_event["mean_passive_fill_event_fill_probability"] == pytest.approx(0.89)
    assert post_event["mean_passive_fill_event_toxicity_probability"] == pytest.approx(0.55)
    assert post_event["mean_execution_adjusted_edge_ticks"] == pytest.approx(-0.60)
    assert post_event["negative_edge_share"] == pytest.approx(1.0)
    assert post_event["dominant_passive_fill_event_side"] == "long"


def test_passive_fill_event_lead_lag_profile_respects_group_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "best_execution_side": ["long", "long", "long", "long"],
            "regime": ["calm", "calm", "stress", "stress"],
            "bid_fill_probability": [0.10, 0.95, 0.95, 0.10],
            "ask_fill_probability": [0.10, 0.10, 0.10, 0.10],
            "long_net_return_ticks": [0.50, 0.60, -1.00, -1.20],
            "short_net_return_ticks": [0.0, 0.0, 0.0, 0.0],
        }
    )

    profile = passive_fill_event_lead_lag_profile(
        frame,
        threshold=0.90,
        window=1,
        regime_col="regime",
        group_cols="symbol",
    )

    assert profile["event_regime"].tolist() == ["calm", "calm", "stress", "stress"]
    assert profile["relative_offset"].tolist() == [-1, 0, 0, 1]
    assert profile["observations"].tolist() == [1, 1, 1, 1]
    assert profile["mean_realized_edge_ticks"].tolist() == pytest.approx([0.50, 0.60, -1.00, -1.20])


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


def test_passive_fill_event_window_sensitivity_sweeps_threshold_publishability() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["thin", "thin", "thin", "thin", "thin"],
            "best_execution_side": ["long", "long", "long", "long", "long"],
            "bid_fill_probability": [0.20, 0.90, 0.30, 0.75, 0.30],
            "ask_fill_probability": [0.10, 0.10, 0.10, 0.10, 0.10],
            "bid_adverse_fill_probability": [0.10, 0.50, 0.10, 0.20, 0.10],
            "ask_adverse_fill_probability": [0.10, 0.10, 0.10, 0.10, 0.10],
            "execution_adjusted_edge_ticks": [0.20, 0.50, -0.60, 0.40, 0.40],
            "long_net_return_ticks": [0.20, 0.50, -0.60, 0.40, 0.40],
            "short_net_return_ticks": [-0.20, -0.50, 0.60, -0.40, -0.40],
        }
    )

    sensitivity = passive_fill_event_window_sensitivity(
        frame,
        thresholds=(0.70, 0.80),
        windows=(1,),
        regime_col="regime",
        max_adverse_post_edge_share=0.60,
        min_mean_post_minus_pre_edge=-0.25,
    )

    assert sensitivity["threshold"].tolist() == pytest.approx([0.70, 0.80])
    assert sensitivity["window"].tolist() == [1, 1]
    assert sensitivity["total_events"].tolist() == [2, 1]
    assert sensitivity["event_toxicity_label"].tolist() == [
        "event_window_pass",
        "event_window_blocker",
    ]
    assert sensitivity["weighted_mean_post_minus_pre_realized_edge"].tolist() == pytest.approx(
        [0.10, -0.80]
    )
    assert sensitivity["sensitivity_label"].tolist() == [
        "event_window_threshold_pass",
        "event_window_threshold_blocker",
    ]


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


def test_queue_position_latency_sensitivity_audits_stale_execution_decisions() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "B", "B"],
            "best_execution_side": ["long", "long", "short", "short", "long"],
            "bid_fill_probability": [0.80, 0.60, 0.30, 0.20, 0.90],
            "ask_fill_probability": [0.20, 0.40, 0.70, 0.75, 0.10],
            "bid_realized_fill": [1.0, 0.0, 1.0, 0.0, 1.0],
            "ask_realized_fill": [0.0, 1.0, 1.0, 1.0, 0.0],
            "execution_adjusted_edge_ticks": [0.20, 0.10, 0.30, 0.40, 0.50],
        }
    )

    curve = queue_position_latency_sensitivity(frame, latencies=[0, 1], group_cols="symbol")

    assert curve["latency_steps"].tolist() == [0, 1]
    assert curve["candidates"].tolist() == [5, 3]
    assert curve["long_candidates"].tolist() == [3, 2]
    assert curve["short_candidates"].tolist() == [2, 1]
    assert curve["mean_decision_fill_probability"].tolist() == pytest.approx([0.75, (0.80 + 0.60 + 0.75) / 3.0])
    assert curve["realized_fill_rate"].tolist() == pytest.approx([0.80, 1.0 / 3.0])
    assert curve["realized_fill_gap_vs_immediate"].tolist() == pytest.approx([0.0, -0.80 + (1.0 / 3.0)])
    assert curve["mean_execution_adjusted_edge_ticks"].tolist() == pytest.approx([0.30, (0.20 + 0.10 + 0.40) / 3.0])
    assert curve["latency_label"].tolist() == ["anchor_latency", "latency_fragile"]


def test_queue_position_latency_sensitivity_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="latencies must be non-negative integers"):
        queue_position_latency_sensitivity(pd.DataFrame(), latencies=[0, -1])


def test_queue_position_latency_edge_survival_prices_delayed_fill_decay() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "B", "B"],
            "best_execution_side": ["long", "long", "short", "short", "long"],
            "bid_realized_fill": [1.0, 0.0, 1.0, 0.0, 1.0],
            "ask_realized_fill": [0.0, 1.0, 1.0, 1.0, 0.0],
            "execution_adjusted_edge_ticks": [0.20, 0.10, 0.30, 0.40, 0.50],
        }
    )

    survival = queue_position_latency_edge_survival(
        frame,
        latencies=[0, 1],
        group_cols="symbol",
        max_realized_edge_decay=0.10,
    )

    assert survival["latency_steps"].tolist() == [0, 1]
    assert survival["candidates"].tolist() == [5, 3]
    assert survival["realized_fill_rate"].tolist() == pytest.approx([0.80, 1.0 / 3.0])
    assert survival["mean_decision_edge_ticks"].tolist() == pytest.approx([0.30, (0.20 + 0.10 + 0.40) / 3.0])
    assert survival["realized_edge_ticks"].tolist() == pytest.approx([0.28, 0.10 / 3.0])
    assert survival["realized_edge_gap_vs_immediate"].tolist() == pytest.approx([0.0, (0.10 / 3.0) - 0.28])
    assert survival["edge_survival_ratio"].tolist() == pytest.approx([1.0, (0.10 / 3.0) / 0.28])
    assert survival["edge_latency_label"].tolist() == ["anchor_latency", "edge_latency_fragile"]


def test_queue_position_latency_edge_survival_rejects_missing_edge_state() -> None:
    with pytest.raises(ValueError, match="missing queue position latency edge survival columns"):
        queue_position_latency_edge_survival(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_latency_edge_survival_scorecard_blocks_lost_edge() -> None:
    survival = pd.DataFrame(
        {
            "latency_steps": [0, 1, 2],
            "candidates": [10, 8, 6],
            "realized_edge_ticks": [0.30, 0.18, 0.05],
            "realized_edge_gap_vs_immediate": [0.0, -0.12, -0.25],
            "edge_survival_ratio": [1.0, 0.60, 1.0 / 6.0],
            "edge_latency_label": [
                "anchor_latency",
                "edge_latency_fragile",
                "edge_latency_fragile",
            ],
        }
    )

    scorecard = queue_position_latency_edge_survival_scorecard(
        survival,
        max_fragile_edge_candidate_share=0.50,
        review_fragile_edge_candidate_share=0.20,
        min_candidate_weighted_edge_gap=-0.20,
        review_candidate_weighted_edge_gap=-0.05,
        min_weighted_edge_survival_ratio=0.50,
    )

    assert scorecard["anchor_edge_ticks"] == pytest.approx(0.30)
    assert scorecard["latency_candidates"] == 14
    assert scorecard["fragile_edge_candidate_share"] == pytest.approx(1.0)
    assert scorecard["candidate_weighted_edge_gap"] == pytest.approx(
        ((8 * -0.12) + (6 * -0.25)) / 14
    )
    assert scorecard["candidate_weighted_edge_survival_ratio"] == pytest.approx(
        ((8 * 0.60) + (6 * (1.0 / 6.0))) / 14
    )
    assert scorecard["worst_latency_steps"] == 2
    assert scorecard["edge_survival_release_decision"] == "block"
    assert scorecard["edge_survival_release_label"] == "queue_latency_edge_survival_blocked"
    assert "fragile_edge_candidate_share" in scorecard["blocking_reasons"]


def test_queue_position_latency_edge_survival_scorecard_reviews_empty_input() -> None:
    scorecard = queue_position_latency_edge_survival_scorecard(pd.DataFrame())

    assert scorecard["edge_survival_release_decision"] == "review"
    assert scorecard["edge_survival_release_label"] == "queue_latency_edge_survival_no_evidence"
    assert scorecard["review_reasons"] == "no_latency_edge_evidence"


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


def test_execution_publishability_release_gate_blocks_regime_capacity_loss() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long"],
            "best_execution_side": ["long"],
            "rows": [100],
            "conflict_rows": [0],
            "review_priority": [0],
        }
    )
    quality_gate = {"quality_gate_label": "queue_execution_publishable"}
    capacity_stability = {"capacity_stability_label": "capacity_stable"}
    regime_capacity_stability = {
        "regime_capacity_stability_label": "regime_capacity_not_replicated",
        "lost_capacity_regimes": 1,
        "stable_regime_share": 0.5,
        "worst_regime": "stress",
    }

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate=quality_gate,
        capacity_stability=capacity_stability,
        regime_capacity_stability=regime_capacity_stability,
    )

    assert gate["decision"] == "block"
    assert gate["passes"] is False
    assert gate["capacity_stability_label"] == "capacity_stable"
    assert gate["regime_capacity_stability_label"] == "regime_capacity_not_replicated"
    assert gate["lost_capacity_regimes"] == 1
    assert gate["stable_regime_share"] == pytest.approx(0.5)
    assert gate["worst_capacity_regime"] == "stress"
    assert "regime_capacity_not_replicated" in gate["blocking_reasons"]


def test_execution_publishability_release_gate_blocks_lcri_regime_survival_loss() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long"],
            "best_execution_side": ["long"],
            "rows": [100],
            "conflict_rows": [0],
            "review_priority": [0],
        }
    )
    quality_gate = {"quality_gate_label": "queue_execution_publishable"}
    capacity_stability = {"capacity_stability_label": "capacity_stable"}
    lcri_regime_attribution = pd.DataFrame(
        {
            "regime": ["open", "stress", "stress", "open"],
            "lcri_side": ["long", "long", "short", "neutral"],
            "rows": [50, 30, 20, 10],
            "execution_survival_share": [0.72, 0.20, 0.55, 0.0],
            "execution_conflict_share": [0.10, 0.80, 0.25, 0.0],
        }
    )

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate=quality_gate,
        capacity_stability=capacity_stability,
        lcri_regime_attribution=lcri_regime_attribution,
        min_lcri_regime_survival_share=0.50,
        max_lcri_regime_conflict_share=0.40,
    )

    assert gate["decision"] == "block"
    assert gate["passes"] is False
    assert gate["lcri_regime_survival_label"] == "lcri_regime_execution_not_preserved"
    assert gate["weak_lcri_regime_sides"] == 1
    assert gate["worst_lcri_regime"] == "stress"
    assert gate["worst_lcri_side"] == "long"
    assert gate["min_lcri_execution_survival_share"] == pytest.approx(0.20)
    assert gate["max_lcri_execution_conflict_share"] == pytest.approx(0.80)
    assert "lcri_regime_execution_not_preserved" in gate["blocking_reasons"]


def test_execution_publishability_release_gate_blocks_lcri_event_window_toxicity() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long"],
            "best_execution_side": ["long"],
            "rows": [100],
            "conflict_rows": [0],
            "review_priority": [0],
        }
    )
    event_window_scorecard = {
        "high_lcri_rows": 40,
        "toxic_high_lcri_row_share": 0.50,
        "event_toxic_high_lcri_row_share": 0.75,
        "weighted_high_lcri_signal_survival_ratio": 0.25,
        "weighted_high_lcri_fill_adverse_spread": -0.10,
        "worst_event_window_regime": "event",
        "worst_event_window_bucket": "high_abs_lcri",
        "worst_event_window_label": "high_lcri_event_toxicity",
        "release_decision": "block",
        "release_label": "execution_lcri_event_window_blocked",
        "blocking_reasons": "event_toxic_high_lcri_share;low_signal_survival",
        "review_reasons": "none",
    }

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate={"quality_gate_label": "queue_execution_publishable"},
        capacity_stability={"capacity_stability_label": "capacity_stable"},
        lcri_event_window_scorecard=event_window_scorecard,
    )

    assert gate["decision"] == "block"
    assert gate["passes"] is False
    assert "execution_lcri_event_window_blocked" in gate["blocking_reasons"]
    assert "event_toxic_high_lcri_share" in gate["blocking_reasons"]
    assert gate["lcri_event_window_release_label"] == "execution_lcri_event_window_blocked"
    assert gate["lcri_event_window_high_lcri_rows"] == 40
    assert gate["lcri_event_window_toxic_high_lcri_row_share"] == pytest.approx(0.50)
    assert gate["lcri_event_window_event_toxic_high_lcri_row_share"] == pytest.approx(0.75)
    assert gate["lcri_event_window_signal_survival_ratio"] == pytest.approx(0.25)
    assert gate["lcri_event_window_fill_adverse_spread"] == pytest.approx(-0.10)
    assert gate["worst_lcri_event_window_regime"] == "event"
    assert gate["worst_lcri_event_window_bucket"] == "high_abs_lcri"
    assert gate["worst_lcri_event_window_label"] == "high_lcri_event_toxicity"


def test_execution_publishability_release_gate_reviews_lcri_event_window_evidence_gap() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long"],
            "best_execution_side": ["long"],
            "rows": [100],
            "conflict_rows": [0],
            "review_priority": [0],
        }
    )
    event_window_scorecard = {
        "high_lcri_rows": 0,
        "toxic_high_lcri_row_share": 0.0,
        "event_toxic_high_lcri_row_share": 0.0,
        "weighted_high_lcri_signal_survival_ratio": 0.0,
        "weighted_high_lcri_fill_adverse_spread": 0.0,
        "worst_event_window_regime": "none",
        "worst_event_window_bucket": "none",
        "worst_event_window_label": "none",
        "release_decision": "review",
        "release_label": "execution_lcri_event_window_review",
        "blocking_reasons": "none",
        "review_reasons": "no_high_lcri_rows",
    }

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate={"quality_gate_label": "queue_execution_publishable"},
        capacity_stability={"capacity_stability_label": "capacity_stable"},
        lcri_event_window_scorecard=event_window_scorecard,
    )

    assert gate["decision"] == "review"
    assert gate["passes"] is False
    assert gate["lcri_event_window_release_label"] == "execution_lcri_event_window_review"
    assert "execution_lcri_event_window_review" in gate["review_reasons"]
    assert "no_high_lcri_rows" in gate["review_reasons"]


def test_execution_publishability_release_gate_rejects_malformed_lcri_event_window_scorecard() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long"],
            "best_execution_side": ["long"],
            "rows": [100],
            "conflict_rows": [0],
            "review_priority": [0],
        }
    )

    with pytest.raises(ValueError, match="execution publishability LCRI event-window"):
        execution_publishability_release_gate(
            review_packet,
            quality_gate={"quality_gate_label": "queue_execution_publishable"},
            capacity_stability={"capacity_stability_label": "capacity_stable"},
            lcri_event_window_scorecard={"release_decision": "block"},
        )


def test_execution_publishability_release_gate_blocks_latency_fragile_queue_evidence() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long"],
            "best_execution_side": ["long"],
            "rows": [100],
            "conflict_rows": [0],
            "review_priority": [0],
        }
    )
    quality_gate = {"quality_gate_label": "queue_execution_publishable"}
    capacity_stability = {"capacity_stability_label": "capacity_stable"}
    latency_sensitivity = pd.DataFrame(
        {
            "latency_steps": [0, 1, 2],
            "candidates": [100, 80, 35],
            "realized_fill_gap_vs_immediate": [0.0, -0.04, -0.22],
            "latency_label": ["anchor_latency", "latency_robust", "latency_fragile"],
        }
    )

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate=quality_gate,
        capacity_stability=capacity_stability,
        latency_sensitivity=latency_sensitivity,
        max_latency_fill_decay=0.10,
        min_latency_candidate_retention_share=0.50,
    )

    assert gate["decision"] == "block"
    assert gate["passes"] is False
    assert gate["latency_sensitivity_label"] == "queue_latency_fragile"
    assert gate["worst_latency_steps"] == 2
    assert gate["worst_latency_fill_gap"] == pytest.approx(-0.22)
    assert gate["min_latency_candidate_retention_share"] == pytest.approx(0.35)
    assert "queue_latency_fragile" in gate["blocking_reasons"]


def test_execution_publishability_release_gate_blocks_bad_fill_calibration_skill() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long"],
            "best_execution_side": ["long"],
            "rows": [100],
            "conflict_rows": [0],
            "review_priority": [0],
        }
    )
    quality_gate = {"quality_gate_label": "queue_execution_publishable"}
    capacity_stability = {"capacity_stability_label": "capacity_stable"}
    fill_brier_decomposition = {
        "calibration_quality_label": "worse_than_base_rate",
        "brier_skill_score": -0.12,
        "reliability": 0.08,
        "resolution": 0.01,
    }

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate=quality_gate,
        capacity_stability=capacity_stability,
        fill_brier_decomposition=fill_brier_decomposition,
    )

    assert gate["decision"] == "block"
    assert gate["passes"] is False
    assert gate["fill_calibration_label"] == "worse_than_base_rate"
    assert gate["fill_brier_skill_score"] == pytest.approx(-0.12)
    assert gate["fill_calibration_reliability"] == pytest.approx(0.08)
    assert gate["fill_calibration_resolution"] == pytest.approx(0.01)
    assert "worse_than_base_rate" in gate["blocking_reasons"]


def test_execution_publishability_release_gate_reviews_low_resolution_fill_calibration() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long"],
            "best_execution_side": ["long"],
            "rows": [100],
            "conflict_rows": [0],
            "review_priority": [0],
        }
    )
    quality_gate = {"quality_gate_label": "queue_execution_publishable"}
    capacity_stability = {"capacity_stability_label": "capacity_stable"}
    fill_brier_decomposition = {
        "calibration_quality_label": "low_resolution",
        "brier_skill_score": 0.01,
        "reliability": 0.01,
        "resolution": 0.005,
    }

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate=quality_gate,
        capacity_stability=capacity_stability,
        fill_brier_decomposition=fill_brier_decomposition,
    )

    assert gate["decision"] == "review"
    assert gate["passes"] is False
    assert gate["fill_calibration_label"] == "low_resolution"
    assert "low_resolution" in gate["review_reasons"]


def test_execution_publishability_release_gate_blocks_fragile_path_risk() -> None:
    review_packet = pd.DataFrame(
        {
            "publishable_side": ["long"],
            "best_execution_side": ["long"],
            "rows": [100],
            "conflict_rows": [0],
            "review_priority": [0],
        }
    )
    quality_gate = {"quality_gate_label": "queue_execution_publishable"}
    capacity_stability = {"capacity_stability_label": "capacity_stable"}
    path_risk = pd.DataFrame(
        {
            "path_id": ["am", "pm", "overall"],
            "tradable_rows": [60, 40, 100],
            "total_edge_ticks": [12.0, 3.0, 15.0],
            "max_drawdown_ticks": [0.8, 3.4, 3.4],
            "turnover_rate": [0.10, 0.72, 0.35],
            "path_risk_label": [
                "execution_path_stable",
                "execution_path_fragile",
                "execution_path_fragile",
            ],
        }
    )

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate=quality_gate,
        capacity_stability=capacity_stability,
        path_risk_scorecard=path_risk,
        max_fragile_path_share=0.25,
    )

    assert gate["decision"] == "block"
    assert gate["passes"] is False
    assert gate["path_risk_label"] == "execution_path_fragile"
    assert gate["fragile_path_share"] == pytest.approx(0.40)
    assert gate["worst_path_id"] == "pm"
    assert gate["worst_path_drawdown_ticks"] == pytest.approx(3.4)
    assert gate["worst_path_turnover_rate"] == pytest.approx(0.72)
    assert "execution_path_fragile" in gate["blocking_reasons"]



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
    path_risk = pd.DataFrame(
        {
            "path_id": ["am", "pm", "overall"],
            "tradable_rows": [50, 30, 80],
            "total_edge_ticks": [8.0, 5.0, 13.0],
            "max_drawdown_ticks": [0.6, 0.4, 0.6],
            "turnover_rate": [0.08, 0.10, 0.09],
            "path_risk_label": [
                "execution_path_stable",
                "execution_path_stable",
                "execution_path_stable",
            ],
        }
    )

    gate = execution_publishability_release_gate(
        review_packet,
        quality_gate=quality_gate,
        capacity_stability=capacity_stability,
        path_risk_scorecard=path_risk,
    )

    assert gate["decision"] == "pass"
    assert gate["passes"] is True
    assert gate["release_gate_label"] == "execution_release_publishable"
    assert gate["path_risk_label"] == "execution_path_stable"
    assert gate["fragile_path_share"] == pytest.approx(0.0)
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


def test_passive_fill_event_policy_stability_flags_heldout_toxicity_regressions() -> None:
    train_policy = pd.DataFrame(
        {
            "lifecycle_path": ["calm|event|calm", "thin|event|stress", "calm|event|calm"],
            "threshold": [0.70, 0.70, 0.80],
            "total_events": [10, 8, 10],
            "candidate_events": [6, 4, 3],
            "event_share": [0.60, 0.50, 0.30],
            "mean_event_fill_probability": [0.82, 0.84, 0.90],
            "mean_event_adverse_fill_probability": [0.20, 0.30, 0.15],
            "mean_post_minus_pre_realized_edge": [0.30, 0.10, 0.40],
            "adverse_post_edge_share": [0.20, 0.25, 0.10],
            "policy_label": [
                "broad_lifecycle_policy",
                "broad_lifecycle_policy",
                "selective_lifecycle_policy",
            ],
        }
    )
    heldout_policy = pd.DataFrame(
        {
            "lifecycle_path": ["calm|event|calm", "thin|event|stress", "calm|event|calm"],
            "threshold": [0.70, 0.70, 0.80],
            "total_events": [9, 6, 9],
            "candidate_events": [3, 3, 0],
            "event_share": [1 / 3, 0.50, 0.0],
            "mean_event_fill_probability": [0.81, 0.83, 0.0],
            "mean_event_adverse_fill_probability": [0.28, 0.50, 0.0],
            "mean_post_minus_pre_realized_edge": [-0.10, -0.35, 0.0],
            "adverse_post_edge_share": [0.70, 0.80, 0.0],
            "policy_label": [
                "lifecycle_policy_review",
                "lifecycle_policy_blocked",
                "no_lifecycle_policy_events",
            ],
        }
    )

    stability = passive_fill_event_policy_stability(train_policy, heldout_policy)

    assert stability["lifecycle_path"].tolist() == [
        "thin|event|stress",
        "calm|event|calm",
        "calm|event|calm",
    ]
    assert stability["threshold"].tolist() == pytest.approx([0.70, 0.70, 0.80])
    assert stability["heldout_stability_label"].tolist() == [
        "heldout_policy_blocker",
        "heldout_policy_review",
        "heldout_policy_no_events",
    ]
    assert stability["candidate_event_retention"].tolist() == pytest.approx([3 / 4, 3 / 6, 0.0])
    assert stability["mean_post_minus_pre_realized_edge_delta"].tolist() == pytest.approx(
        [-0.45, -0.40, -0.40]
    )
    assert stability["adverse_post_edge_share_delta"].tolist() == pytest.approx([0.55, 0.50, -0.10])


def test_passive_fill_event_policy_stability_scorecard_blocks_candidate_weighted_regressions() -> None:
    stability = pd.DataFrame(
        {
            "lifecycle_path": ["thin|event|stress", "calm|event|calm", "calm|event|calm"],
            "threshold": [0.70, 0.70, 0.80],
            "train_total_events": [8, 10, 10],
            "heldout_total_events": [6, 9, 9],
            "train_candidate_events": [4, 6, 3],
            "heldout_candidate_events": [3, 3, 0],
            "candidate_event_retention": [0.75, 0.50, 0.0],
            "train_event_share": [0.50, 0.60, 0.30],
            "heldout_event_share": [0.50, 1 / 3, 0.0],
            "event_share_delta": [0.0, -0.2667, -0.30],
            "train_mean_event_fill_probability": [0.84, 0.82, 0.90],
            "heldout_mean_event_fill_probability": [0.83, 0.81, 0.0],
            "mean_event_fill_probability_delta": [-0.01, -0.01, -0.90],
            "train_mean_event_adverse_fill_probability": [0.30, 0.20, 0.15],
            "heldout_mean_event_adverse_fill_probability": [0.50, 0.28, 0.0],
            "mean_event_adverse_fill_probability_delta": [0.20, 0.08, -0.15],
            "train_mean_post_minus_pre_realized_edge": [0.10, 0.30, 0.40],
            "heldout_mean_post_minus_pre_realized_edge": [-0.35, -0.10, 0.0],
            "mean_post_minus_pre_realized_edge_delta": [-0.45, -0.40, -0.40],
            "train_adverse_post_edge_share": [0.25, 0.20, 0.10],
            "heldout_adverse_post_edge_share": [0.80, 0.70, 0.0],
            "adverse_post_edge_share_delta": [0.55, 0.50, -0.10],
            "train_policy_label": [
                "broad_lifecycle_policy",
                "broad_lifecycle_policy",
                "selective_lifecycle_policy",
            ],
            "heldout_policy_label": [
                "lifecycle_policy_blocked",
                "lifecycle_policy_review",
                "no_lifecycle_policy_events",
            ],
            "heldout_stability_label": [
                "heldout_policy_blocker",
                "heldout_policy_review",
                "heldout_policy_no_events",
            ],
        }
    )

    scorecard = passive_fill_event_policy_stability_scorecard(
        stability,
        max_blocker_candidate_share=0.20,
        max_review_candidate_share=0.25,
        min_weighted_edge_delta=-0.30,
    )

    assert scorecard["policy_stability_decision"] == "block"
    assert scorecard["policy_stability_label"] == "passive_fill_policy_stability_blocked"
    assert scorecard["total_train_candidate_events"] == 13
    assert scorecard["candidate_event_retention"] == pytest.approx(6 / 13)
    assert scorecard["blocker_train_candidate_share"] == pytest.approx(4 / 13)
    assert scorecard["review_train_candidate_share"] == pytest.approx(6 / 13)
    assert scorecard["weighted_mean_post_minus_pre_realized_edge_delta"] == pytest.approx(
        (-0.45 * 4 - 0.40 * 6 - 0.40 * 3) / 13
    )
    assert scorecard["worst_policy_path"] == "thin|event|stress"
    assert scorecard["worst_threshold"] == pytest.approx(0.70)


def test_passive_fill_event_policy_stability_scorecard_passes_stable_policy() -> None:
    stability = pd.DataFrame(
        {
            "lifecycle_path": ["calm|event|calm"],
            "threshold": [0.70],
            "train_candidate_events": [10],
            "heldout_candidate_events": [9],
            "candidate_event_retention": [0.90],
            "mean_post_minus_pre_realized_edge_delta": [-0.02],
            "adverse_post_edge_share_delta": [0.01],
            "heldout_stability_label": ["heldout_policy_stable"],
        }
    )

    scorecard = passive_fill_event_policy_stability_scorecard(stability)

    assert scorecard["policy_stability_decision"] == "pass"
    assert scorecard["policy_stability_label"] == "passive_fill_policy_stability_pass"
    assert scorecard["blocking_reasons"] == "none"


def test_passive_fill_event_policy_stability_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing passive fill event policy stability train columns"):
        passive_fill_event_policy_stability(pd.DataFrame({"lifecycle_path": ["a"]}), pd.DataFrame())
    with pytest.raises(ValueError, match="missing passive fill event policy stability scorecard columns"):
        passive_fill_event_policy_stability_scorecard(pd.DataFrame({"lifecycle_path": ["a"]}))


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


def test_queue_position_expected_value_frontier_scores_queue_and_fill_cutoffs() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["calm", "calm", "stress", "stress", "calm"],
            "best_execution_side": ["long", "long", "short", "short", "abstain"],
            "bid_queue_share": [0.10, 0.45, 0.20, 0.70, 0.15],
            "ask_queue_share": [0.60, 0.35, 0.20, 0.30, 0.10],
            "bid_fill_probability": [0.90, 0.65, 0.30, 0.20, 0.80],
            "ask_fill_probability": [0.20, 0.40, 0.85, 0.70, 0.30],
            "bid_adverse_fill_probability": [0.10, 0.25, 0.30, 0.20, 0.10],
            "ask_adverse_fill_probability": [0.35, 0.30, 0.15, 0.45, 0.20],
            "execution_adjusted_edge_ticks": [0.60, 0.20, 0.50, -0.10, 0.0],
        }
    )

    frontier = queue_position_expected_value_frontier(
        frame,
        min_fill_probabilities=(0.60, 0.80),
        max_queue_shares=(0.25, 0.50),
        adverse_selection_cost_ticks=0.40,
        queue_drag_cost_ticks=0.20,
        regime_col="regime",
    )

    assert frontier["regime"].tolist() == [
        "calm",
        "calm",
        "calm",
        "calm",
        "stress",
        "stress",
        "stress",
        "stress",
    ]
    assert frontier["min_fill_probability"].tolist() == pytest.approx([
        0.60,
        0.60,
        0.80,
        0.80,
        0.60,
        0.60,
        0.80,
        0.80,
    ])
    assert frontier["max_queue_share"].tolist() == pytest.approx([
        0.25,
        0.50,
        0.25,
        0.50,
        0.25,
        0.50,
        0.25,
        0.50,
    ])
    assert frontier["candidate_rows"].tolist() == [1, 2, 1, 1, 1, 2, 1, 1]
    assert frontier["candidate_share"].tolist() == pytest.approx([0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5])
    calm_broad = frontier[(frontier["regime"] == "calm") & (frontier["max_queue_share"] == 0.50)].iloc[0]
    assert calm_broad["mean_fill_probability"] == pytest.approx((0.90 + 0.65) / 2)
    assert calm_broad["expected_value_ticks"] == pytest.approx((0.60 * 0.90 + 0.20 * 0.65) / 2)
    assert calm_broad["risk_adjusted_expected_value_ticks"] == pytest.approx(
        ((0.60 * 0.90 - 0.10 * 0.40 - 0.10 * 0.20) + (0.20 * 0.65 - 0.25 * 0.40 - 0.45 * 0.20)) / 2
    )
    assert calm_broad["policy_label"] == "broad_positive_ev_queue_policy"
    stress_broad = frontier[(frontier["regime"] == "stress") & (frontier["max_queue_share"] == 0.50)].iloc[0]
    assert stress_broad["policy_label"] == "queue_policy_toxicity_review"


def test_queue_position_expected_value_policy_selection_picks_best_deployable_policy() -> None:
    frontier = pd.DataFrame(
        {
            "regime": ["open", "open", "close", "close"],
            "min_fill_probability": [0.50, 0.70, 0.50, 0.70],
            "max_queue_share": [0.75, 0.50, 0.75, 0.50],
            "tradable_rows": [10, 10, 8, 8],
            "candidate_rows": [5, 2, 4, 3],
            "candidate_share": [0.50, 0.20, 0.50, 0.375],
            "long_rows": [3, 1, 2, 1],
            "short_rows": [2, 1, 2, 2],
            "mean_queue_share": [0.50, 0.25, 0.60, 0.30],
            "mean_fill_probability": [0.60, 0.78, 0.62, 0.75],
            "mean_adverse_fill_probability": [0.25, 0.20, 0.30, 0.22],
            "mean_execution_adjusted_edge_ticks": [0.70, 0.90, 0.20, 0.30],
            "expected_value_ticks": [0.42, 0.70, 0.12, 0.225],
            "risk_adjusted_expected_value_ticks": [0.20, 0.44, -0.04, 0.02],
            "policy_label": ["review", "review", "reject", "review"],
        }
    )

    selection = queue_position_expected_value_policy_selection(frontier, min_candidate_share=0.25)

    assert selection.columns.tolist() == [
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
    assert selection["regime"].tolist() == ["close", "open"]
    assert selection["selected_min_fill_probability"].tolist() == pytest.approx([0.70, 0.50])
    assert selection["selected_max_queue_share"].tolist() == pytest.approx([0.50, 0.75])
    assert selection["selection_label"].tolist() == ["deployable", "deployable"]


def test_queue_position_expected_value_policy_selection_flags_capacity_blockers() -> None:
    frontier = pd.DataFrame(
        {
            "regime": ["all", "all"],
            "min_fill_probability": [0.50, 0.70],
            "max_queue_share": [0.75, 0.50],
            "tradable_rows": [10, 10],
            "candidate_rows": [1, 0],
            "candidate_share": [0.10, 0.0],
            "expected_value_ticks": [0.30, 0.0],
            "risk_adjusted_expected_value_ticks": [0.20, 0.0],
            "mean_fill_probability": [0.60, 0.0],
            "mean_queue_share": [0.40, 0.0],
            "mean_adverse_fill_probability": [0.20, 0.0],
        }
    )

    selection = queue_position_expected_value_policy_selection(frontier, min_candidate_share=0.25)

    assert selection.loc[0, "candidate_rows"] == 1
    assert selection.loc[0, "selection_label"] == "capacity_constrained"


def test_queue_position_expected_value_policy_scorecard_summarizes_deployment_readiness() -> None:
    selection = pd.DataFrame(
        {
            "regime": ["close", "open", "stress"],
            "selected_min_fill_probability": [0.70, 0.50, 0.80],
            "selected_max_queue_share": [0.50, 0.75, 0.40],
            "tradable_rows": [8, 10, 5],
            "candidate_rows": [3, 5, 1],
            "candidate_share": [0.375, 0.50, 0.20],
            "risk_adjusted_expected_value_ticks": [0.02, 0.44, -0.05],
            "expected_value_ticks": [0.225, 0.70, 0.10],
            "mean_fill_probability": [0.75, 0.60, 0.81],
            "mean_queue_share": [0.30, 0.50, 0.20],
            "mean_adverse_fill_probability": [0.22, 0.25, 0.45],
            "policy_rank": [1, 2, 3],
            "selection_label": ["deployable", "deployable", "negative_expected_value"],
        }
    )

    scorecard = queue_position_expected_value_policy_scorecard(selection)

    assert scorecard.columns.tolist() == [
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
    row = scorecard.iloc[0]
    assert row["regimes"] == 3
    assert row["deployable_regimes"] == 2
    assert row["blocked_regimes"] == 1
    assert row["deployable_share"] == pytest.approx(2 / 3)
    assert row["negative_expected_value_regimes"] == 1
    assert row["candidate_weighted_share"] == pytest.approx((3 * 0.375 + 5 * 0.50 + 1 * 0.20) / 9)
    assert row["candidate_weighted_risk_adjusted_expected_value_ticks"] == pytest.approx(
        (3 * 0.02 + 5 * 0.44 + 1 * -0.05) / 9
    )
    assert row["readiness_label"] == "mixed_review"


def test_queue_position_expected_value_frontier_rejects_invalid_costs_and_missing_columns() -> None:
    with pytest.raises(ValueError, match="adverse_selection_cost_ticks must be finite and non-negative"):
        queue_position_expected_value_frontier(pd.DataFrame(), adverse_selection_cost_ticks=-0.1)
    with pytest.raises(ValueError, match="missing queue position expected value frontier columns"):
        queue_position_expected_value_frontier(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_expected_value_stress_table_haircuts_selected_policies() -> None:
    selection = pd.DataFrame(
        {
            "regime": ["calm", "event"],
            "selected_min_fill_probability": [0.60, 0.80],
            "selected_max_queue_share": [0.50, 0.25],
            "candidate_rows": [10, 4],
            "candidate_share": [0.50, 0.20],
            "expected_value_ticks": [0.60, 0.32],
            "risk_adjusted_expected_value_ticks": [0.42, 0.08],
            "mean_fill_probability": [0.75, 0.80],
            "mean_queue_share": [0.20, 0.25],
            "mean_adverse_fill_probability": [0.10, 0.30],
            "selection_label": ["deployable", "deployable"],
        }
    )

    stressed = queue_position_expected_value_stress_table(
        selection,
        stress_scenarios={"base": (0.0, 0.0), "latency_hit": (0.25, 0.10)},
        adverse_selection_cost_ticks=0.50,
        queue_drag_cost_ticks=0.25,
        min_candidate_share=0.25,
        min_stressed_expected_value_ticks=0.10,
    )

    assert stressed.columns.tolist() == [
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
    calm_latency = stressed[(stressed["scenario"] == "latency_hit") & (stressed["regime"] == "calm")].iloc[0]
    event_latency = stressed[(stressed["scenario"] == "latency_hit") & (stressed["regime"] == "event")].iloc[0]
    assert calm_latency["stressed_fill_probability"] == pytest.approx(0.75 * 0.75)
    assert calm_latency["stressed_expected_value_ticks"] == pytest.approx(0.80 * 0.75 * 0.75 - 0.20 * 0.50 - 0.20 * 0.25)
    assert calm_latency["stress_label"] == "stress_robust"
    assert event_latency["stress_label"] == "capacity_or_ev_fragile"


def test_queue_position_expected_value_stress_summary_reduces_regime_scenario_fragility() -> None:
    stress = pd.DataFrame(
        {
            "scenario": ["base", "base", "latency_hit", "latency_hit"],
            "regime": ["calm", "event", "calm", "event"],
            "candidate_rows": [10, 4, 10, 4],
            "candidate_share": [0.50, 0.20, 0.50, 0.20],
            "stressed_expected_value_ticks": [0.45, 0.12, 0.30, -0.05],
            "expected_value_decay_ticks": [0.15, 0.20, 0.30, 0.37],
            "stress_label": [
                "stress_robust",
                "capacity_fragile",
                "stress_robust",
                "capacity_or_ev_fragile",
            ],
        }
    )

    summary = queue_position_expected_value_stress_summary(
        stress,
        max_fragile_candidate_share=0.30,
        review_fragile_candidate_share=0.10,
        min_candidate_weighted_ev_ticks=0.05,
        min_worst_scenario_ev_ticks=0.0,
    )

    assert summary == {
        "stress_rows": 4,
        "scenarios": 2,
        "regimes": 2,
        "candidate_rows": 28,
        "fragile_candidate_rows": 8,
        "fragile_candidate_share": pytest.approx(8 / 28),
        "candidate_weighted_expected_value_ticks": pytest.approx(
            (10 * 0.45 + 4 * 0.12 + 10 * 0.30 + 4 * -0.05) / 28
        ),
        "candidate_weighted_decay_ticks": pytest.approx(
            (10 * 0.15 + 4 * 0.20 + 10 * 0.30 + 4 * 0.37) / 28
        ),
        "worst_scenario": "latency_hit",
        "worst_scenario_expected_value_ticks": pytest.approx((10 * 0.30 + 4 * -0.05) / 14),
        "worst_regime": "event",
        "worst_regime_expected_value_ticks": pytest.approx((4 * 0.12 + 4 * -0.05) / 8),
        "stress_release_decision": "review",
        "stress_release_label": "queue_expected_value_stress_review",
        "blocking_reasons": "none",
        "review_reasons": "fragile_candidate_share",
    }


def test_queue_position_expected_value_stress_summary_blocks_negative_worst_scenario() -> None:
    stress = pd.DataFrame(
        {
            "scenario": ["base", "toxicity_hit"],
            "regime": ["all", "all"],
            "candidate_rows": [5, 5],
            "candidate_share": [1.0, 1.0],
            "stressed_expected_value_ticks": [0.10, -0.20],
            "expected_value_decay_ticks": [0.0, 0.30],
            "stress_label": ["stress_robust", "expected_value_fragile"],
        }
    )

    summary = queue_position_expected_value_stress_summary(
        stress,
        max_fragile_candidate_share=0.90,
        min_candidate_weighted_ev_ticks=-1.0,
        min_worst_scenario_ev_ticks=0.0,
    )

    assert summary["stress_release_decision"] == "block"
    assert summary["stress_release_label"] == "queue_expected_value_stress_blocked"
    assert summary["blocking_reasons"] == "worst_scenario_expected_value"


def test_queue_position_expected_value_stress_table_rejects_invalid_scenarios() -> None:
    selection = pd.DataFrame(
        {
            "regime": ["all"],
            "candidate_rows": [1],
            "candidate_share": [1.0],
            "expected_value_ticks": [0.2],
            "mean_fill_probability": [0.5],
            "mean_queue_share": [0.2],
            "mean_adverse_fill_probability": [0.1],
        }
    )

    with pytest.raises(ValueError, match="stress_scenarios values must be"):
        queue_position_expected_value_stress_table(selection, stress_scenarios={"bad": (1.2, 0.0)})
    with pytest.raises(ValueError, match="stress_scenarios values must be"):
        queue_position_expected_value_stress_table(selection, stress_scenarios={"bad": 0.25})


def test_execution_adjusted_lcri_event_window_attribution_flags_high_lcri_event_fragility() -> None:
    frame = pd.DataFrame(
        {
            "passive_fill_event_window_regime": [
                "calm",
                "calm",
                "event",
                "event",
                "post_event",
                "post_event",
            ],
            "lcri": [0.2, 0.5, 2.0, -2.5, 1.8, -3.0],
            "execution_adjusted_lcri_score": [0.2, 0.4, 0.2, -0.3, 1.4, -2.4],
            "execution_adjusted_edge_ticks": [0.2, 0.3, -0.4, -0.6, 0.6, 0.8],
            "best_execution_side": ["long", "short", "long", "short", "long", "short"],
            "bid_fill_probability": [0.50, 0.40, 0.92, 0.20, 0.70, 0.10],
            "ask_fill_probability": [0.30, 0.55, 0.10, 0.95, 0.20, 0.75],
            "bid_adverse_fill_probability": [0.10, 0.20, 0.65, 0.10, 0.20, 0.10],
            "ask_adverse_fill_probability": [0.20, 0.15, 0.10, 0.70, 0.10, 0.20],
        }
    )

    output = execution_adjusted_lcri_event_window_attribution(frame, bins=2)

    assert set(output["passive_fill_event_window_regime"]) == {"calm", "event", "post_event"}
    event = output[output["passive_fill_event_window_regime"] == "event"].iloc[0]
    post_event = output[output["passive_fill_event_window_regime"] == "post_event"].iloc[0]
    calm = output[output["passive_fill_event_window_regime"] == "calm"].iloc[0]

    assert event["bucket"] == "high_abs_lcri"
    assert event["rows"] == 2
    assert event["signal_survival_ratio"] == pytest.approx(1.0 / 9.0)
    assert event["mean_selected_fill_probability"] == pytest.approx(0.935)
    assert event["mean_selected_adverse_fill_probability"] == pytest.approx(0.675)
    assert event["negative_edge_share"] == pytest.approx(1.0)
    assert event["event_window_execution_label"] == "high_lcri_event_toxicity"
    assert post_event["event_window_execution_label"] == "event_window_edge_survives"
    assert calm["event_window_execution_label"] == "low_lcri_reference"


def test_execution_adjusted_lcri_event_window_attribution_rejects_missing_regime() -> None:
    with pytest.raises(
        ValueError, match="missing execution-adjusted LCRI event-window attribution columns"
    ):
        execution_adjusted_lcri_event_window_attribution(pd.DataFrame({"lcri": [1.0]}))


def test_execution_adjusted_lcri_event_window_release_scorecard_blocks_toxic_events() -> None:
    attribution = pd.DataFrame(
        {
            "passive_fill_event_window_regime": ["event", "post_event", "calm"],
            "bucket": ["high_abs_lcri", "high_abs_lcri", "low_abs_lcri"],
            "rows": [8, 6, 10],
            "signal_survival_ratio": [0.20, 0.80, 0.90],
            "tradable_share": [1.0, 1.0, 0.4],
            "fill_minus_adverse_probability_spread": [0.05, 0.40, 0.20],
            "mean_execution_adjusted_edge_ticks": [-0.30, 0.50, 0.10],
            "negative_edge_share": [0.75, 0.10, 0.05],
            "event_window_execution_label": [
                "high_lcri_event_toxicity",
                "event_window_edge_survives",
                "low_lcri_reference",
            ],
        }
    )

    scorecard = execution_adjusted_lcri_event_window_release_scorecard(
        attribution,
        max_toxic_high_lcri_row_share=0.30,
        min_high_lcri_survival_ratio=0.50,
        min_high_lcri_fill_adverse_spread=0.25,
    )

    assert scorecard == {
        "high_lcri_rows": 14,
        "toxic_high_lcri_rows": 8,
        "toxic_high_lcri_row_share": pytest.approx(8 / 14),
        "event_high_lcri_rows": 8,
        "event_toxic_high_lcri_rows": 8,
        "event_toxic_high_lcri_row_share": pytest.approx(1.0),
        "weighted_high_lcri_signal_survival_ratio": pytest.approx((0.20 * 8 + 0.80 * 6) / 14),
        "weighted_high_lcri_fill_adverse_spread": pytest.approx((0.05 * 8 + 0.40 * 6) / 14),
        "weighted_high_lcri_negative_edge_share": pytest.approx((0.75 * 8 + 0.10 * 6) / 14),
        "worst_event_window_regime": "event",
        "worst_event_window_bucket": "high_abs_lcri",
        "worst_event_window_label": "high_lcri_event_toxicity",
        "release_decision": "block",
        "release_label": "execution_lcri_event_window_blocked",
        "blocking_reasons": "toxic_high_lcri_share;event_toxic_high_lcri_share;low_signal_survival;low_fill_adverse_spread",
        "review_reasons": "none",
    }


def test_execution_adjusted_lcri_event_window_release_scorecard_passes_surviving_edges() -> None:
    attribution = pd.DataFrame(
        {
            "passive_fill_event_window_regime": ["event", "post_event"],
            "bucket": ["high_abs_lcri", "high_abs_lcri"],
            "rows": [5, 5],
            "signal_survival_ratio": [0.75, 0.80],
            "tradable_share": [1.0, 1.0],
            "fill_minus_adverse_probability_spread": [0.30, 0.35],
            "mean_execution_adjusted_edge_ticks": [0.40, 0.50],
            "negative_edge_share": [0.05, 0.10],
            "event_window_execution_label": [
                "event_window_edge_survives",
                "event_window_edge_survives",
            ],
        }
    )

    scorecard = execution_adjusted_lcri_event_window_release_scorecard(attribution)

    assert scorecard["release_decision"] == "pass"
    assert scorecard["release_label"] == "execution_lcri_event_window_pass"
    assert scorecard["blocking_reasons"] == "none"


def test_execution_adjusted_lcri_event_window_release_scorecard_reviews_without_high_lcri() -> None:
    attribution = pd.DataFrame(
        {
            "passive_fill_event_window_regime": ["event", "calm"],
            "bucket": ["low_abs_lcri", "medium_abs_lcri"],
            "rows": [5, 5],
            "signal_survival_ratio": [0.0, 0.0],
            "tradable_share": [1.0, 1.0],
            "fill_minus_adverse_probability_spread": [-0.50, -0.25],
            "mean_execution_adjusted_edge_ticks": [-0.40, -0.30],
            "negative_edge_share": [1.0, 1.0],
            "event_window_execution_label": [
                "low_lcri_reference",
                "low_lcri_reference",
            ],
        }
    )

    scorecard = execution_adjusted_lcri_event_window_release_scorecard(attribution)

    assert scorecard["release_decision"] == "review"
    assert scorecard["release_label"] == "execution_lcri_event_window_review"
    assert scorecard["blocking_reasons"] == "none"
    assert scorecard["review_reasons"] == "no_high_lcri_rows"


def test_execution_adjusted_lcri_event_window_release_scorecard_rejects_missing_columns() -> None:
    with pytest.raises(
        ValueError, match="missing execution-adjusted LCRI event-window release scorecard columns"
    ):
        execution_adjusted_lcri_event_window_release_scorecard(pd.DataFrame({"rows": [1]}))


def test_queue_position_latency_regime_surface_segments_decision_regimes() -> None:
    frame = pd.DataFrame(
        {
            "event_window_regime": ["pre", "pre", "post", "post"],
            "symbol": ["A", "A", "A", "A"],
            "best_execution_side": ["long", "short", "long", "abstain"],
            "bid_fill_probability": [0.70, 0.40, 0.80, 0.10],
            "ask_fill_probability": [0.20, 0.60, 0.30, 0.10],
            "bid_realized_fill": [1.0, 0.0, 0.0, 1.0],
            "ask_realized_fill": [0.0, 1.0, 1.0, 0.0],
            "execution_adjusted_edge_ticks": [0.50, 0.30, 0.20, -0.10],
        }
    )

    surface = queue_position_latency_regime_surface(
        frame,
        regime_col="event_window_regime",
        group_cols="symbol",
        latencies=(0, 1),
        max_realized_fill_decay=0.25,
    )

    assert surface["event_window_regime"].tolist() == ["post", "post", "pre", "pre"]
    assert surface["latency_steps"].tolist() == [0, 1, 0, 1]
    assert surface["candidates"].tolist() == [1, 1, 2, 2]
    assert surface["realized_fill_rate"].tolist() == pytest.approx([0.0, 1.0, 1.0, 0.5])
    assert surface["realized_fill_gap_vs_immediate"].tolist() == pytest.approx([0.0, 1.0, 0.0, -0.5])
    assert surface["mean_decision_fill_probability"].tolist() == pytest.approx(
        [0.8, 0.8, 0.65, 0.65]
    )
    assert surface["latency_regime_label"].tolist() == [
        "anchor_latency",
        "latency_robust",
        "anchor_latency",
        "latency_fragile",
    ]


def test_queue_position_latency_regime_surface_requires_regime_column() -> None:
    with pytest.raises(ValueError, match="missing queue position latency regime surface columns"):
        queue_position_latency_regime_surface(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_expected_value_policy_drift_quantifies_recalibration_risk() -> None:
    train_selection = pd.DataFrame(
        {
            "regime": ["calm", "stress", "auction"],
            "selected_min_fill_probability": [0.30, 0.45, 0.55],
            "selected_max_queue_share": [0.70, 0.50, 0.35],
            "candidate_share": [0.50, 0.35, 0.00],
            "risk_adjusted_expected_value_ticks": [0.40, 0.30, -0.10],
            "selection_label": ["deployable", "deployable", "negative_expected_value"],
        }
    )
    holdout_selection = pd.DataFrame(
        {
            "regime": ["calm", "stress", "auction"],
            "selected_min_fill_probability": [0.35, 0.75, 0.55],
            "selected_max_queue_share": [0.68, 0.20, 0.35],
            "candidate_share": [0.48, 0.12, 0.00],
            "risk_adjusted_expected_value_ticks": [0.36, 0.05, -0.20],
            "selection_label": ["deployable", "deployable", "negative_expected_value"],
        }
    )

    drift = queue_position_expected_value_policy_drift(
        train_selection,
        holdout_selection,
        max_threshold_drift=0.10,
        max_ev_decay_ratio=0.50,
        min_holdout_candidate_share=0.15,
    )

    assert drift.columns.tolist() == [
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
    assert drift["policy_drift_label"].tolist() == [
        "policy_stable",
        "policy_recalibration_required",
        "not_deployable",
    ]
    assert drift["threshold_l1_drift"].tolist() == pytest.approx([0.07, 0.60, 0.0])
    assert drift["ev_decay_ratio"].tolist() == pytest.approx([0.10, 5.0 / 6.0, 1.0])
    assert drift.loc[1, "review_reasons"] == "threshold_drift;holdout_capacity;ev_decay"


def test_queue_position_expected_value_policy_drift_marks_missing_holdout_regime() -> None:
    train_selection = pd.DataFrame(
        {
            "regime": ["calm"],
            "selected_min_fill_probability": [0.30],
            "selected_max_queue_share": [0.70],
            "candidate_share": [0.50],
            "risk_adjusted_expected_value_ticks": [0.40],
            "selection_label": ["deployable"],
        }
    )
    holdout_selection = train_selection.iloc[0:0].copy()

    drift = queue_position_expected_value_policy_drift(train_selection, holdout_selection)

    assert drift.loc[0, "policy_drift_label"] == "holdout_missing_regime"
    assert drift.loc[0, "review_reasons"] == "missing_holdout_regime"
    assert drift.loc[0, "holdout_candidate_share"] == pytest.approx(0.0)


def test_queue_position_expected_value_oos_validation_flags_policy_decay() -> None:
    selection = pd.DataFrame(
        {
            "regime": ["calm", "event"],
            "selected_min_fill_probability": [0.60, 0.70],
            "selected_max_queue_share": [0.50, 0.25],
            "candidate_rows": [12, 8],
            "candidate_share": [0.40, 0.25],
            "risk_adjusted_expected_value_ticks": [0.30, 0.20],
            "selection_label": ["deployable", "deployable"],
        }
    )
    holdout_frontier = pd.DataFrame(
        {
            "regime": ["calm", "calm", "event"],
            "min_fill_probability": [0.60, 0.70, 0.70],
            "max_queue_share": [0.50, 0.50, 0.25],
            "tradable_rows": [30, 30, 20],
            "candidate_rows": [9, 6, 1],
            "candidate_share": [0.30, 0.20, 0.05],
            "risk_adjusted_expected_value_ticks": [0.18, 0.24, -0.05],
            "expected_value_ticks": [0.24, 0.30, 0.02],
            "mean_fill_probability": [0.72, 0.80, 0.75],
            "mean_queue_share": [0.32, 0.35, 0.20],
            "mean_adverse_fill_probability": [0.20, 0.18, 0.45],
        }
    )

    validation = queue_position_expected_value_oos_validation(
        selection,
        holdout_frontier,
        min_holdout_candidate_share=0.10,
        max_ev_decay_ratio=0.50,
        min_holdout_expected_value_ticks=0.0,
    )

    assert validation["regime"].tolist() == ["calm", "event"]
    assert validation["holdout_candidate_rows"].tolist() == [9, 1]
    assert validation["holdout_risk_adjusted_expected_value_ticks"].tolist() == pytest.approx(
        [0.18, -0.05]
    )
    assert validation["ev_decay_ratio"].tolist() == pytest.approx([0.40, 1.25])
    assert validation["oos_validation_label"].tolist() == ["oos_stable", "oos_broken"]
    assert validation["review_reasons"].tolist() == ["none", "capacity;negative_ev;ev_decay"]


def test_queue_position_expected_value_oos_validation_marks_missing_policy() -> None:
    selection = pd.DataFrame(
        {
            "regime": ["event"],
            "selected_min_fill_probability": [0.80],
            "selected_max_queue_share": [0.25],
            "candidate_rows": [5],
            "candidate_share": [0.20],
            "risk_adjusted_expected_value_ticks": [0.10],
            "selection_label": ["deployable"],
        }
    )
    holdout_frontier = pd.DataFrame(
        {
            "regime": ["event"],
            "min_fill_probability": [0.60],
            "max_queue_share": [0.50],
            "tradable_rows": [20],
            "candidate_rows": [8],
            "candidate_share": [0.40],
            "risk_adjusted_expected_value_ticks": [0.20],
            "expected_value_ticks": [0.25],
            "mean_fill_probability": [0.70],
            "mean_queue_share": [0.30],
            "mean_adverse_fill_probability": [0.20],
        }
    )

    validation = queue_position_expected_value_oos_validation(selection, holdout_frontier)

    assert validation.loc[0, "oos_validation_label"] == "oos_missing_policy"
    assert validation.loc[0, "review_reasons"] == "missing_holdout_policy"


def test_queue_position_expected_value_oos_validation_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position expected value OOS validation columns"):
        queue_position_expected_value_oos_validation(
            pd.DataFrame({"regime": ["calm"]}),
            pd.DataFrame({"regime": ["calm"]}),
        )


def test_queue_position_latency_edge_regime_surface_prices_fragile_event_windows() -> None:
    frame = pd.DataFrame(
        {
            "event_window_regime": ["pre", "pre", "post", "post"],
            "symbol": ["A", "A", "A", "A"],
            "best_execution_side": ["long", "short", "long", "abstain"],
            "bid_realized_fill": [1.0, 0.0, 0.0, 1.0],
            "ask_realized_fill": [0.0, 1.0, 1.0, 0.0],
            "execution_adjusted_edge_ticks": [0.50, 0.30, 0.20, -0.10],
        }
    )

    surface = queue_position_latency_edge_regime_surface(
        frame,
        regime_col="event_window_regime",
        group_cols="symbol",
        latencies=(0, 1),
        max_realized_edge_decay=0.20,
    )

    assert surface["event_window_regime"].tolist() == ["post", "post", "pre", "pre"]
    assert surface["latency_steps"].tolist() == [0, 1, 0, 1]
    assert surface["candidates"].tolist() == [1, 1, 2, 2]
    assert surface["realized_fill_rate"].tolist() == pytest.approx([0.0, 1.0, 1.0, 0.5])
    assert surface["mean_decision_edge_ticks"].tolist() == pytest.approx([0.20, 0.20, 0.40, 0.40])
    assert surface["realized_edge_ticks"].tolist() == pytest.approx([0.0, 0.20, 0.40, 0.15])
    assert surface["realized_edge_gap_vs_immediate"].tolist() == pytest.approx([0.0, 0.20, 0.0, -0.25])
    assert surface["edge_survival_ratio"].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.375])
    assert surface["edge_latency_regime_label"].tolist() == [
        "anchor_latency",
        "edge_latency_regime_robust",
        "anchor_latency",
        "edge_latency_regime_fragile",
    ]


def test_queue_position_latency_edge_regime_surface_requires_regime_column() -> None:
    with pytest.raises(ValueError, match="missing queue position latency edge regime surface columns"):
        queue_position_latency_edge_regime_surface(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_latency_release_scorecard_blocks_fragile_latency_regimes() -> None:
    surface = pd.DataFrame(
        {
            "event_window_regime": ["calm", "calm", "event", "event", "post", "post"],
            "latency_steps": [0, 1, 0, 1, 0, 2],
            "candidates": [10, 8, 6, 5, 4, 4],
            "realized_fill_rate": [0.80, 0.76, 0.90, 0.45, 0.75, 0.60],
            "realized_fill_gap_vs_immediate": [0.0, -0.04, 0.0, -0.45, 0.0, -0.15],
            "mean_execution_adjusted_edge_ticks": [0.30, 0.28, 0.60, 0.20, 0.35, 0.30],
            "latency_regime_label": [
                "anchor_latency",
                "latency_robust",
                "anchor_latency",
                "latency_fragile",
                "anchor_latency",
                "latency_fragile",
            ],
        }
    )

    scorecard = queue_position_latency_release_scorecard(
        surface,
        regime_col="event_window_regime",
        max_fragile_candidate_share=0.25,
        min_weighted_fill_gap=-0.12,
    )

    assert scorecard == {
        "regimes": 3,
        "latency_rows": 3,
        "anchor_candidates": 20,
        "latency_candidates": 17,
        "candidate_retention_share": pytest.approx(17 / 20),
        "fragile_latency_rows": 2,
        "fragile_candidate_share": pytest.approx(9 / 17),
        "candidate_weighted_fill_gap": pytest.approx((-0.04 * 8 - 0.45 * 5 - 0.15 * 4) / 17),
        "worst_regime": "event",
        "worst_latency_steps": 1,
        "worst_fill_gap": pytest.approx(-0.45),
        "latency_release_decision": "block",
        "latency_release_label": "queue_latency_release_blocked",
        "blocking_reasons": "fragile_candidate_share;weighted_fill_gap",
        "review_reasons": "none",
    }


def test_queue_position_latency_release_scorecard_passes_latency_robust_surface() -> None:
    surface = pd.DataFrame(
        {
            "event_window_regime": ["calm", "calm"],
            "latency_steps": [0, 1],
            "candidates": [10, 9],
            "realized_fill_gap_vs_immediate": [0.0, -0.03],
            "latency_regime_label": ["anchor_latency", "latency_robust"],
        }
    )

    scorecard = queue_position_latency_release_scorecard(surface, regime_col="event_window_regime")

    assert scorecard["latency_release_decision"] == "pass"
    assert scorecard["latency_release_label"] == "queue_latency_release_pass"
    assert scorecard["blocking_reasons"] == "none"


def test_queue_position_latency_release_scorecard_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position latency release scorecard columns"):
        queue_position_latency_release_scorecard(pd.DataFrame({"latency_steps": [0]}))


def test_queue_position_path_tail_loss_scorecard_flags_clustered_left_tail() -> None:
    frame = pd.DataFrame(
        {
            "session": ["am", "am", "am", "am", "pm", "pm", "pm"],
            "best_execution_side": ["long", "long", "short", "abstain", "short", "short", "long"],
            "execution_adjusted_edge_ticks": [1.0, -3.0, -2.0, 9.0, 0.5, -0.5, 0.25],
        }
    )

    output = queue_position_path_tail_loss_scorecard(
        frame,
        group_cols="session",
        tail_probability=0.50,
        max_tail_loss_ticks=2.0,
        max_severe_loss_share=0.40,
        max_loss_run_length=1,
        severe_loss_ticks=2.0,
    )

    assert output["path_id"].tolist() == ["am", "pm", "overall"]
    assert output["tradable_rows"].tolist() == [3, 3, 6]
    assert output["loss_rows"].tolist() == [2, 1, 3]
    assert output["tail_loss_threshold_ticks"].tolist() == pytest.approx([2.0, 0.0, 0.25])
    assert output["conditional_tail_loss_ticks"].tolist() == pytest.approx([2.5, 0.5, 5.5 / 3.0])
    assert output["severe_loss_share"].tolist() == pytest.approx([2.0 / 3.0, 0.0, 2.0 / 6.0])
    assert output["max_loss_run_length"].tolist() == [2, 1, 2]
    assert output["tail_loss_label"].tolist() == [
        "execution_tail_loss_fragile",
        "execution_tail_loss_stable",
        "execution_tail_loss_fragile",
    ]


def test_queue_position_path_tail_loss_scorecard_rejects_missing_edge() -> None:
    with pytest.raises(ValueError, match="missing queue position path tail loss columns"):
        queue_position_path_tail_loss_scorecard(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_path_tail_loss_release_gate_blocks_concentrated_tail_losses() -> None:
    scorecard = pd.DataFrame(
        {
            "path_id": ["am", "pm", "overnight", "overall"],
            "rows": [4, 4, 4, 12],
            "tradable_rows": [4, 4, 4, 12],
            "loss_rows": [1, 3, 1, 5],
            "mean_loss_ticks": [0.10, 0.90, 0.05, 0.35],
            "tail_loss_threshold_ticks": [0.25, 1.00, 0.10, 0.50],
            "conditional_tail_loss_ticks": [0.25, 2.25, 0.10, 1.50],
            "severe_loss_share": [0.0, 0.75, 0.0, 5.0 / 12.0],
            "max_loss_run_length": [1, 3, 1, 3],
            "tail_loss_label": [
                "execution_tail_loss_stable",
                "execution_tail_loss_fragile",
                "execution_tail_loss_stable",
                "execution_tail_loss_fragile",
            ],
        }
    )

    gate = queue_position_path_tail_loss_release_gate(
        scorecard,
        max_fragile_path_share=0.25,
        max_overall_conditional_tail_loss_ticks=1.0,
        max_overall_severe_loss_share=0.25,
        max_overall_loss_run_length=2,
    )

    assert gate == {
        "paths": 3,
        "fragile_paths": 1,
        "fragile_path_share": pytest.approx(1.0 / 3.0),
        "total_tradable_rows": 12,
        "overall_conditional_tail_loss_ticks": pytest.approx(1.5),
        "overall_severe_loss_share": pytest.approx(5.0 / 12.0),
        "overall_max_loss_run_length": 3,
        "worst_path_id": "pm",
        "worst_path_tail_loss_label": "execution_tail_loss_fragile",
        "tail_loss_release_decision": "block",
        "tail_loss_release_label": "queue_tail_loss_release_blocked",
        "blocking_reasons": "fragile_path_share;overall_conditional_tail_loss;overall_severe_loss_share;overall_loss_run_length",
        "review_reasons": "none",
    }


def test_queue_position_path_tail_loss_release_gate_passes_stable_tail_paths() -> None:
    scorecard = pd.DataFrame(
        {
            "path_id": ["am", "overall"],
            "rows": [4, 4],
            "tradable_rows": [4, 4],
            "loss_rows": [1, 1],
            "mean_loss_ticks": [0.05, 0.05],
            "tail_loss_threshold_ticks": [0.10, 0.10],
            "conditional_tail_loss_ticks": [0.10, 0.10],
            "severe_loss_share": [0.0, 0.0],
            "max_loss_run_length": [1, 1],
            "tail_loss_label": ["execution_tail_loss_stable", "execution_tail_loss_stable"],
        }
    )

    gate = queue_position_path_tail_loss_release_gate(scorecard)

    assert gate["tail_loss_release_decision"] == "pass"
    assert gate["tail_loss_release_label"] == "queue_tail_loss_release_pass"
    assert gate["blocking_reasons"] == "none"
    assert gate["review_reasons"] == "none"


def test_queue_position_path_tail_loss_release_gate_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position path tail loss release gate columns"):
        queue_position_path_tail_loss_release_gate(pd.DataFrame({"path_id": ["overall"]}))


def test_queue_position_path_risk_scorecard_tracks_drawdown_and_turnover() -> None:
    frame = pd.DataFrame(
        {
            "session": ["am", "am", "am", "pm", "pm"],
            "best_execution_side": ["long", "long", "short", "short", "abstain"],
            "execution_adjusted_edge_ticks": [1.0, -2.0, 3.0, -1.0, 5.0],
        }
    )

    output = queue_position_path_risk_scorecard(frame, group_cols="session")

    assert output["path_id"].tolist() == ["am", "pm", "overall"]
    assert output["rows"].tolist() == [3, 2, 5]
    assert output["tradable_rows"].tolist() == [3, 1, 4]
    assert output["mean_edge_ticks"].tolist() == pytest.approx([2.0 / 3.0, -1.0, 0.25])
    assert output["total_edge_ticks"].tolist() == pytest.approx([2.0, -1.0, 1.0])
    assert output["max_drawdown_ticks"].tolist() == pytest.approx([2.0, 1.0, 2.0])
    assert output["hit_rate"].tolist() == pytest.approx([2.0 / 3.0, 0.0, 0.5])
    assert output["turnover_events"].tolist() == [1, 0, 1]
    assert output["turnover_rate"].tolist() == pytest.approx([1.0 / 3.0, 0.0, 0.25])
    assert output["path_risk_label"].tolist() == [
        "execution_path_stable",
        "execution_path_fragile",
        "execution_path_stable",
    ]


def test_queue_position_path_risk_scorecard_flags_side_flip_churn() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["long", "short", "long", "short"],
            "execution_adjusted_edge_ticks": [0.5, 0.5, 0.5, 0.5],
        }
    )

    output = queue_position_path_risk_scorecard(frame, max_turnover_rate=0.50)

    assert output.loc[0, "turnover_events"] == 3
    assert output.loc[0, "turnover_rate"] == pytest.approx(0.75)
    assert output.loc[0, "path_risk_label"] == "execution_path_fragile"


def test_queue_position_path_risk_scorecard_rejects_missing_execution_edge() -> None:
    with pytest.raises(ValueError, match="missing queue position path risk columns"):
        queue_position_path_risk_scorecard(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_path_risk_concentration_flags_event_path_crowding() -> None:
    scorecard = pd.DataFrame(
        {
            "path_id": ["macro", "auction", "lunch", "overall"],
            "tradable_rows": [10, 8, 2, 20],
            "total_edge_ticks": [9.0, 1.0, -2.0, 8.0],
            "max_drawdown_ticks": [0.5, 4.0, 0.5, 4.0],
            "path_risk_label": [
                "execution_path_stable",
                "execution_path_fragile",
                "execution_path_stable",
                "execution_path_fragile",
            ],
        }
    )

    concentration = queue_position_path_risk_concentration(
        scorecard,
        max_top_edge_share=0.70,
        max_top_drawdown_share=0.70,
        max_fragile_path_share=0.25,
    )

    assert concentration == {
        "paths": 3,
        "fragile_paths": 1,
        "fragile_path_share": pytest.approx(1.0 / 3.0),
        "positive_edge_paths": 2,
        "drawdown_paths": 3,
        "top_edge_path_id": "macro",
        "top_edge_share": pytest.approx(0.90),
        "edge_concentration_hhi": pytest.approx(0.82),
        "top_drawdown_path_id": "auction",
        "top_drawdown_share": pytest.approx(0.80),
        "drawdown_concentration_hhi": pytest.approx(0.66),
        "path_concentration_label": "queue_path_concentration_fragile",
        "review_reasons": "edge_concentration;drawdown_concentration;fragile_path_share",
    }


def test_queue_position_path_risk_concentration_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position path concentration columns"):
        queue_position_path_risk_concentration(pd.DataFrame({"path_id": ["overall"]}))


def test_queue_position_path_risk_release_gate_blocks_fragile_execution_paths() -> None:
    scorecard = pd.DataFrame(
        {
            "path_id": ["am", "pm", "overnight", "overall"],
            "rows": [4, 4, 4, 12],
            "tradable_rows": [4, 4, 4, 12],
            "abstain_rows": [0, 0, 0, 0],
            "mean_edge_ticks": [0.25, -0.25, 0.10, 0.03],
            "total_edge_ticks": [1.0, -1.0, 0.4, 0.4],
            "max_drawdown_ticks": [0.5, 3.0, 1.0, 3.2],
            "hit_rate": [0.75, 0.25, 0.50, 0.50],
            "turnover_events": [1, 3, 1, 5],
            "turnover_rate": [0.25, 0.75, 0.25, 5.0 / 12.0],
            "path_risk_label": [
                "execution_path_stable",
                "execution_path_fragile",
                "execution_path_stable",
                "execution_path_fragile",
            ],
        }
    )

    gate = queue_position_path_risk_release_gate(
        scorecard,
        max_fragile_path_share=0.25,
        max_overall_drawdown_ticks=2.0,
        min_overall_total_edge_ticks=1.0,
    )

    assert gate == {
        "paths": 3,
        "fragile_paths": 1,
        "fragile_path_share": pytest.approx(1.0 / 3.0),
        "total_tradable_rows": 12,
        "overall_total_edge_ticks": pytest.approx(0.4),
        "overall_max_drawdown_ticks": pytest.approx(3.2),
        "overall_turnover_rate": pytest.approx(5.0 / 12.0),
        "worst_path_id": "pm",
        "worst_path_risk_label": "execution_path_fragile",
        "path_risk_release_decision": "block",
        "path_risk_release_label": "queue_path_risk_release_blocked",
        "blocking_reasons": "fragile_path_share;overall_drawdown;overall_total_edge",
        "review_reasons": "none",
    }


def test_queue_position_path_risk_release_gate_passes_stable_paths() -> None:
    scorecard = pd.DataFrame(
        {
            "path_id": ["am", "overall"],
            "rows": [4, 4],
            "tradable_rows": [4, 4],
            "abstain_rows": [0, 0],
            "mean_edge_ticks": [0.25, 0.25],
            "total_edge_ticks": [1.0, 1.0],
            "max_drawdown_ticks": [0.2, 0.2],
            "hit_rate": [0.75, 0.75],
            "turnover_events": [1, 1],
            "turnover_rate": [0.25, 0.25],
            "path_risk_label": ["execution_path_stable", "execution_path_stable"],
        }
    )

    gate = queue_position_path_risk_release_gate(scorecard)

    assert gate["path_risk_release_decision"] == "pass"
    assert gate["path_risk_release_label"] == "queue_path_risk_release_pass"
    assert gate["blocking_reasons"] == "none"
    assert gate["review_reasons"] == "none"


def test_queue_position_path_risk_release_gate_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position path risk release gate columns"):
        queue_position_path_risk_release_gate(pd.DataFrame({"path_id": ["overall"]}))


def test_passive_fill_event_window_transition_matrix_tracks_edge_decay_by_regime_path() -> None:
    frame = pd.DataFrame(
        {
            "session": ["A", "A", "A", "B", "B"],
            "passive_fill_event_window_regime": ["pre_event", "event", "post_event", "event", "calm"],
            "passive_fill_event_side": ["long", "long", "long", "short", "none"],
            "passive_fill_event_toxicity_probability": [0.20, 0.40, 0.80, 0.10, 0.00],
            "execution_adjusted_edge_ticks": [0.60, 0.20, -0.40, 0.50, 0.10],
        }
    )

    matrix = passive_fill_event_window_transition_matrix(frame, group_cols="session")

    assert matrix["from_passive_fill_event_window_regime"].tolist() == [
        "event",
        "pre_event",
        "event",
    ]
    assert matrix["to_passive_fill_event_window_regime"].tolist() == [
        "post_event",
        "event",
        "calm",
    ]
    assert matrix["rows"].tolist() == [1, 1, 1]
    assert matrix["transition_share"].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert matrix["mean_edge_delta_ticks"].tolist() == pytest.approx([-0.60, -0.40, -0.40])
    assert matrix["to_negative_edge_share"].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert matrix["mean_to_passive_fill_event_toxicity_probability"].tolist() == pytest.approx(
        [0.80, 0.40, 0.00]
    )
    assert matrix["dominant_to_passive_fill_event_side"].tolist() == ["long", "long", "none"]


def test_passive_fill_event_window_transition_matrix_respects_group_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "session": ["A", "B"],
            "passive_fill_event_window_regime": ["pre_event", "post_event"],
            "passive_fill_event_side": ["long", "short"],
            "passive_fill_event_toxicity_probability": [0.20, 0.90],
            "execution_adjusted_edge_ticks": [0.60, -0.40],
        }
    )

    matrix = passive_fill_event_window_transition_matrix(frame, group_cols="session")

    assert matrix.empty


def test_passive_fill_event_window_transition_scorecard_blocks_toxic_decay_paths() -> None:
    matrix = pd.DataFrame(
        {
            "from_passive_fill_event_window_regime": ["event", "pre_event", "event"],
            "to_passive_fill_event_window_regime": ["post_event", "event", "calm"],
            "rows": [18, 10, 2],
            "transition_share": [0.60, 0.33, 0.07],
            "mean_from_execution_adjusted_edge_ticks": [0.25, 0.60, 0.15],
            "mean_to_execution_adjusted_edge_ticks": [-0.30, 0.20, 0.05],
            "mean_edge_delta_ticks": [-0.55, -0.40, -0.10],
            "to_negative_edge_share": [0.72, 0.10, 0.00],
            "mean_to_passive_fill_event_toxicity_probability": [0.81, 0.35, 0.05],
            "dominant_to_passive_fill_event_side": ["long", "long", "none"],
        }
    )

    scorecard = passive_fill_event_window_transition_scorecard(matrix)

    assert scorecard["transition_release_label"] == "block"
    assert scorecard["worst_transition_path"] == "event->post_event"
    assert scorecard["worst_path_rows"] == 18
    assert scorecard["worst_path_transition_share"] == pytest.approx(0.60)
    assert scorecard["worst_path_mean_edge_delta_ticks"] == pytest.approx(-0.55)
    assert scorecard["worst_path_to_negative_edge_share"] == pytest.approx(0.72)
    assert scorecard["worst_path_to_toxicity_probability"] == pytest.approx(0.81)
    assert scorecard["blocking_reasons"] == "toxic_event_post_event_decay"
    assert scorecard["review_reasons"] == "none"


def test_passive_fill_event_window_transition_scorecard_reviews_moderate_decay() -> None:
    matrix = pd.DataFrame(
        {
            "from_passive_fill_event_window_regime": ["pre_event"],
            "to_passive_fill_event_window_regime": ["event"],
            "rows": [12],
            "transition_share": [1.0],
            "mean_from_execution_adjusted_edge_ticks": [0.50],
            "mean_to_execution_adjusted_edge_ticks": [0.18],
            "mean_edge_delta_ticks": [-0.32],
            "to_negative_edge_share": [0.25],
            "mean_to_passive_fill_event_toxicity_probability": [0.62],
            "dominant_to_passive_fill_event_side": ["short"],
        }
    )

    scorecard = passive_fill_event_window_transition_scorecard(matrix)

    assert scorecard["transition_release_label"] == "review"
    assert scorecard["worst_transition_path"] == "pre_event->event"
    assert scorecard["blocking_reasons"] == "none"
    assert scorecard["review_reasons"] == "meaningful_transition_edge_decay"


def test_passive_fill_event_window_transition_scorecard_passes_empty_matrix() -> None:
    scorecard = passive_fill_event_window_transition_scorecard(pd.DataFrame())

    assert scorecard["transition_release_label"] == "pass"
    assert scorecard["observed_transition_paths"] == 0
    assert scorecard["total_transition_rows"] == 0
    assert scorecard["worst_transition_path"] == "none"


def test_queue_position_path_drawdown_episodes_identifies_underwater_runs() -> None:
    frame = pd.DataFrame(
        {
            "session": ["A", "A", "A", "A", "A", "A", "B", "B"],
            "best_execution_side": ["long", "long", "short", "short", "short", "long", "long", "long"],
            "execution_adjusted_edge_ticks": [1.0, -0.4, -0.8, 0.3, 1.1, -0.2, 0.5, -0.1],
            "passive_fill_event_window_regime": [
                "calm",
                "event",
                "event",
                "post_event",
                "post_event",
                "calm",
                "pre_event",
                "event",
            ],
        }
    )

    episodes = queue_position_path_drawdown_episodes(
        frame,
        group_cols="session",
        event_window_col="passive_fill_event_window_regime",
    )

    assert episodes["path_id"].tolist() == ["A", "A", "B"]
    assert episodes["episode_start_row"].tolist() == [1, 5, 7]
    assert episodes["episode_end_row"].tolist() == [4, 5, 7]
    assert episodes["trough_row"].tolist() == [2, 5, 7]
    assert episodes["episode_rows"].tolist() == [4, 1, 1]
    assert episodes["max_drawdown_ticks"].tolist() == pytest.approx([1.2, 0.2, 0.1])
    assert episodes["recovery_edge_ticks"].tolist() == pytest.approx([1.4, 0.0, 0.0])
    assert episodes["episode_turnover_events"].tolist() == [1, 0, 0]
    assert episodes["dominant_event_window_regime"].tolist() == ["event", "calm", "event"]
    assert episodes["episode_risk_label"].tolist() == [
        "path_drawdown_recovered",
        "path_drawdown_open",
        "path_drawdown_open",
    ]


def test_queue_position_path_drawdown_episodes_ignores_abstains_and_zero_drawdown() -> None:
    frame = pd.DataFrame(
        {
            "best_execution_side": ["abstain", "long", "long", "abstain"],
            "execution_adjusted_edge_ticks": [-10.0, 0.2, 0.1, -5.0],
        }
    )

    episodes = queue_position_path_drawdown_episodes(frame)

    assert episodes.empty
    assert episodes.columns.tolist() == [
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


def test_queue_position_path_drawdown_summary_flags_unrecovered_clustered_damage() -> None:
    episodes = pd.DataFrame(
        {
            "path_id": ["A", "A", "B", "C"],
            "episode_id": [0, 1, 0, 0],
            "episode_rows": [4, 2, 1, 3],
            "max_drawdown_ticks": [1.6, 0.4, 2.4, 0.2],
            "recovery_edge_ticks": [0.3, 0.0, 0.0, 0.5],
            "episode_total_edge_ticks": [-0.7, -0.4, -2.4, 0.1],
            "episode_turnover_events": [2, 0, 1, 0],
            "dominant_event_window_regime": ["event", "event", "post_event", "calm"],
            "episode_risk_label": [
                "path_drawdown_recovered",
                "path_drawdown_open",
                "path_drawdown_open",
                "path_drawdown_recovered",
            ],
        }
    )

    summary = queue_position_path_drawdown_summary(
        episodes,
        severe_drawdown_ticks=1.0,
        max_open_episode_share=0.25,
        max_top_regime_drawdown_share=0.55,
    )

    assert summary == {
        "episodes": 4,
        "paths_with_drawdown": 3,
        "open_episodes": 2,
        "open_episode_share": pytest.approx(0.50),
        "severe_episodes": 2,
        "severe_episode_share": pytest.approx(0.50),
        "mean_drawdown_ticks": pytest.approx(1.15),
        "max_drawdown_ticks": pytest.approx(2.4),
        "total_drawdown_ticks": pytest.approx(4.6),
        "total_recovery_edge_ticks": pytest.approx(0.8),
        "recovery_coverage_ratio": pytest.approx(0.8 / 4.6),
        "dominant_drawdown_regime": "post_event",
        "dominant_regime_drawdown_share": pytest.approx(2.4 / 4.6),
        "top_path_id": "B",
        "top_path_drawdown_share": pytest.approx(2.4 / 4.6),
        "drawdown_summary_label": "queue_drawdown_review",
        "blocking_reasons": "none",
        "review_reasons": "open_drawdown_share;severe_drawdown_share",
    }


def test_queue_position_path_drawdown_summary_blocks_concentrated_event_damage() -> None:
    episodes = pd.DataFrame(
        {
            "path_id": ["auction", "lunch"],
            "max_drawdown_ticks": [3.0, 0.5],
            "recovery_edge_ticks": [0.0, 0.1],
            "dominant_event_window_regime": ["event", "calm"],
            "episode_risk_label": ["path_drawdown_open", "path_drawdown_recovered"],
        }
    )

    summary = queue_position_path_drawdown_summary(
        episodes,
        severe_drawdown_ticks=2.0,
        max_drawdown_ticks=2.5,
        max_top_regime_drawdown_share=0.70,
    )

    assert summary["drawdown_summary_label"] == "queue_drawdown_blocked"
    assert summary["dominant_drawdown_regime"] == "event"
    assert summary["blocking_reasons"] == "max_drawdown;regime_drawdown_concentration"


def test_queue_position_path_drawdown_summary_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position path drawdown summary columns"):
        queue_position_path_drawdown_summary(pd.DataFrame({"path_id": ["overall"]}))


def test_queue_position_lcri_tail_adverse_selection_surface_flags_toxic_fill_pockets() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["calm", "calm", "calm", "calm", "stress", "stress", "stress"],
            "best_execution_side": ["long", "long", "long", "long", "short", "short", "abstain"],
            "lcri": [1.0, 2.0, 3.0, 4.0, -1.0, -3.0, 4.0],
            "bid_fill_probability": [0.80, 0.85, 0.90, 0.95, 0.10, 0.20, 0.99],
            "ask_fill_probability": [0.10, 0.20, 0.20, 0.20, 0.85, 0.90, 0.01],
            "bid_adverse_fill_probability": [0.20, 0.30, 0.95, 0.85, 0.05, 0.10, 0.50],
            "ask_adverse_fill_probability": [0.10, 0.10, 0.10, 0.10, 0.95, 0.30, 0.50],
            "bid_realized_fill": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "ask_realized_fill": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "execution_adjusted_edge_ticks": [0.50, 0.20, -0.20, -0.60, -0.30, 0.40, 1.0],
        }
    )

    surface = queue_position_lcri_tail_adverse_selection_surface(
        frame,
        lcri_bins=2,
        fill_probability_bins=1,
        max_abs_fill_residual=0.25,
        min_fill_minus_adverse_rate=0.0,
    )

    assert surface.columns.tolist() == [
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
    toxic = surface.loc[
        (surface["regime"] == "calm")
        & (surface["best_execution_side"] == "long")
        & (surface["lcri_tail_bin"] == 2)
    ].iloc[0]
    assert toxic["rows"] == 2
    assert toxic["mean_predicted_fill_probability"] == pytest.approx(0.925)
    assert toxic["realized_fill_rate"] == pytest.approx(0.50)
    assert toxic["mean_selected_adverse_probability"] == pytest.approx(0.90)
    assert toxic["fill_minus_adverse_rate"] == pytest.approx(-0.40)
    assert toxic["tail_adverse_selection_label"] == "tail_adverse_toxic"

    short = surface.loc[surface["best_execution_side"] == "short"].iloc[-1]
    assert short["tail_adverse_selection_label"] == "tail_adverse_publishable"


def test_queue_position_lcri_tail_adverse_selection_surface_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position LCRI tail adverse selection columns"):
        queue_position_lcri_tail_adverse_selection_surface(pd.DataFrame({"lcri": [1.0]}))


def test_queue_position_lcri_tail_adverse_selection_release_scorecard_blocks_toxic_tail() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["calm", "stress", "stress"],
            "best_execution_side": ["long", "long", "short"],
            "lcri_tail_bin": [5, 5, 4],
            "fill_probability_bin": [5, 4, 5],
            "rows": [12, 8, 5],
            "mean_abs_lcri": [4.2, 3.7, 2.4],
            "mean_predicted_fill_probability": [0.92, 0.80, 0.70],
            "realized_fill_rate": [0.40, 0.55, 0.78],
            "mean_selected_adverse_probability": [0.88, 0.70, 0.30],
            "fill_residual": [-0.52, -0.25, 0.08],
            "absolute_fill_residual": [0.52, 0.25, 0.08],
            "fill_minus_adverse_rate": [-0.48, -0.15, 0.48],
            "mean_execution_adjusted_edge_ticks": [-0.30, 0.10, 0.20],
            "tail_adverse_selection_label": [
                "tail_adverse_toxic",
                "tail_adverse_toxic",
                "tail_adverse_publishable",
            ],
        }
    )

    scorecard = queue_position_lcri_tail_adverse_selection_release_scorecard(
        surface,
        min_cell_rows=5,
        block_toxic_row_share=0.40,
        review_toxic_row_share=0.20,
        block_fill_minus_adverse_rate=0.0,
        review_fill_residual=0.20,
    )

    assert scorecard["tail_adverse_release_label"] == "block"
    assert scorecard["total_tail_rows"] == 25
    assert scorecard["eligible_tail_cells"] == 3
    assert scorecard["toxic_tail_row_share"] == pytest.approx(20 / 25)
    assert scorecard["candidate_weighted_fill_minus_adverse_rate"] == pytest.approx(
        ((-0.48 * 12) + (-0.15 * 8) + (0.48 * 5)) / 25
    )
    assert scorecard["worst_tail_cell"] == "calm:long:lcri_tail=5:fill_bin=5"
    assert scorecard["blocking_reasons"] == "toxic_tail_row_share,negative_tail_fill_minus_adverse"


def test_queue_position_lcri_tail_adverse_selection_release_scorecard_reviews_miscalibration() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["calm", "stress"],
            "best_execution_side": ["long", "short"],
            "lcri_tail_bin": [5, 5],
            "fill_probability_bin": [5, 4],
            "rows": [10, 10],
            "mean_abs_lcri": [4.0, 3.8],
            "mean_predicted_fill_probability": [0.85, 0.75],
            "realized_fill_rate": [0.60, 0.70],
            "mean_selected_adverse_probability": [0.45, 0.40],
            "fill_residual": [-0.25, -0.05],
            "absolute_fill_residual": [0.25, 0.05],
            "fill_minus_adverse_rate": [0.15, 0.30],
            "mean_execution_adjusted_edge_ticks": [0.05, 0.10],
            "tail_adverse_selection_label": [
                "tail_fill_overstated",
                "tail_adverse_publishable",
            ],
        }
    )

    scorecard = queue_position_lcri_tail_adverse_selection_release_scorecard(
        surface,
        min_cell_rows=5,
        block_toxic_row_share=0.50,
        review_fill_residual=0.20,
    )

    assert scorecard["tail_adverse_release_label"] == "review"
    assert scorecard["review_reasons"] == "tail_fill_miscalibration"
    assert scorecard["worst_tail_cell"] == "calm:long:lcri_tail=5:fill_bin=5"


def test_queue_position_lcri_tail_adverse_selection_release_scorecard_rejects_missing_columns() -> None:
    with pytest.raises(
        ValueError,
        match="missing queue position LCRI tail adverse selection release scorecard columns",
    ):
        queue_position_lcri_tail_adverse_selection_release_scorecard(pd.DataFrame({"rows": [1]}))


def test_queue_position_trade_confirmation_surface_flags_cancel_driven_overprediction() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["calm", "calm", "stress", "stress"],
            "best_execution_side": ["long", "long", "short", "short"],
            "bid_fill_probability": [0.80, 0.70, 0.20, 0.10],
            "ask_fill_probability": [0.20, 0.10, 0.90, 0.85],
            "bid_trade_confirmed_fill": [1.0, 0.0, 0.0, 0.0],
            "ask_trade_confirmed_fill": [0.0, 0.0, 0.0, 1.0],
            "bid_trade_confirmed_fill_latency": [0.20, float("nan"), float("nan"), float("nan")],
            "ask_trade_confirmed_fill_latency": [float("nan"), float("nan"), float("nan"), 1.40],
            "bid_queue_advance_without_trade": [0.0, 55.0, 0.0, 0.0],
            "ask_queue_advance_without_trade": [0.0, 0.0, 60.0, 0.0],
            "bid_queue_clear_size": [50.0, 50.0, 50.0, 50.0],
            "ask_queue_clear_size": [60.0, 60.0, 60.0, 60.0],
            "bid_queue_clear_share": [0.20, 0.40, 0.10, 0.10],
            "ask_queue_clear_share": [0.30, 0.30, 0.70, 0.90],
        }
    )

    surface = queue_position_trade_confirmation_surface(
        frame,
        bins=2,
        regime_col="regime",
        max_latency=1.0,
    )

    assert surface["queue_clear_bucket"].tolist() == ["q01", "q02"]
    calm_low = surface.iloc[0]
    assert calm_low["regime"] == "calm"
    assert calm_low["rows"] == 2
    assert calm_low["trade_confirmed_fill_rate"] == pytest.approx(0.5)
    assert calm_low["cancel_only_clear_rate"] == pytest.approx(0.5)
    assert calm_low["mean_trade_confirmed_fill_latency"] == pytest.approx(0.2)
    assert calm_low["stale_trade_confirmed_fill_share"] == pytest.approx(0.0)
    assert calm_low["confirmation_shortfall"] == pytest.approx(0.25)
    assert calm_low["confirmation_surface_label"] == "high_prediction_not_trade_confirmed"

    stress_high = surface.iloc[1]
    assert stress_high["regime"] == "stress"
    assert stress_high["rows"] == 2
    assert stress_high["trade_confirmed_fill_rate"] == pytest.approx(0.5)
    assert stress_high["cancel_only_clear_rate"] == pytest.approx(0.5)
    assert stress_high["mean_trade_confirmed_fill_latency"] == pytest.approx(1.4)
    assert stress_high["stale_trade_confirmed_fill_share"] == pytest.approx(1.0)
    assert stress_high["confirmation_surface_label"] == "latency_risk"


def test_queue_position_trade_confirmation_surface_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position trade confirmation surface columns"):
        queue_position_trade_confirmation_surface(pd.DataFrame({"best_execution_side": ["long"]}))


def test_queue_position_trade_confirmation_release_scorecard_blocks_cancel_driven_overprediction() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["calm", "stress", "stress"],
            "queue_clear_bucket": ["q01", "q01", "q02"],
            "rows": [20, 30, 10],
            "mean_queue_clear_share": [0.25, 0.50, 0.80],
            "mean_predicted_fill_probability": [0.80, 0.70, 0.60],
            "trade_confirmed_fill_rate": [0.35, 0.20, 0.55],
            "cancel_only_clear_rate": [0.60, 0.50, 0.05],
            "mean_trade_confirmed_fill_latency": [0.10, 0.20, 0.15],
            "stale_trade_confirmed_fill_share": [0.00, 0.05, 0.00],
            "confirmation_calibration_error": [-0.45, -0.50, -0.05],
            "confirmation_shortfall": [0.45, 0.50, 0.05],
            "confirmation_surface_label": [
                "high_prediction_not_trade_confirmed",
                "cancel_driven_queue_clearance",
                "trade_confirmed_execution_ok",
            ],
        }
    )

    scorecard = queue_position_trade_confirmation_release_scorecard(
        surface,
        min_cell_rows=5,
        max_confirmation_shortfall=0.20,
        max_cancel_only_clear_rate=0.15,
    )

    assert scorecard["trade_confirmation_release_label"] == "block"
    assert scorecard["publishable"] is False
    assert scorecard["evaluated_cells"] == 3
    assert scorecard["supported_cells"] == 3
    assert scorecard["total_rows"] == 60
    assert scorecard["worst_confirmation_cell"] == "stress:q01"
    assert scorecard["max_confirmation_shortfall"] == pytest.approx(0.50)
    assert scorecard["max_cancel_only_clear_rate"] == pytest.approx(0.60)
    assert scorecard["blocking_reasons"] == "confirmation_shortfall,cancel_only_queue_clearance"


def test_queue_position_trade_confirmation_release_scorecard_reviews_stale_confirmed_fills() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["calm", "calm"],
            "queue_clear_bucket": ["q01", "q02"],
            "rows": [12, 8],
            "mean_queue_clear_share": [0.20, 0.60],
            "mean_predicted_fill_probability": [0.60, 0.50],
            "trade_confirmed_fill_rate": [0.55, 0.45],
            "cancel_only_clear_rate": [0.02, 0.03],
            "mean_trade_confirmed_fill_latency": [0.20, 1.50],
            "stale_trade_confirmed_fill_share": [0.05, 0.40],
            "confirmation_calibration_error": [-0.05, -0.05],
            "confirmation_shortfall": [0.05, 0.05],
            "confirmation_surface_label": ["trade_confirmed_execution_ok", "latency_risk"],
        }
    )

    scorecard = queue_position_trade_confirmation_release_scorecard(
        surface,
        min_cell_rows=5,
        max_stale_trade_confirmed_fill_share=0.25,
    )

    assert scorecard["trade_confirmation_release_label"] == "review"
    assert scorecard["publishable"] is False
    assert scorecard["review_reasons"] == "stale_trade_confirmed_fills"
    assert scorecard["worst_confirmation_cell"] == "calm:q02"
    assert scorecard["max_stale_trade_confirmed_fill_share"] == pytest.approx(0.40)


def test_queue_position_trade_confirmation_release_scorecard_passes_confirmed_surface() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["calm", "stress"],
            "queue_clear_bucket": ["q01", "q02"],
            "rows": [20, 20],
            "mean_queue_clear_share": [0.20, 0.60],
            "mean_predicted_fill_probability": [0.55, 0.40],
            "trade_confirmed_fill_rate": [0.57, 0.38],
            "cancel_only_clear_rate": [0.02, 0.04],
            "mean_trade_confirmed_fill_latency": [0.12, 0.20],
            "stale_trade_confirmed_fill_share": [0.00, 0.05],
            "confirmation_calibration_error": [0.02, -0.02],
            "confirmation_shortfall": [-0.02, 0.02],
            "confirmation_surface_label": [
                "trade_confirmed_execution_ok",
                "trade_confirmed_execution_ok",
            ],
        }
    )

    scorecard = queue_position_trade_confirmation_release_scorecard(surface, min_cell_rows=5)

    assert scorecard["trade_confirmation_release_label"] == "pass"
    assert scorecard["publishable"] is True
    assert scorecard["blocking_reasons"] == "none"
    assert scorecard["review_reasons"] == "none"


def test_queue_position_trade_confirmation_regime_scorecard_prioritizes_regime_specific_blockers() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["calm", "calm", "stress", "stress"],
            "queue_clear_bucket": ["q01", "q02", "q01", "q02"],
            "rows": [20, 20, 30, 10],
            "mean_queue_clear_share": [0.20, 0.50, 0.25, 0.80],
            "mean_predicted_fill_probability": [0.52, 0.40, 0.80, 0.70],
            "trade_confirmed_fill_rate": [0.50, 0.42, 0.20, 0.40],
            "cancel_only_clear_rate": [0.02, 0.03, 0.55, 0.30],
            "mean_trade_confirmed_fill_latency": [0.10, 0.15, 0.30, 1.50],
            "stale_trade_confirmed_fill_share": [0.00, 0.05, 0.10, 0.40],
            "confirmation_calibration_error": [-0.02, 0.02, -0.60, -0.30],
            "confirmation_shortfall": [0.02, -0.02, 0.60, 0.30],
            "confirmation_surface_label": [
                "trade_confirmed_execution_ok",
                "trade_confirmed_execution_ok",
                "cancel_driven_queue_clearance",
                "latency_risk",
            ],
        }
    )

    scorecard = queue_position_trade_confirmation_regime_scorecard(
        surface,
        min_cell_rows=5,
        max_confirmation_shortfall=0.20,
        max_cancel_only_clear_rate=0.15,
        max_stale_trade_confirmed_fill_share=0.25,
    )

    assert scorecard["regime"].tolist() == ["stress", "calm"]
    stress = scorecard.iloc[0]
    assert stress["trade_confirmation_regime_label"] == "block"
    assert not bool(stress["publishable"])
    assert stress["rows"] == 40
    assert stress["weighted_confirmation_shortfall"] == pytest.approx(0.525)
    assert stress["weighted_cancel_only_clear_rate"] == pytest.approx(0.4875)
    assert stress["max_stale_trade_confirmed_fill_share"] == pytest.approx(0.40)
    assert stress["worst_confirmation_cell"] == "stress:q01"
    assert stress["blocking_reasons"] == "confirmation_shortfall,cancel_only_queue_clearance"
    assert stress["review_reasons"] == "stale_trade_confirmed_fills"
    assert stress["regime_priority_rank"] == 1

    calm = scorecard.iloc[1]
    assert calm["trade_confirmation_regime_label"] == "pass"
    assert bool(calm["publishable"])
    assert calm["blocking_reasons"] == "none"
    assert calm["review_reasons"] == "none"
    assert calm["regime_priority_rank"] == 2


def test_queue_position_trade_confirmation_regime_scorecard_rejects_missing_columns() -> None:
    with pytest.raises(
        ValueError,
        match="missing queue position trade confirmation regime scorecard columns",
    ):
        queue_position_trade_confirmation_regime_scorecard(pd.DataFrame({"rows": [1]}))


def test_queue_position_trade_confirmation_release_scorecard_rejects_missing_columns() -> None:
    with pytest.raises(
        ValueError,
        match="missing queue position trade confirmation release scorecard columns",
    ):
        queue_position_trade_confirmation_release_scorecard(pd.DataFrame({"rows": [1]}))


def test_queue_position_unfilled_opportunity_curve_quantifies_missed_lcri_edge() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["calm", "calm", "stress", "stress"],
            "passive_fill_event_window": ["pre_event", "pre_event", "event", "event"],
            "best_execution_side": ["long", "long", "short", "short"],
            "lcri": [1.0, 3.0, -2.0, -4.0],
            "long_net_return_ticks": [2.0, 4.0, -1.0, 0.0],
            "short_net_return_ticks": [-2.0, -4.0, 3.0, 5.0],
            "bid_realized_fill": [1.0, 0.0, 0.0, 0.0],
            "ask_realized_fill": [0.0, 0.0, 1.0, 0.0],
            "bid_fill_probability": [0.80, 0.20, 0.10, 0.10],
            "ask_fill_probability": [0.10, 0.10, 0.70, 0.30],
            "execution_adjusted_edge_ticks": [2.0, 0.0, 3.0, 0.0],
        }
    )

    curve = queue_position_unfilled_opportunity_curve(frame, lcri_bins=2)

    assert curve["lcri_tail_bin"].tolist() == [1, 2]
    assert curve["rows"].tolist() == [2, 2]
    assert curve["mean_abs_lcri"].tolist() == pytest.approx([1.5, 3.5])
    assert curve["realized_fill_rate"].tolist() == pytest.approx([1.0, 0.0])
    assert curve["mean_signal_edge_ticks"].tolist() == pytest.approx([2.5, 4.5])
    assert curve["mean_captured_edge_ticks"].tolist() == pytest.approx([2.5, 0.0])
    assert curve["mean_unfilled_opportunity_ticks"].tolist() == pytest.approx([0.0, 4.5])
    assert curve["edge_capture_rate"].tolist() == pytest.approx([1.0, 0.0])
    assert curve["unfilled_opportunity_label"].tolist() == [
        "opportunity_captured",
        "unfilled_tail_opportunity",
    ]


def test_queue_position_unfilled_opportunity_curve_can_group_by_event_window() -> None:
    frame = pd.DataFrame(
        {
            "passive_fill_event_window": ["pre_event", "pre_event", "event", "event"],
            "best_execution_side": ["long", "long", "short", "short"],
            "lcri": [1.0, 3.0, -2.0, -4.0],
            "long_net_return_ticks": [2.0, 4.0, -1.0, 0.0],
            "short_net_return_ticks": [-2.0, -4.0, 3.0, 5.0],
            "bid_realized_fill": [1.0, 0.0, 0.0, 0.0],
            "ask_realized_fill": [0.0, 0.0, 1.0, 0.0],
            "bid_fill_probability": [0.80, 0.20, 0.10, 0.10],
            "ask_fill_probability": [0.10, 0.10, 0.70, 0.30],
        }
    )

    curve = queue_position_unfilled_opportunity_curve(
        frame,
        lcri_bins=1,
        group_cols="passive_fill_event_window",
    )

    assert curve["passive_fill_event_window"].tolist() == ["event", "pre_event"]
    assert curve["rows"].tolist() == [2, 2]
    assert curve["realized_fill_rate"].tolist() == pytest.approx([0.5, 0.5])
    assert curve["mean_unfilled_opportunity_ticks"].tolist() == pytest.approx([2.5, 2.0])
    assert curve["unfilled_opportunity_label"].tolist() == [
        "unfilled_tail_opportunity",
        "unfilled_tail_opportunity",
    ]


def test_queue_position_unfilled_opportunity_scorecard_blocks_tail_edge_nonfills() -> None:
    curve = pd.DataFrame(
        {
            "passive_fill_event_window": ["pre_event", "event", "event"],
            "lcri_tail_bin": [1, 1, 2],
            "rows": [30, 20, 10],
            "mean_abs_lcri": [1.5, 2.0, 4.0],
            "mean_predicted_fill_probability": [0.70, 0.80, 0.65],
            "realized_fill_rate": [0.65, 0.30, 0.10],
            "mean_signal_edge_ticks": [1.0, 2.0, 5.0],
            "mean_captured_edge_ticks": [0.65, 0.60, 0.50],
            "mean_unfilled_opportunity_ticks": [0.35, 1.40, 4.50],
            "edge_capture_rate": [0.65, 0.30, 0.10],
            "unfilled_opportunity_share": [0.35, 0.70, 0.90],
            "unfilled_opportunity_label": [
                "opportunity_captured",
                "unfilled_tail_opportunity",
                "unfilled_tail_opportunity",
            ],
        }
    )

    scorecard = queue_position_unfilled_opportunity_scorecard(
        curve,
        min_tail_bin=2,
        max_tail_unfilled_opportunity_share=0.50,
        min_tail_edge_capture_rate=0.40,
        min_tail_rows=5,
    )

    assert scorecard["unfilled_opportunity_release_label"] == "block"
    assert scorecard["publishable"] is False
    assert scorecard["evaluated_cells"] == 3
    assert scorecard["tail_cells"] == 1
    assert scorecard["tail_rows"] == 10
    assert scorecard["max_tail_unfilled_opportunity_share"] == pytest.approx(0.90)
    assert scorecard["min_tail_edge_capture_rate"] == pytest.approx(0.10)
    assert scorecard["worst_tail_cell"] == "event:tail_2"
    assert scorecard["blocking_reasons"] == "tail_opportunity_share,tail_capture_shortfall"


def test_queue_position_unfilled_opportunity_scorecard_reviews_thin_tail_evidence() -> None:
    curve = pd.DataFrame(
        {
            "lcri_tail_bin": [1, 2],
            "rows": [25, 3],
            "mean_abs_lcri": [1.0, 3.0],
            "mean_predicted_fill_probability": [0.60, 0.55],
            "realized_fill_rate": [0.60, 0.50],
            "mean_signal_edge_ticks": [1.0, 2.0],
            "mean_captured_edge_ticks": [0.60, 1.0],
            "mean_unfilled_opportunity_ticks": [0.40, 1.0],
            "edge_capture_rate": [0.60, 0.50],
            "unfilled_opportunity_share": [0.40, 0.50],
            "unfilled_opportunity_label": ["opportunity_captured", "partial_opportunity_capture"],
        }
    )

    scorecard = queue_position_unfilled_opportunity_scorecard(curve, min_tail_bin=2, min_tail_rows=5)

    assert scorecard["unfilled_opportunity_release_label"] == "review"
    assert scorecard["publishable"] is False
    assert scorecard["review_reasons"] == "thin_tail_opportunity_evidence"


def test_queue_position_unfilled_opportunity_scorecard_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing queue position unfilled opportunity scorecard columns"):
        queue_position_unfilled_opportunity_scorecard(pd.DataFrame({"rows": [1]}))


def test_queue_position_fill_monotonicity_scorecard_flags_deeper_queue_fill_inversions() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["calm", "calm", "calm", "calm", "stress", "stress"],
            "best_execution_side": ["long", "long", "long", "long", "short", "short"],
            "queue_share_bin": [1, 1, 2, 2, 1, 2],
            "fill_probability_bin": [1, 2, 1, 2, 1, 1],
            "rows": [10, 10, 10, 10, 8, 8],
            "mean_queue_share": [0.10, 0.15, 0.55, 0.60, 0.20, 0.70],
            "mean_predicted_fill_probability": [0.40, 0.50, 0.65, 0.75, 0.70, 0.62],
            "realized_fill_rate": [0.35, 0.45, 0.70, 0.80, 0.65, 0.60],
            "calibration_error": [-0.05, -0.05, 0.05, 0.05, -0.05, -0.02],
            "absolute_calibration_error": [0.05, 0.05, 0.05, 0.05, 0.05, 0.02],
            "brier_score": [0.20, 0.20, 0.18, 0.18, 0.16, 0.15],
            "mean_execution_adjusted_edge_ticks": [0.5, 0.6, -0.2, -0.1, 0.4, 0.3],
        }
    )

    scorecard = queue_position_fill_monotonicity_scorecard(surface, inversion_tolerance=0.05)

    calm = scorecard[scorecard["regime"] == "calm"].iloc[0]
    stress = scorecard[scorecard["regime"] == "stress"].iloc[0]
    assert calm["queue_steps"] == 1
    assert calm["predicted_fill_inversions"] == 1
    assert calm["realized_fill_inversions"] == 1
    assert calm["max_predicted_fill_inversion"] == pytest.approx(0.25)
    assert calm["max_realized_fill_inversion"] == pytest.approx(0.35)
    assert calm["monotonicity_label"] == "queue_fill_monotonicity_block"
    assert stress["monotonicity_label"] == "queue_fill_monotonicity_pass"


def test_queue_position_fill_monotonicity_scorecard_reviews_thin_queue_ladder() -> None:
    surface = pd.DataFrame(
        {
            "regime": ["calm"],
            "best_execution_side": ["long"],
            "queue_share_bin": [1],
            "fill_probability_bin": [1],
            "rows": [10],
            "mean_queue_share": [0.10],
            "mean_predicted_fill_probability": [0.50],
            "realized_fill_rate": [0.45],
        }
    )

    scorecard = queue_position_fill_monotonicity_scorecard(surface)

    assert scorecard["queue_steps"].tolist() == [0]
    assert scorecard["monotonicity_label"].tolist() == ["queue_fill_monotonicity_review"]


def test_passive_fill_event_window_transition_stability_flags_holdout_edge_decay() -> None:
    train = pd.DataFrame(
        {
            "from_passive_fill_event_window_regime": ["event", "pre_event"],
            "to_passive_fill_event_window_regime": ["post_event", "event"],
            "rows": [40, 60],
            "transition_share": [0.40, 0.60],
            "mean_edge_delta_ticks": [-0.10, 0.20],
            "to_negative_edge_share": [0.20, 0.10],
            "mean_to_passive_fill_event_toxicity_probability": [0.30, 0.20],
        }
    )
    heldout = pd.DataFrame(
        {
            "from_passive_fill_event_window_regime": ["event", "pre_event"],
            "to_passive_fill_event_window_regime": ["post_event", "event"],
            "rows": [50, 50],
            "transition_share": [0.50, 0.50],
            "mean_edge_delta_ticks": [-0.90, 0.10],
            "to_negative_edge_share": [0.80, 0.20],
            "mean_to_passive_fill_event_toxicity_probability": [0.85, 0.25],
        }
    )

    stability = passive_fill_event_window_transition_stability(train, heldout)

    assert stability["transition_path"].tolist() == ["event->post_event", "pre_event->event"]
    event_post = stability.iloc[0]
    assert event_post["mean_edge_delta_ticks_train"] == pytest.approx(-0.10)
    assert event_post["mean_edge_delta_ticks_heldout"] == pytest.approx(-0.90)
    assert event_post["edge_delta_drift_ticks"] == pytest.approx(-0.80)
    assert event_post["negative_edge_share_drift"] == pytest.approx(0.60)
    assert event_post["toxicity_probability_drift"] == pytest.approx(0.55)
    assert event_post["transition_stability_label"] == "transition_stability_block"


def test_passive_fill_event_window_transition_stability_scorecard_blocks_unstable_event_post_path() -> None:
    stability = pd.DataFrame(
        {
            "transition_path": ["event->post_event", "pre_event->event"],
            "rows_train": [40, 60],
            "rows_heldout": [50, 50],
            "transition_share_train": [0.40, 0.60],
            "transition_share_heldout": [0.50, 0.50],
            "mean_edge_delta_ticks_train": [-0.10, 0.20],
            "mean_edge_delta_ticks_heldout": [-0.90, 0.10],
            "edge_delta_drift_ticks": [-0.80, -0.10],
            "negative_edge_share_drift": [0.60, 0.10],
            "toxicity_probability_drift": [0.55, 0.05],
            "transition_stability_label": ["transition_stability_block", "transition_stability_pass"],
        }
    )

    scorecard = passive_fill_event_window_transition_stability_scorecard(stability)

    assert scorecard["transition_stability_release_label"] == "block"
    assert scorecard["publishable"] is False
    assert scorecard["evaluated_transition_paths"] == 2
    assert scorecard["blocked_transition_paths"] == 1
    assert scorecard["worst_transition_path"] == "event->post_event"
    assert scorecard["worst_edge_delta_drift_ticks"] == pytest.approx(-0.80)
    assert scorecard["blocking_reasons"] == "event_post_holdout_decay,unstable_transition_paths"
