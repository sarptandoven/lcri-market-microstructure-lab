import pandas as pd
import pytest

from lcri_lab.execution import (
    FillProbabilityConfig,
    add_execution_adjusted_edge,
    add_passive_fill_probabilities,
    add_queue_position_features,
    execution_adjusted_edge_summary,
    passive_fill_calibration_curve,
    passive_fill_calibration_summary,
    passive_fill_event_regime_summary,
    passive_fill_event_toxicity_scorecard,
    passive_fill_event_window_diagnostics,
    execution_publishability_review_packet,
    passive_fill_edge_curve,
    queue_position_capacity_frontier,
    queue_position_capacity_stability,
    queue_position_edge_decay,
    queue_position_fill_surface,
    queue_position_fraction_sweep,
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
