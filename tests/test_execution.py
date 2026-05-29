import pandas as pd
import pytest

from lcri_lab.execution import (
    FillProbabilityConfig,
    add_execution_adjusted_edge,
    add_passive_fill_probabilities,
    add_queue_position_features,
    execution_adjusted_edge_summary,
    passive_fill_edge_curve,
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


def test_execution_functions_reject_non_finite_inputs() -> None:
    frame = _book_frame()
    frame.loc[0, "bid_sz_1"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        add_queue_position_features(frame, levels=2)
