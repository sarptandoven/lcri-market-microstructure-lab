import pandas as pd
import pytest

from lcri_lab.publishability import (
    PublishabilityConfig,
    add_publishability_gate,
    publishability_margin_diagnostics,
    publishability_margin_summary,
)


def test_publishability_margin_diagnostics_quantifies_threshold_frontier() -> None:
    frame = pd.DataFrame(
        {
            "lcri_probability": [0.66, 0.34, 0.54],
            "long_net_return_ticks": [1.00, -0.20, 0.61],
            "short_net_return_ticks": [-0.20, 0.95, 0.50],
        }
    )

    output = publishability_margin_diagnostics(
        frame,
        config=PublishabilityConfig(
            min_edge_ticks=0.50,
            probability_threshold=0.60,
            latency_penalty_ticks=0.05,
        ),
    )

    assert output["preferred_side"].tolist() == ["long", "short", "long"]
    assert output["publishability_margin"].tolist() == pytest.approx([0.06, 0.06, -0.06])
    assert output["frontier_distance"].tolist() == pytest.approx([0.06, 0.06, 0.06])
    assert output["is_threshold_fragile"].tolist() == [False, False, False]


def test_publishability_margin_diagnostics_marks_near_frontier_rows() -> None:
    frame = pd.DataFrame(
        {
            "lcri_probability": [0.59],
            "long_net_return_ticks": [0.54],
            "short_net_return_ticks": [0.10],
        }
    )

    output = publishability_margin_diagnostics(
        frame,
        config=PublishabilityConfig(min_edge_ticks=0.50, probability_threshold=0.60),
    )

    assert output.loc[0, "publishability_margin"] == pytest.approx(-0.01)
    assert bool(output.loc[0, "is_threshold_fragile"])


def test_publishability_margin_summary_counts_frontier_risk() -> None:
    diagnostics = pd.DataFrame(
        {
            "preferred_side": ["long", "short", "long"],
            "publishability_margin": [0.10, -0.02, -0.20],
            "frontier_distance": [0.10, 0.02, 0.20],
            "is_threshold_fragile": [False, True, False],
        }
    )

    summary = publishability_margin_summary(diagnostics)

    assert summary == {
        "rows": 3,
        "publishable_margin_rows": 1,
        "abstain_margin_rows": 2,
        "threshold_fragile_rows": 1,
        "threshold_fragile_share": pytest.approx(1 / 3),
        "minimum_frontier_distance": 0.02,
        "closest_frontier_side": "short",
    }


def test_publishability_margin_summary_rejects_incomplete_diagnostics() -> None:
    with pytest.raises(ValueError, match="missing publishability diagnostic"):
        publishability_margin_summary(pd.DataFrame({"preferred_side": ["long"]}))


def test_publishability_gate_selects_long_short_and_abstain() -> None:
    frame = pd.DataFrame(
        {
            "lcri_probability": [0.72, 0.22, 0.53],
            "long_net_return_ticks": [2.0, -2.0, 0.4],
            "short_net_return_ticks": [-2.0, 1.5, 0.3],
        }
    )

    gated = add_publishability_gate(
        frame,
        config=PublishabilityConfig(
            min_edge_ticks=0.5,
            probability_threshold=0.6,
            crowding_penalty_ticks=0.25,
            latency_penalty_ticks=0.25,
        ),
    )

    assert gated["publishable_side"].tolist() == ["long", "short", "abstain"]
    assert gated["is_publishable"].tolist() == [True, True, False]
    assert gated["publishable_edge_ticks"].tolist() == pytest.approx([1.5, 1.0, -0.1])


def test_publishability_gate_requires_cost_aware_columns() -> None:
    with pytest.raises(ValueError, match="missing publishability columns"):
        add_publishability_gate(pd.DataFrame({"lcri_probability": [0.7]}))


def test_publishability_gate_rejects_non_finite_inputs() -> None:
    frame = pd.DataFrame(
        {
            "lcri_probability": [float("nan")],
            "long_net_return_ticks": [1.0],
            "short_net_return_ticks": [-1.0],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        add_publishability_gate(frame)


@pytest.mark.parametrize("threshold", [0.2, 1.1])
def test_publishability_config_rejects_invalid_probability_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="probability_threshold"):
        PublishabilityConfig(probability_threshold=threshold)


def test_publishability_config_rejects_non_finite_controls() -> None:
    with pytest.raises(ValueError, match="min_edge_ticks"):
        PublishabilityConfig(min_edge_ticks=float("nan"))
