import pandas as pd
import pytest

from lcri_lab.publishability import PublishabilityConfig, add_publishability_gate


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


def test_publishability_config_rejects_invalid_probability_threshold() -> None:
    with pytest.raises(ValueError, match="probability_threshold"):
        PublishabilityConfig(probability_threshold=0.2)
