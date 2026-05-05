import pandas as pd
import pytest

from lcri_lab.labels import add_transaction_cost_labels


def test_add_transaction_cost_labels_accounts_for_costs() -> None:
    frame = pd.DataFrame(
        {
            "mid": [100.00, 100.00, 100.00],
            "next_mid": [100.03, 99.97, 100.005],
        }
    )

    labeled = add_transaction_cost_labels(frame, tick_size=0.01, cost_ticks=1.0)

    assert labeled["gross_return_ticks"].tolist() == pytest.approx([3.0, -3.0, 0.5])
    assert labeled["long_net_return_ticks"].tolist() == pytest.approx([2.0, -4.0, -0.5])
    assert labeled["short_net_return_ticks"].tolist() == pytest.approx([-4.0, 2.0, -1.5])
    assert labeled["cost_aware_direction"].tolist() == [1, 0, -1]


def test_add_transaction_cost_labels_rejects_non_finite_prices() -> None:
    frame = pd.DataFrame({"mid": [100.0], "next_mid": [float("nan")]})

    with pytest.raises(ValueError, match="finite"):
        add_transaction_cost_labels(frame, tick_size=0.01)

def test_add_transaction_cost_labels_rejects_missing_prices() -> None:
    with pytest.raises(ValueError, match="missing transaction label columns"):
        add_transaction_cost_labels(pd.DataFrame({"mid": [100.0]}), tick_size=0.01)

def test_add_transaction_cost_labels_rejects_invalid_tick_size() -> None:
    frame = pd.DataFrame({"mid": [100.0], "next_mid": [100.0]})

    with pytest.raises(ValueError, match="tick_size"):
        add_transaction_cost_labels(frame, tick_size=0.0)

def test_add_transaction_cost_labels_rejects_invalid_costs() -> None:
    frame = pd.DataFrame({"mid": [1.0], "next_mid": [1.0]})

    for cost_ticks in [-1.0, float("nan")]:
        with pytest.raises(ValueError, match="cost_ticks"):
            add_transaction_cost_labels(frame, tick_size=0.01, cost_ticks=cost_ticks)
