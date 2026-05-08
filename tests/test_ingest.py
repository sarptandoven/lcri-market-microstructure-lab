import pandas as pd
import pytest

from lcri_lab.ingest import add_l2_state_features, normalize_l2_snapshots


def _raw_snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bid_px_1": [99.99, 100.00],
            "ask_px_1": [100.01, 100.03],
            "bid_sz_1": [10.0, 12.0],
            "ask_sz_1": [9.0, 11.0],
        }
    )


def test_normalize_l2_snapshots_derives_top_of_book_fields() -> None:
    normalized = normalize_l2_snapshots(_raw_snapshots(), tick_size=0.01, levels=1)

    assert normalized["mid"].tolist() == pytest.approx([100.0, 100.015])
    assert normalized["spread"].tolist() == pytest.approx([0.02, 0.03])
    assert normalized["spread_ticks"].tolist() == pytest.approx([2.0, 3.0])
    assert normalized["next_mid"].tolist() == pytest.approx([100.015, 100.015])
    assert normalized["regime"].tolist() == ["unclassified", "unclassified"]


def test_normalize_l2_snapshots_can_derive_state_features() -> None:
    normalized = normalize_l2_snapshots(
        _raw_snapshots(), tick_size=0.01, levels=1, derive_state=True
    )

    assert normalized["volatility"].tolist() == pytest.approx([0.0, 0.000106066])
    assert normalized["replenishment_rate"].tolist() == pytest.approx([1.0, 0.826087])


def test_normalize_l2_snapshots_rejects_crossed_quotes() -> None:
    frame = _raw_snapshots()
    frame.loc[0, "ask_px_1"] = 99.98

    with pytest.raises(ValueError, match="best ask"):
        normalize_l2_snapshots(frame, tick_size=0.01, levels=1)


def test_normalize_l2_snapshots_rejects_inverted_price_ladder() -> None:
    frame = _raw_snapshots().assign(
        bid_px_2=[100.0, 99.99],
        ask_px_2=[100.02, 100.04],
        bid_sz_2=[8.0, 8.0],
        ask_sz_2=[7.0, 7.0],
    )

    with pytest.raises(ValueError, match="bid prices"):
        normalize_l2_snapshots(frame, tick_size=0.01, levels=2)


def test_normalize_l2_snapshots_rejects_inverted_ask_ladder() -> None:
    frame = _raw_snapshots().assign(
        bid_px_2=[99.98, 99.99],
        ask_px_2=[100.0, 100.04],
        bid_sz_2=[8.0, 8.0],
        ask_sz_2=[7.0, 7.0],
    )

    with pytest.raises(ValueError, match="ask prices"):
        normalize_l2_snapshots(frame, tick_size=0.01, levels=2)


def test_normalize_l2_snapshots_rejects_non_integer_levels() -> None:
    with pytest.raises(ValueError, match="integer"):
        normalize_l2_snapshots(_raw_snapshots(), tick_size=0.01, levels=1.5)


def test_add_l2_state_features_rejects_non_integer_volatility_window() -> None:
    with pytest.raises(ValueError, match="integer"):
        add_l2_state_features(_raw_snapshots().assign(mid=[100.0, 100.01]), levels=1, volatility_window=2.5)


def test_add_l2_state_features_rejects_non_finite_inputs() -> None:
    frame = pd.DataFrame(
        {"mid": [100.0, float("nan")], "bid_sz_1": [1.0, 1.0], "ask_sz_1": [1.0, 1.0]}
    )

    with pytest.raises(ValueError, match="finite"):
        add_l2_state_features(frame, levels=1)
