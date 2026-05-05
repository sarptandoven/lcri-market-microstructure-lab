import pytest

from lcri_lab.schema import l2_price_columns, l2_side_size_columns, l2_size_columns, snapshot_required_columns


def test_snapshot_required_columns_tracks_levels() -> None:
    columns = snapshot_required_columns(levels=2)

    assert columns[:6] == ["mid", "next_mid", "spread", "spread_ticks", "volatility", "replenishment_rate"]
    assert columns[-4:] == l2_size_columns(levels=2)


def test_l2_price_columns_track_levels() -> None:
    assert l2_price_columns(levels=2) == ["bid_px_1", "ask_px_1", "bid_px_2", "ask_px_2"]


def test_l2_side_size_columns_track_one_side() -> None:
    assert l2_side_size_columns("ask", levels=2) == ["ask_sz_1", "ask_sz_2"]


def test_snapshot_required_columns_rejects_invalid_levels() -> None:
    with pytest.raises(ValueError, match="levels"):
        snapshot_required_columns(levels=0)
