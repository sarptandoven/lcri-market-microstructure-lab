import pytest

from lcri_lab.schema import snapshot_required_columns


def test_snapshot_required_columns_tracks_levels() -> None:
    columns = snapshot_required_columns(levels=2)

    assert columns[:6] == ["mid", "next_mid", "spread", "spread_ticks", "volatility", "replenishment_rate"]
    assert columns[-4:] == ["bid_sz_1", "ask_sz_1", "bid_sz_2", "ask_sz_2"]


def test_snapshot_required_columns_rejects_invalid_levels() -> None:
    with pytest.raises(ValueError, match="levels"):
        snapshot_required_columns(levels=0)
