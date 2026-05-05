from __future__ import annotations


def _validate_levels(levels: int) -> None:
    if levels < 1:
        raise ValueError("levels must be at least 1")


def l2_price_columns(levels: int = 5) -> list[str]:
    _validate_levels(levels)
    return [
        f"{side}_px_{level}"
        for level in range(1, levels + 1)
        for side in ("bid", "ask")
    ]


def l2_size_columns(levels: int = 5) -> list[str]:
    _validate_levels(levels)
    return [
        f"{side}_sz_{level}"
        for level in range(1, levels + 1)
        for side in ("bid", "ask")
    ]


def snapshot_required_columns(levels: int = 5) -> list[str]:
    return [
        "mid",
        "next_mid",
        "spread",
        "spread_ticks",
        "volatility",
        "replenishment_rate",
        *l2_size_columns(levels),
    ]
