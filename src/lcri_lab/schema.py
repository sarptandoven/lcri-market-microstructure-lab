from __future__ import annotations


def snapshot_required_columns(levels: int = 5) -> list[str]:
    if levels < 1:
        raise ValueError("levels must be at least 1")
    size_columns = [
        f"{side}_sz_{level}"
        for level in range(1, levels + 1)
        for side in ("bid", "ask")
    ]
    return [
        "mid",
        "next_mid",
        "spread",
        "spread_ticks",
        "volatility",
        "replenishment_rate",
        *size_columns,
    ]
