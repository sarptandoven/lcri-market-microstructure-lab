import numpy as np
import pytest

from lcri_lab.baseline import LiquidityBaseline, compute_lcri, design_feature_names
from lcri_lab.features import compute_features
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def test_baseline_predicts_and_computes_lcri() -> None:
    books = simulate_order_books(SimulationConfig(rows=500, seed=2))
    features = compute_features(books)
    baseline = LiquidityBaseline().fit(features)
    scored = compute_lcri(features, baseline)

    assert "expected_imbalance" in scored.columns
    assert "lcri" in scored.columns
    assert np.isfinite(scored["lcri"]).all()
    assert set(baseline.residual_scale_by_regime) == set(scored["regime"].unique())


def test_design_feature_names_include_interactions() -> None:
    names = design_feature_names()

    assert "spread_ticks" in names
    assert "imbalance_fracture" in names
    assert "liquidity_void_ratio" in names
    assert "resilience_asymmetry" in names
    assert "spread_x_replenishment" in names
    assert "log_depth_x_depth_slope" in names


def test_baseline_rejects_empty_fit_frame() -> None:
    books = simulate_order_books(SimulationConfig(rows=10, seed=8))
    features = compute_features(books).iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        LiquidityBaseline().fit(features)
