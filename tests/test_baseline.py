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
    assert "spread_stress_squared" in names
    assert "volatility_stress_squared" in names
    assert "liquidity_void_x_volatility" in names
    assert "replenishment_inverse" in names


def test_nonlinear_liquidity_basis_can_absorb_convex_stress_response() -> None:
    books = simulate_order_books(SimulationConfig(rows=700, seed=12))
    features = compute_features(books)
    features["raw_imbalance"] = (
        0.06 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        - 0.35 * features["volatility"].to_numpy(dtype=float) ** 2
        + 0.25 * features["liquidity_void_ratio"].to_numpy(dtype=float)
        * features["volatility"].to_numpy(dtype=float)
        - 0.18 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )

    baseline = LiquidityBaseline(ridge=1e-8).fit(features)
    residual = features["raw_imbalance"].to_numpy(dtype=float) - baseline.predict(features)

    assert len(baseline.coefficients) == len(design_feature_names()) + 1
    assert float(np.sqrt(np.mean(residual**2))) < 1e-6


def test_baseline_rejects_invalid_ridge() -> None:
    with pytest.raises(ValueError, match="ridge"):
        LiquidityBaseline(ridge=float("nan"))


def test_baseline_rejects_empty_fit_frame() -> None:
    books = simulate_order_books(SimulationConfig(rows=10, seed=8))
    features = compute_features(books).iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        LiquidityBaseline().fit(features)


def test_baseline_rejects_non_finite_features() -> None:
    books = simulate_order_books(SimulationConfig(rows=10, seed=9))
    features = compute_features(books)
    features.loc[0, "liquidity_score"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        LiquidityBaseline().fit(features)
