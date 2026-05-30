import numpy as np
import pytest

from lcri_lab.baseline import (
    LiquidityBaseline,
    baseline_basis_comparison,
    baseline_component_attribution,
    baseline_liquidity_stress_curve,
    compute_lcri,
    design_feature_names,
)
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


def test_baseline_component_attribution_exposes_nonlinear_liquidity_dominance() -> None:
    books = simulate_order_books(SimulationConfig(rows=700, seed=14))
    features = compute_features(books)
    features["raw_imbalance"] = (
        0.09 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        + 0.30 * features["liquidity_void_ratio"].to_numpy(dtype=float)
        * features["volatility"].to_numpy(dtype=float)
        - 0.25 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )
    baseline = LiquidityBaseline(ridge=1e-8).fit(features)

    attribution = baseline_component_attribution(features, baseline)
    component_share = attribution.groupby("component")["contribution_share"].sum()
    top = attribution.iloc[0]

    assert attribution.columns.tolist() == [
        "component",
        "feature",
        "coefficient",
        "mean_contribution",
        "mean_abs_contribution",
        "contribution_share",
    ]
    assert component_share["nonlinear_liquidity"] > 0.90
    assert top["component"] == "nonlinear_liquidity"
    assert top["feature"] in {
        "spread_stress_squared",
        "liquidity_void_x_volatility",
        "replenishment_inverse",
    }
    assert attribution["contribution_share"].sum() == pytest.approx(1.0)


def test_baseline_liquidity_stress_curve_is_centered_and_monotone_for_convex_spread() -> None:
    books = simulate_order_books(SimulationConfig(rows=700, seed=16))
    features = compute_features(books)
    features["raw_imbalance"] = 0.08 * features["spread_ticks"].to_numpy(dtype=float) ** 2
    baseline = LiquidityBaseline(ridge=1e-8).fit(features)

    curve = baseline_liquidity_stress_curve(
        features,
        baseline,
        feature="spread_stress_squared",
        grid_size=7,
    )

    assert curve.columns.tolist() == [
        "feature",
        "quantile",
        "feature_value",
        "expected_imbalance",
        "delta_vs_median",
    ]
    assert curve["feature"].unique().tolist() == ["spread_stress_squared"]
    assert curve["quantile"].tolist() == pytest.approx([0.0, 1 / 6, 2 / 6, 0.5, 4 / 6, 5 / 6, 1.0])
    assert curve.loc[3, "delta_vs_median"] == pytest.approx(0.0, abs=1e-12)
    assert curve["expected_imbalance"].is_monotonic_increasing
    assert curve.loc[0, "delta_vs_median"] < 0.0
    assert curve.loc[6, "delta_vs_median"] > 0.0


def test_baseline_liquidity_stress_curve_rejects_unknown_feature() -> None:
    books = simulate_order_books(SimulationConfig(rows=50, seed=17))
    features = compute_features(books)
    baseline = LiquidityBaseline().fit(features)

    with pytest.raises(ValueError, match="unknown design feature"):
        baseline_liquidity_stress_curve(features, baseline, feature="not_a_feature")


def test_baseline_basis_comparison_quantifies_out_of_sample_nonlinear_lift() -> None:
    books = simulate_order_books(SimulationConfig(rows=900, seed=18))
    features = compute_features(books)
    features["raw_imbalance"] = (
        0.10 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        - 0.32 * features["volatility"].to_numpy(dtype=float) ** 2
        + 0.40 * features["liquidity_void_ratio"].to_numpy(dtype=float)
        * features["volatility"].to_numpy(dtype=float)
        - 0.20 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )

    comparison = baseline_basis_comparison(features, train_fraction=0.55, ridge=1e-8)
    by_basis = comparison.set_index("basis")

    assert comparison.columns.tolist() == [
        "basis",
        "features",
        "train_rows",
        "test_rows",
        "train_rmse",
        "test_rmse",
        "test_rmse_lift_vs_core",
        "test_residual_mean",
        "test_residual_std",
        "overfit_ratio",
    ]
    assert by_basis.index.tolist() == ["core", "interaction", "nonlinear_liquidity"]
    assert by_basis.loc["core", "features"] < by_basis.loc["nonlinear_liquidity", "features"]
    assert by_basis.loc["core", "test_rmse_lift_vs_core"] == pytest.approx(0.0)
    assert by_basis.loc["nonlinear_liquidity", "test_rmse"] < by_basis.loc["core", "test_rmse"] * 0.30
    assert by_basis.loc["nonlinear_liquidity", "test_rmse_lift_vs_core"] > 0.70
    assert by_basis.loc["nonlinear_liquidity", "overfit_ratio"] < 2.0


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
