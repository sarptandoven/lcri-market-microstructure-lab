import numpy as np
import pandas as pd
import pytest

from lcri_lab.baseline import (
    LiquidityBaseline,
    baseline_basis_comparison,
    baseline_component_attribution,
    baseline_liquidity_stress_curve,
    baseline_nonlinear_publishability_summary,
    baseline_regime_basis_comparison,
    baseline_regime_publishability_summary,
    baseline_rolling_basis_comparison,
    baseline_rolling_basis_summary,
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


def test_baseline_rolling_basis_comparison_tracks_stable_nonlinear_lift() -> None:
    books = simulate_order_books(SimulationConfig(rows=960, seed=19))
    features = compute_features(books)
    nonlinear_signal = (
        0.12 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        - 0.28 * features["volatility"].to_numpy(dtype=float) ** 2
        + 0.36 * features["liquidity_void_ratio"].to_numpy(dtype=float)
        * features["volatility"].to_numpy(dtype=float)
        - 0.18 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )
    features["raw_imbalance"] = nonlinear_signal

    rolling = baseline_rolling_basis_comparison(
        features,
        train_window=320,
        test_window=160,
        step=160,
        ridge=1e-8,
    )
    nonlinear = rolling[rolling["basis"] == "nonlinear_liquidity"]
    core = rolling[rolling["basis"] == "core"]

    assert rolling.columns.tolist() == [
        "fold",
        "basis",
        "features",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "train_rows",
        "test_rows",
        "train_rmse",
        "test_rmse",
        "test_rmse_lift_vs_core",
        "test_residual_mean",
        "test_residual_std",
        "overfit_ratio",
        "is_fold_winner",
    ]
    assert rolling["fold"].nunique() >= 3
    assert core["test_rmse_lift_vs_core"].tolist() == pytest.approx([0.0] * len(core))
    assert nonlinear["test_rmse_lift_vs_core"].min() > 0.70
    assert nonlinear["is_fold_winner"].all()
    assert nonlinear["overfit_ratio"].max() < 2.0


def test_baseline_rolling_basis_comparison_rejects_windows_without_holdout() -> None:
    books = simulate_order_books(SimulationConfig(rows=30, seed=20))
    features = compute_features(books)

    with pytest.raises(ValueError, match="at least one rolling fold"):
        baseline_rolling_basis_comparison(features, train_window=20, test_window=20)


def test_baseline_regime_basis_comparison_surfaces_state_specific_nonlinear_lift() -> None:
    books = simulate_order_books(SimulationConfig(rows=980, seed=25))
    features = compute_features(books)
    stress = (
        0.13 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        - 0.26 * features["volatility"].to_numpy(dtype=float) ** 2
        + 0.38
        * features["liquidity_void_ratio"].to_numpy(dtype=float)
        * features["volatility"].to_numpy(dtype=float)
        - 0.19 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )
    regime_multiplier = features["regime"].map(
        {"normal": 1.00, "stressed": 1.25, "thin": 0.85}
    ).fillna(1.10)
    features["raw_imbalance"] = stress * regime_multiplier.to_numpy(dtype=float)

    comparison = baseline_regime_basis_comparison(
        features,
        train_window=320,
        test_window=160,
        step=160,
        ridge=1e-8,
    )
    by_key = comparison.set_index(["regime", "basis"])

    assert comparison.columns.tolist() == [
        "regime",
        "basis",
        "folds",
        "test_rows",
        "mean_test_rmse",
        "mean_test_rmse_lift_vs_core",
        "positive_lift_rate",
        "winner_rate",
        "worst_fold_lift",
        "publishability_note",
    ]
    assert set(comparison["basis"]) == {"core", "interaction", "nonlinear_liquidity"}
    assert by_key.xs("core", level="basis")["mean_test_rmse_lift_vs_core"].tolist() == pytest.approx(
        [0.0] * comparison["regime"].nunique()
    )
    nonlinear = comparison[comparison["basis"] == "nonlinear_liquidity"]
    assert nonlinear["mean_test_rmse_lift_vs_core"].min() > 0.55
    assert nonlinear["positive_lift_rate"].min() == pytest.approx(1.0)
    assert nonlinear["winner_rate"].min() == pytest.approx(1.0)
    assert set(nonlinear["publishability_note"]) == {"regime_persistent_nonlinear_lift"}


def test_baseline_regime_basis_comparison_rejects_missing_regime_column() -> None:
    books = simulate_order_books(SimulationConfig(rows=100, seed=26))
    features = compute_features(books).drop(columns=["regime"])

    with pytest.raises(ValueError, match="missing regime basis comparison columns"):
        baseline_regime_basis_comparison(features, train_window=40, test_window=20)


def test_baseline_regime_publishability_summary_requires_every_regime_supported() -> None:
    comparison = pd.DataFrame(
        {
            "regime": ["normal", "stressed", "thin", "normal", "stressed", "thin"],
            "basis": [
                "core",
                "core",
                "core",
                "nonlinear_liquidity",
                "nonlinear_liquidity",
                "nonlinear_liquidity",
            ],
            "folds": [3, 3, 3, 3, 3, 3],
            "test_rows": [180, 120, 90, 180, 120, 90],
            "mean_test_rmse_lift_vs_core": [0.0, 0.0, 0.0, 0.62, 0.58, 0.54],
            "positive_lift_rate": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "winner_rate": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "worst_fold_lift": [0.0, 0.0, 0.0, 0.51, 0.49, 0.47],
            "publishability_note": [
                "core_reference",
                "core_reference",
                "core_reference",
                "regime_persistent_nonlinear_lift",
                "regime_persistent_nonlinear_lift",
                "regime_persistent_nonlinear_lift",
            ],
        }
    )

    summary = baseline_regime_publishability_summary(comparison, min_regime_lift=0.50)

    assert summary == {
        "regimes": 3,
        "supported_regimes": 3,
        "unsupported_regimes": 0,
        "min_regime_mean_lift": pytest.approx(0.54),
        "min_regime_worst_fold_lift": pytest.approx(0.47),
        "min_regime_winner_rate": pytest.approx(1.0),
        "weakest_regime": "thin",
        "publishable": True,
        "review_note": "nonlinear_lift_regime_robust",
    }


def test_baseline_regime_publishability_summary_flags_weak_regime() -> None:
    comparison = pd.DataFrame(
        {
            "regime": ["normal", "stressed"],
            "basis": ["nonlinear_liquidity", "nonlinear_liquidity"],
            "mean_test_rmse_lift_vs_core": [0.55, 0.12],
            "positive_lift_rate": [1.0, 0.50],
            "winner_rate": [1.0, 0.50],
            "worst_fold_lift": [0.45, -0.05],
        }
    )

    summary = baseline_regime_publishability_summary(comparison, min_regime_lift=0.30)

    assert summary["publishable"] is False
    assert summary["supported_regimes"] == 1
    assert summary["unsupported_regimes"] == 1
    assert summary["weakest_regime"] == "stressed"
    assert summary["review_note"] == "nonlinear_lift_regime_fragile"


def test_baseline_rolling_basis_summary_scores_persistent_nonlinear_lift() -> None:
    books = simulate_order_books(SimulationConfig(rows=960, seed=21))
    features = compute_features(books)
    features["raw_imbalance"] = (
        0.15 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        - 0.24 * features["volatility"].to_numpy(dtype=float) ** 2
        + 0.42
        * features["liquidity_void_ratio"].to_numpy(dtype=float)
        * features["volatility"].to_numpy(dtype=float)
        - 0.16 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )

    rolling = baseline_rolling_basis_comparison(
        features,
        train_window=320,
        test_window=160,
        step=160,
        ridge=1e-8,
    )
    summary = baseline_rolling_basis_summary(rolling, lift_floor=0.50, max_overfit_ratio=2.0)
    nonlinear = summary.set_index("basis").loc["nonlinear_liquidity"]

    assert summary.columns.tolist() == [
        "basis",
        "folds",
        "winner_rate",
        "positive_lift_rate",
        "median_test_rmse_lift_vs_core",
        "min_test_rmse_lift_vs_core",
        "max_overfit_ratio",
        "median_test_residual_abs_mean",
        "stable_lift",
        "publishability_note",
    ]
    assert nonlinear["folds"] == rolling["fold"].nunique()
    assert nonlinear["winner_rate"] == pytest.approx(1.0)
    assert nonlinear["positive_lift_rate"] == pytest.approx(1.0)
    assert nonlinear["stable_lift"] is True
    assert nonlinear["publishability_note"] == "persistent_out_of_sample_lift"


def test_baseline_nonlinear_publishability_summary_requires_lift_and_attribution() -> None:
    books = simulate_order_books(SimulationConfig(rows=960, seed=24))
    features = compute_features(books)
    features["raw_imbalance"] = (
        0.14 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        - 0.22 * features["volatility"].to_numpy(dtype=float) ** 2
        + 0.44
        * features["liquidity_void_ratio"].to_numpy(dtype=float)
        * features["volatility"].to_numpy(dtype=float)
        - 0.20 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )
    baseline = LiquidityBaseline(ridge=1e-8).fit(features)
    attribution = baseline_component_attribution(features, baseline)
    rolling = baseline_rolling_basis_comparison(
        features,
        train_window=320,
        test_window=160,
        step=160,
        ridge=1e-8,
    )
    rolling_summary = baseline_rolling_basis_summary(rolling, lift_floor=0.50, max_overfit_ratio=2.0)

    summary = baseline_nonlinear_publishability_summary(
        attribution,
        rolling_summary,
        min_contribution_share=0.75,
        min_median_lift=0.50,
    )

    assert summary == {
        "nonlinear_contribution_share": pytest.approx(
            attribution.groupby("component")["contribution_share"].sum().loc["nonlinear_liquidity"]
        ),
        "nonlinear_winner_rate": pytest.approx(1.0),
        "nonlinear_positive_lift_rate": pytest.approx(1.0),
        "nonlinear_median_test_rmse_lift_vs_core": pytest.approx(
            rolling_summary.set_index("basis").loc[
                "nonlinear_liquidity", "median_test_rmse_lift_vs_core"
            ]
        ),
        "nonlinear_min_test_rmse_lift_vs_core": pytest.approx(
            rolling_summary.set_index("basis").loc[
                "nonlinear_liquidity", "min_test_rmse_lift_vs_core"
            ]
        ),
        "nonlinear_stable_lift": True,
        "publishable": True,
        "review_note": "nonlinear_baseline_supported",
    }


def test_baseline_nonlinear_publishability_summary_flags_unsupported_baseline() -> None:
    attribution = pd.DataFrame(
        {
            "component": ["core", "nonlinear_liquidity"],
            "feature": ["spread_ticks", "spread_stress_squared"],
            "contribution_share": [0.80, 0.20],
        }
    )
    rolling_summary = pd.DataFrame(
        {
            "basis": ["core", "nonlinear_liquidity"],
            "winner_rate": [0.60, 0.40],
            "positive_lift_rate": [0.0, 0.50],
            "median_test_rmse_lift_vs_core": [0.0, 0.08],
            "min_test_rmse_lift_vs_core": [0.0, -0.10],
            "stable_lift": [False, False],
        }
    )

    summary = baseline_nonlinear_publishability_summary(attribution, rolling_summary)

    assert summary["publishable"] is False
    assert summary["review_note"] == "nonlinear_baseline_under_supported"


def test_baseline_rolling_basis_summary_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing rolling basis summary columns"):
        baseline_rolling_basis_summary(pd.DataFrame({"basis": ["core"]}))


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
