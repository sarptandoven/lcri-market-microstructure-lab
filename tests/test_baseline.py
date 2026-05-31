import numpy as np
import pandas as pd
import pytest

from lcri_lab.baseline import (
    LiquidityBaseline,
    baseline_basis_comparison,
    baseline_component_attribution,
    baseline_liquidity_stress_curve,
    baseline_nonlinear_coefficient_stability,
    baseline_nonlinear_coefficient_stability_summary,
    baseline_nonlinear_publishability_summary,
    baseline_regime_basis_comparison,
    baseline_regime_publishability_summary,
    baseline_regime_tail_lift_diagnostics,
    baseline_regime_tail_lift_summary,
    baseline_rolling_basis_comparison,
    baseline_rolling_basis_summary,
    baseline_stress_residual_drift,
    baseline_stress_tail_publishability_summary,
    baseline_tail_lift_diagnostics,
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


def test_baseline_nonlinear_coefficient_stability_flags_sign_stable_stress_terms() -> None:
    books = simulate_order_books(SimulationConfig(rows=960, seed=33))
    features = compute_features(books)
    features["raw_imbalance"] = (
        0.11 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        - 0.31 * features["volatility"].to_numpy(dtype=float) ** 2
        + 0.37 * features["liquidity_void_ratio"].to_numpy(dtype=float)
        * features["volatility"].to_numpy(dtype=float)
        - 0.16 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )

    stability = baseline_nonlinear_coefficient_stability(
        features,
        train_window=320,
        step=160,
        ridge=1e-8,
    )
    by_feature = stability.set_index("feature")

    assert stability.columns.tolist() == [
        "feature",
        "component",
        "windows",
        "mean_coefficient",
        "std_coefficient",
        "mean_abs_coefficient",
        "coefficient_cv",
        "sign_consistency",
        "stability_label",
    ]
    assert by_feature.loc["spread_stress_squared", "component"] == "nonlinear_liquidity"
    assert by_feature.loc["spread_stress_squared", "sign_consistency"] == pytest.approx(1.0)
    assert by_feature.loc["volatility_stress_squared", "sign_consistency"] == pytest.approx(1.0)
    assert by_feature.loc["liquidity_void_x_volatility", "sign_consistency"] == pytest.approx(1.0)
    assert by_feature.loc["spread_stress_squared", "stability_label"] in {
        "sign_stable_dominant",
        "sign_stable",
    }
    assert stability["mean_abs_coefficient"].is_monotonic_decreasing


def test_baseline_nonlinear_coefficient_stability_rejects_invalid_windows() -> None:
    books = simulate_order_books(SimulationConfig(rows=100, seed=34))
    features = compute_features(books)

    with pytest.raises(ValueError, match="train_window must be a positive integer"):
        baseline_nonlinear_coefficient_stability(features, train_window=0)
    with pytest.raises(ValueError, match="at least one coefficient stability window"):
        baseline_nonlinear_coefficient_stability(features, train_window=200)


def test_baseline_nonlinear_coefficient_stability_summary_gates_fragile_terms() -> None:
    stability = pd.DataFrame(
        {
            "feature": [
                "spread_stress_squared",
                "volatility_stress_squared",
                "liquidity_void_x_volatility",
                "replenishment_inverse",
            ],
            "component": ["nonlinear_liquidity"] * 4,
            "windows": [5, 5, 5, 5],
            "mean_coefficient": [0.40, -0.30, 0.10, 0.02],
            "std_coefficient": [0.05, 0.04, 0.20, 0.01],
            "mean_abs_coefficient": [0.40, 0.30, 0.10, 0.02],
            "coefficient_cv": [0.125, 0.133, 2.0, 0.50],
            "sign_consistency": [1.0, 0.90, 0.60, 0.75],
            "stability_label": [
                "sign_stable_dominant",
                "sign_stable",
                "sign_unstable",
                "inactive",
            ],
        }
    )

    summary = baseline_nonlinear_coefficient_stability_summary(
        stability,
        min_sign_consistency=0.80,
        max_coefficient_cv=1.0,
        min_stable_share=0.75,
    )

    assert summary == {
        "features": 4,
        "stable_features": 2,
        "unstable_features": 2,
        "stable_feature_share": pytest.approx(0.50),
        "min_sign_consistency": pytest.approx(0.60),
        "max_coefficient_cv": pytest.approx(2.0),
        "weakest_feature": "liquidity_void_x_volatility",
        "stability_label": "nonlinear_coefficient_instability",
        "publishable": False,
        "review_note": "nonlinear_coefficients_fragile",
    }


def test_baseline_nonlinear_coefficient_stability_summary_accepts_robust_terms() -> None:
    stability = pd.DataFrame(
        {
            "feature": ["spread_stress_squared", "volatility_stress_squared"],
            "component": ["nonlinear_liquidity", "nonlinear_liquidity"],
            "windows": [4, 4],
            "mean_coefficient": [0.40, -0.30],
            "std_coefficient": [0.04, 0.05],
            "mean_abs_coefficient": [0.40, 0.30],
            "coefficient_cv": [0.10, 0.17],
            "sign_consistency": [1.0, 0.95],
            "stability_label": ["sign_stable_dominant", "sign_stable_dominant"],
        }
    )

    summary = baseline_nonlinear_coefficient_stability_summary(stability)

    assert summary["stable_features"] == 2
    assert summary["stability_label"] == "nonlinear_coefficients_stable"
    assert summary["publishable"] is True
    assert summary["review_note"] == "nonlinear_coefficients_reliable"


def test_baseline_nonlinear_coefficient_stability_summary_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="missing nonlinear coefficient stability columns"):
        baseline_nonlinear_coefficient_stability_summary(pd.DataFrame({"feature": ["x"]}))


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


def test_baseline_tail_lift_diagnostics_audits_stressed_holdout_tails() -> None:
    books = simulate_order_books(SimulationConfig(rows=900, seed=27))
    features = compute_features(books)
    stress = features["liquidity_void_ratio"].to_numpy(dtype=float) * features[
        "volatility"
    ].to_numpy(dtype=float)
    features["raw_imbalance"] = (
        0.08 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        + 0.55 * stress
        - 0.18 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )

    diagnostics = baseline_tail_lift_diagnostics(
        features,
        feature="liquidity_void_x_volatility",
        train_fraction=0.55,
        ridge=1e-8,
    )
    by_bucket = diagnostics.set_index("tail_bucket")

    assert diagnostics.columns.tolist() == [
        "tail_bucket",
        "feature",
        "test_rows",
        "feature_min",
        "feature_max",
        "core_rmse",
        "nonlinear_rmse",
        "nonlinear_rmse_lift_vs_core",
        "core_residual_mean",
        "nonlinear_residual_mean",
        "tail_publishability_note",
    ]
    assert diagnostics["tail_bucket"].tolist() == ["low_tail", "body", "high_tail"]
    assert by_bucket.loc["high_tail", "nonlinear_rmse_lift_vs_core"] > 0.70
    assert by_bucket.loc["high_tail", "tail_publishability_note"] == "nonlinear_tail_lift_supported"
    assert by_bucket.loc["body", "test_rows"] > by_bucket.loc["high_tail", "test_rows"]


def test_baseline_tail_lift_diagnostics_rejects_unknown_feature() -> None:
    books = simulate_order_books(SimulationConfig(rows=80, seed=28))
    features = compute_features(books)

    with pytest.raises(ValueError, match="unknown design feature"):
        baseline_tail_lift_diagnostics(features, feature="not_a_feature")


def test_baseline_regime_tail_lift_diagnostics_exposes_fragile_stress_pockets() -> None:
    books = simulate_order_books(SimulationConfig(rows=900, seed=41))
    features = compute_features(books)
    spread = features["spread_ticks"].to_numpy(dtype=float)
    void_vol = features["liquidity_void_ratio"].to_numpy(dtype=float) * features[
        "volatility"
    ].to_numpy(dtype=float)
    features["raw_imbalance"] = 0.07 * spread**2 + 0.42 * void_vol

    diagnostics = baseline_regime_tail_lift_diagnostics(
        features,
        feature="liquidity_void_x_volatility",
        train_fraction=0.55,
        tail_quantile=0.25,
        ridge=1e-8,
        min_regime_tail_lift=0.20,
    )
    high_tail = diagnostics[diagnostics["tail_bucket"] == "high_tail"]

    assert diagnostics.columns.tolist() == [
        "regime",
        "tail_bucket",
        "feature",
        "test_rows",
        "feature_min",
        "feature_max",
        "core_rmse",
        "nonlinear_rmse",
        "nonlinear_rmse_lift_vs_core",
        "core_residual_mean",
        "nonlinear_residual_mean",
        "review_note",
    ]
    assert set(diagnostics["tail_bucket"]) == {"low_tail", "body", "high_tail"}
    assert set(diagnostics["regime"]) == set(features.iloc[int(len(features) * 0.55) :]["regime"])
    assert (high_tail["nonlinear_rmse_lift_vs_core"] > 0.20).all()
    assert (high_tail["review_note"] == "regime_tail_lift_supported").all()


def test_baseline_regime_tail_lift_summary_gates_weakest_tail_by_regime() -> None:
    diagnostics = pd.DataFrame(
        {
            "regime": ["stable", "stable", "thin", "thin"],
            "tail_bucket": ["low_tail", "high_tail", "low_tail", "high_tail"],
            "feature": ["spread_stress_squared"] * 4,
            "test_rows": [25, 26, 27, 28],
            "nonlinear_rmse_lift_vs_core": [0.22, 0.31, -0.03, 0.18],
            "core_residual_mean": [0.04, -0.03, 0.02, -0.01],
            "nonlinear_residual_mean": [0.01, -0.01, 0.01, 0.00],
        }
    )

    summary = baseline_regime_tail_lift_summary(diagnostics, min_regime_tail_lift=0.15)
    by_regime = summary.set_index("regime")

    assert summary.columns.tolist() == [
        "feature",
        "regime",
        "tail_buckets",
        "test_rows",
        "min_regime_tail_lift",
        "median_regime_tail_lift",
        "worst_tail_bucket",
        "supported_tail_buckets",
        "unsupported_tail_buckets",
        "max_core_residual_abs_mean",
        "max_nonlinear_residual_abs_mean",
        "publishable",
        "review_note",
    ]
    assert by_regime.loc["stable", "publishable"] is True
    assert by_regime.loc["stable", "review_note"] == "regime_stress_tail_supported"
    assert by_regime.loc["thin", "publishable"] is False
    assert by_regime.loc["thin", "worst_tail_bucket"] == "low_tail"
    assert by_regime.loc["thin", "review_note"] == "regime_stress_tail_fragile"


def test_baseline_stress_tail_publishability_summary_gates_all_stress_tails() -> None:
    diagnostics = pd.DataFrame(
        {
            "feature": [
                "spread_stress_squared",
                "spread_stress_squared",
                "liquidity_void_x_volatility",
                "liquidity_void_x_volatility",
            ],
            "tail_bucket": ["low_tail", "high_tail", "low_tail", "high_tail"],
            "test_rows": [30, 31, 32, 33],
            "nonlinear_rmse_lift_vs_core": [0.22, 0.31, 0.19, 0.44],
            "core_residual_mean": [0.04, -0.03, 0.02, -0.01],
            "nonlinear_residual_mean": [0.01, -0.01, 0.00, 0.00],
        }
    )

    summary = baseline_stress_tail_publishability_summary(diagnostics, min_tail_lift=0.15)
    by_feature = summary.set_index("feature")

    assert summary.columns.tolist() == [
        "feature",
        "tail_buckets",
        "test_rows",
        "min_tail_lift",
        "median_tail_lift",
        "worst_tail_bucket",
        "worst_tail_lift",
        "supported_tail_buckets",
        "unsupported_tail_buckets",
        "max_core_residual_abs_mean",
        "max_nonlinear_residual_abs_mean",
        "publishable",
        "review_note",
    ]
    assert by_feature.loc["spread_stress_squared", "publishable"] is True
    assert by_feature.loc["spread_stress_squared", "worst_tail_bucket"] == "low_tail"
    assert by_feature.loc["liquidity_void_x_volatility", "publishable"] is True
    assert by_feature.loc["liquidity_void_x_volatility", "min_tail_lift"] == pytest.approx(0.19)
    assert summary["unsupported_tail_buckets"].sum() == 0


def test_baseline_stress_tail_publishability_summary_flags_unsupported_tail() -> None:
    diagnostics = pd.DataFrame(
        {
            "feature": ["volatility_stress_squared", "volatility_stress_squared"],
            "tail_bucket": ["low_tail", "high_tail"],
            "test_rows": [20, 21],
            "nonlinear_rmse_lift_vs_core": [0.12, -0.04],
            "core_residual_mean": [0.03, -0.05],
            "nonlinear_residual_mean": [0.01, -0.06],
        }
    )

    summary = baseline_stress_tail_publishability_summary(diagnostics, min_tail_lift=0.10)
    row = summary.iloc[0]

    assert row["publishable"] is False
    assert row["unsupported_tail_buckets"] == 1
    assert row["worst_tail_bucket"] == "high_tail"
    assert row["review_note"] == "nonlinear_stress_tail_fragile"


def test_baseline_stress_residual_drift_shows_nonlinear_neutralization_by_bucket() -> None:
    books = simulate_order_books(SimulationConfig(rows=900, seed=44))
    features = compute_features(books)
    stress = features["liquidity_void_ratio"].to_numpy(dtype=float) * features[
        "volatility"
    ].to_numpy(dtype=float)
    features["raw_imbalance"] = (
        0.10 * features["spread_ticks"].to_numpy(dtype=float) ** 2
        + 0.75 * stress
        - 0.22 / (1.0 + features["replenishment_rate"].to_numpy(dtype=float))
    )

    drift = baseline_stress_residual_drift(
        features,
        feature="liquidity_void_x_volatility",
        buckets=4,
        train_fraction=0.60,
        ridge=1e-8,
    )

    assert drift.columns.tolist() == [
        "stress_bucket",
        "feature",
        "test_rows",
        "feature_min",
        "feature_max",
        "core_residual_mean",
        "nonlinear_residual_mean",
        "residual_mean_abs_reduction",
        "core_residual_drift_vs_low_bucket",
        "nonlinear_residual_drift_vs_low_bucket",
        "drift_publishability_note",
    ]
    assert drift["stress_bucket"].tolist() == ["q1", "q2", "q3", "q4"]
    assert drift["test_rows"].sum() == len(features) - int(len(features) * 0.60)
    assert drift["residual_mean_abs_reduction"].min() > 0.0
    assert drift["core_residual_drift_vs_low_bucket"].abs().max() > 0.02
    assert drift["nonlinear_residual_drift_vs_low_bucket"].abs().max() < 1e-6
    assert drift["drift_publishability_note"].unique().tolist() == [
        "nonlinear_residual_drift_neutralized"
    ]


def test_baseline_stress_residual_drift_rejects_unknown_stress_feature() -> None:
    books = simulate_order_books(SimulationConfig(rows=80, seed=45))
    features = compute_features(books)

    with pytest.raises(ValueError, match="unknown design feature"):
        baseline_stress_residual_drift(features, feature="missing_stress")


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


def test_baseline_nonlinear_publishability_summary_requires_coefficient_stability() -> None:
    attribution = pd.DataFrame(
        {
            "component": ["core", "nonlinear_liquidity", "nonlinear_liquidity"],
            "feature": [
                "spread_ticks",
                "spread_stress_squared",
                "liquidity_void_x_volatility",
            ],
            "contribution_share": [0.20, 0.45, 0.35],
        }
    )
    rolling_summary = pd.DataFrame(
        {
            "basis": ["core", "nonlinear_liquidity"],
            "winner_rate": [0.0, 1.0],
            "positive_lift_rate": [0.0, 1.0],
            "median_test_rmse_lift_vs_core": [0.0, 0.55],
            "min_test_rmse_lift_vs_core": [0.0, 0.52],
            "stable_lift": [False, True],
        }
    )
    coefficient_stability = pd.DataFrame(
        {
            "feature": ["spread_stress_squared", "liquidity_void_x_volatility"],
            "component": ["nonlinear_liquidity", "nonlinear_liquidity"],
            "windows": [4, 4],
            "mean_abs_coefficient": [0.25, 0.35],
            "sign_consistency": [1.0, 0.50],
            "stability_label": ["sign_stable", "sign_unstable"],
        }
    )

    summary = baseline_nonlinear_publishability_summary(
        attribution,
        rolling_summary,
        coefficient_stability=coefficient_stability,
        min_coefficient_sign_consistency=0.80,
    )

    assert summary["nonlinear_min_coefficient_sign_consistency"] == pytest.approx(0.50)
    assert summary["nonlinear_stable_coefficient_rate"] == pytest.approx(0.50)
    assert summary["nonlinear_coefficients_stable"] is False
    assert summary["publishable"] is False
    assert summary["review_note"] == "nonlinear_baseline_coefficient_instability"


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
