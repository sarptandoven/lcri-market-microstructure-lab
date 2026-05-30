from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import pandas as pd

from lcri_lab.features import feature_columns

INTERACTION_FEATURES = [
    "spread_x_replenishment",
    "volatility_x_spread_depth_ratio",
    "log_depth_x_depth_slope",
]

NONLINEAR_LIQUIDITY_FEATURES = [
    "spread_stress_squared",
    "volatility_stress_squared",
    "liquidity_void_x_volatility",
    "replenishment_inverse",
]


@dataclass
class LiquidityBaseline:
    ridge: float = 1e-3
    coefficients: Optional[np.ndarray] = None
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    residual_scale_by_regime: Optional[dict[str, float]] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be a finite non-negative value")

    def fit(self, frame: pd.DataFrame) -> "LiquidityBaseline":
        if frame.empty:
            raise ValueError("cannot fit baseline on an empty frame")
        x = _design_matrix(frame)
        y = frame["raw_imbalance"].to_numpy(dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        xz = (x - self.mean_) / self.scale_
        xz = np.column_stack([np.ones(len(xz)), xz])

        penalty = np.sqrt(self.ridge) * np.eye(xz.shape[1])
        penalty[0, 0] = 0.0
        augmented_x = np.vstack([xz, penalty])
        augmented_y = np.concatenate([y, np.zeros(xz.shape[1])])
        self.coefficients = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]

        residual = y - self.predict(frame)
        scales: dict[str, float] = {}
        for regime, values in pd.Series(residual).groupby(frame["regime"].to_numpy()):
            scale = float(np.std(values.to_numpy(dtype=float)))
            scales[str(regime)] = max(scale, 1e-6)
        self.residual_scale_by_regime = scales
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("baseline must be fit before prediction")
        x = _design_matrix(frame)
        xz = (x - self.mean_) / self.scale_
        xz = np.column_stack([np.ones(len(xz)), xz])
        return np.sum(xz * self.coefficients, axis=1)


def compute_lcri(frame: pd.DataFrame, baseline: LiquidityBaseline) -> pd.DataFrame:
    output = frame.copy()
    expected = baseline.predict(output)
    residual = output["raw_imbalance"].to_numpy(dtype=float) - expected
    output["expected_imbalance"] = expected
    output["imbalance_residual"] = residual

    if baseline.residual_scale_by_regime is None:
        raise RuntimeError("baseline residual scales are unavailable")

    default_scale = float(np.std(residual)) or 1.0
    scales = output["regime"].map(baseline.residual_scale_by_regime).fillna(default_scale)
    output["lcri"] = residual / scales.to_numpy(dtype=float)
    return output


def design_feature_names() -> list[str]:
    return [*feature_columns(), *INTERACTION_FEATURES, *NONLINEAR_LIQUIDITY_FEATURES]


def baseline_basis_comparison(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    ridge: float = 1e-3,
) -> pd.DataFrame:
    """Compare chronological out-of-sample residual fit across baseline bases.

    LCRI neutralization is publishable only if nonlinear liquidity terms improve
    held-out residual control rather than merely increasing in-sample flexibility.
    This diagnostic fits core, interaction-augmented, and full nonlinear liquidity
    ridge bases on the earliest ``train_fraction`` of the frame, then reports test
    RMSE lift versus the core basis on the remaining chronological holdout.
    """
    columns = [
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
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and in (0, 1)")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    if "raw_imbalance" not in frame.columns:
        raise ValueError("missing baseline basis comparison columns: ['raw_imbalance']")

    y = frame["raw_imbalance"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("raw_imbalance values must be finite")
    train_rows = int(len(frame) * train_fraction)
    if train_rows < 1 or train_rows >= len(frame):
        raise ValueError("train_fraction leaves no train or test rows")

    x = _design_matrix(frame)
    feature_names = design_feature_names()
    basis_indexes = {
        "core": list(range(len(feature_columns()))),
        "interaction": list(range(len(feature_columns()) + len(INTERACTION_FEATURES))),
        "nonlinear_liquidity": list(range(len(feature_names))),
    }

    def fit_predict(indexes: list[int]) -> tuple[np.ndarray, np.ndarray]:
        x_basis = x[:, indexes]
        x_train = x_basis[:train_rows]
        x_test = x_basis[train_rows:]
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale == 0.0] = 1.0
        train_design = np.column_stack([np.ones(train_rows), (x_train - mean) / scale])
        test_design = np.column_stack([np.ones(len(x_test)), (x_test - mean) / scale])
        penalty = np.sqrt(ridge) * np.eye(train_design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.lstsq(
            np.vstack([train_design, penalty]),
            np.concatenate([y[:train_rows], np.zeros(train_design.shape[1])]),
            rcond=None,
        )[0]
        return train_design @ coefficients, test_design @ coefficients

    rows: list[dict[str, float | int | str]] = []
    core_test_rmse: float | None = None
    for basis, indexes in basis_indexes.items():
        train_pred, test_pred = fit_predict(indexes)
        train_residual = y[:train_rows] - train_pred
        test_residual = y[train_rows:] - test_pred
        train_rmse = float(np.sqrt(np.mean(train_residual**2)))
        test_rmse = float(np.sqrt(np.mean(test_residual**2)))
        if core_test_rmse is None:
            core_test_rmse = test_rmse
        lift = 0.0 if core_test_rmse <= 0.0 else (core_test_rmse - test_rmse) / core_test_rmse
        if train_rmse > 0.0:
            overfit_ratio = test_rmse / train_rmse
        else:
            overfit_ratio = 1.0 if test_rmse == 0.0 else float("inf")
        rows.append(
            {
                "basis": basis,
                "features": int(len(indexes)),
                "train_rows": int(train_rows),
                "test_rows": int(len(frame) - train_rows),
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "test_rmse_lift_vs_core": float(lift),
                "test_residual_mean": float(test_residual.mean()),
                "test_residual_std": float(test_residual.std()),
                "overfit_ratio": float(overfit_ratio),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def baseline_component_attribution(frame: pd.DataFrame, baseline: LiquidityBaseline) -> pd.DataFrame:
    """Attribute a fitted baseline's prediction to core, interaction, and nonlinear terms.

    LCRI is only as publishable as its neutralization model: if residual imbalance is
    mostly explained by convex stress terms, reviewers need to know the signal is
    not just a linear liquidity proxy in disguise. This diagnostic decomposes the
    standardized ridge prediction into feature-level absolute contribution shares.
    """
    columns = [
        "component",
        "feature",
        "coefficient",
        "mean_contribution",
        "mean_abs_contribution",
        "contribution_share",
    ]
    if baseline.coefficients is None or baseline.mean_ is None or baseline.scale_ is None:
        raise RuntimeError("baseline must be fit before attribution")
    feature_names = design_feature_names()
    expected_shape = (len(feature_names) + 1,)
    if baseline.coefficients.shape != expected_shape:
        raise ValueError("baseline coefficients do not match design feature names")
    if baseline.mean_.shape != (len(feature_names),) or baseline.scale_.shape != (len(feature_names),):
        raise ValueError("baseline normalization arrays do not match design feature names")
    if frame.empty:
        return pd.DataFrame(columns=columns)

    x = _design_matrix(frame)
    xz = (x - baseline.mean_) / baseline.scale_
    contributions = xz * baseline.coefficients[1:]
    mean_contribution = contributions.mean(axis=0)
    mean_abs_contribution = np.abs(contributions).mean(axis=0)
    denominator = float(mean_abs_contribution.sum())
    shares = mean_abs_contribution / denominator if denominator > 0.0 else np.zeros_like(mean_abs_contribution)

    rows = [
        {
            "component": _component_for_feature(feature),
            "feature": feature,
            "coefficient": float(coefficient),
            "mean_contribution": float(mean_value),
            "mean_abs_contribution": float(abs_value),
            "contribution_share": float(share),
        }
        for feature, coefficient, mean_value, abs_value, share in zip(
            feature_names,
            baseline.coefficients[1:],
            mean_contribution,
            mean_abs_contribution,
            shares,
            strict=True,
        )
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["contribution_share", "mean_abs_contribution", "feature"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def baseline_liquidity_stress_curve(
    frame: pd.DataFrame,
    baseline: LiquidityBaseline,
    *,
    feature: str,
    grid_size: int = 11,
) -> pd.DataFrame:
    """Trace the fitted baseline response to one liquidity-stress design feature.

    This partial-dependence style diagnostic keeps every other design column at its
    sample median, then walks the selected feature across empirical quantiles. It
    makes nonlinear neutralization auditable: reviewers can see whether the LCRI
    baseline learned a plausible spread/volatility/void/replenishment response
    instead of hiding it inside one opaque prediction vector.
    """
    columns = ["feature", "quantile", "feature_value", "expected_imbalance", "delta_vs_median"]
    if baseline.coefficients is None or baseline.mean_ is None or baseline.scale_ is None:
        raise RuntimeError("baseline must be fit before stress curve")
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2")

    feature_names = design_feature_names()
    if feature not in feature_names:
        raise ValueError(f"unknown design feature: {feature}")
    expected_shape = (len(feature_names) + 1,)
    if baseline.coefficients.shape != expected_shape:
        raise ValueError("baseline coefficients do not match design feature names")
    if baseline.mean_.shape != (len(feature_names),) or baseline.scale_.shape != (len(feature_names),):
        raise ValueError("baseline normalization arrays do not match design feature names")
    if frame.empty:
        return pd.DataFrame(columns=columns)

    x = _design_matrix(frame)
    if not np.isfinite(x).all():
        raise ValueError("design matrix values must be finite")
    feature_index = feature_names.index(feature)
    quantiles = np.linspace(0.0, 1.0, grid_size)
    feature_values = np.quantile(x[:, feature_index], quantiles)
    median_design = np.median(x, axis=0)
    median_feature_value = float(np.quantile(x[:, feature_index], 0.5))

    def predict_at(value: float) -> float:
        design = median_design.copy()
        design[feature_index] = value
        standardized = (design - baseline.mean_) / baseline.scale_
        row = np.concatenate([[1.0], standardized])
        return float(np.sum(row * baseline.coefficients))

    median_prediction = predict_at(median_feature_value)
    rows = [
        {
            "feature": feature,
            "quantile": float(quantile),
            "feature_value": float(value),
            "expected_imbalance": predict_at(float(value)),
        }
        for quantile, value in zip(quantiles, feature_values, strict=True)
    ]
    for row in rows:
        row["delta_vs_median"] = float(row["expected_imbalance"] - median_prediction)
    return pd.DataFrame(rows, columns=columns)


def _component_for_feature(feature: str) -> str:
    if feature in NONLINEAR_LIQUIDITY_FEATURES:
        return "nonlinear_liquidity"
    if feature in INTERACTION_FEATURES:
        return "interaction"
    return "core"


def _design_matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = feature_columns()
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    x = frame[cols].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("feature columns must be finite")
    interactions = np.column_stack(
        [
            frame["spread_ticks"].to_numpy(dtype=float) * frame["replenishment_rate"].to_numpy(dtype=float),
            frame["volatility"].to_numpy(dtype=float) * frame["spread_depth_ratio"].to_numpy(dtype=float),
            frame["log_total_depth"].to_numpy(dtype=float) * frame["depth_slope"].to_numpy(dtype=float),
        ]
    )
    if not np.isfinite(interactions).all():
        raise ValueError("feature interactions must be finite")
    nonlinear = np.column_stack(
        [
            frame["spread_ticks"].to_numpy(dtype=float) ** 2,
            frame["volatility"].to_numpy(dtype=float) ** 2,
            frame["liquidity_void_ratio"].to_numpy(dtype=float)
            * frame["volatility"].to_numpy(dtype=float),
            1.0 / (1.0 + frame["replenishment_rate"].to_numpy(dtype=float)),
        ]
    )
    if not np.isfinite(nonlinear).all():
        raise ValueError("nonlinear liquidity features must be finite")
    return np.column_stack([x, interactions, nonlinear])
