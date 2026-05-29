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
