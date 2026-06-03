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


def baseline_rolling_basis_comparison(
    frame: pd.DataFrame,
    *,
    train_window: int = 500,
    test_window: int = 250,
    step: int | None = None,
    ridge: float = 1e-3,
) -> pd.DataFrame:
    """Compare baseline bases across rolling chronological train/test blocks.

    A single holdout can flatter nonlinear liquidity terms if one regime dominates
    the split. This diagnostic repeatedly trains core, interaction, and nonlinear
    bases on a rolling window and scores the immediately following holdout, making
    nonlinear LCRI neutralization auditable as a stable out-of-sample improvement.
    """
    columns = [
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
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for name, value in {"train_window": train_window, "test_window": test_window}.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if step is None:
        step = test_window
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    if "raw_imbalance" not in frame.columns:
        raise ValueError("missing rolling baseline basis comparison columns: ['raw_imbalance']")

    y = frame["raw_imbalance"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("raw_imbalance values must be finite")
    x = _design_matrix(frame)
    feature_names = design_feature_names()
    basis_indexes = {
        "core": list(range(len(feature_columns()))),
        "interaction": list(range(len(feature_columns()) + len(INTERACTION_FEATURES))),
        "nonlinear_liquidity": list(range(len(feature_names))),
    }

    rows: list[dict[str, bool | float | int | str]] = []
    fold = 0
    for train_start in range(0, len(frame) - train_window - test_window + 1, step):
        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + test_window
        y_train = y[train_start:train_end]
        y_test = y[test_start:test_end]
        fold_rows: list[dict[str, bool | float | int | str]] = []
        core_test_rmse: float | None = None
        for basis, indexes in basis_indexes.items():
            x_basis = x[:, indexes]
            x_train = x_basis[train_start:train_end]
            x_test = x_basis[test_start:test_end]
            mean = x_train.mean(axis=0)
            scale = x_train.std(axis=0)
            scale[scale == 0.0] = 1.0
            train_design = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
            test_design = np.column_stack([np.ones(len(x_test)), (x_test - mean) / scale])
            penalty = np.sqrt(ridge) * np.eye(train_design.shape[1])
            penalty[0, 0] = 0.0
            coefficients = np.linalg.lstsq(
                np.vstack([train_design, penalty]),
                np.concatenate([y_train, np.zeros(train_design.shape[1])]),
                rcond=None,
            )[0]
            train_residual = y_train - train_design @ coefficients
            test_residual = y_test - test_design @ coefficients
            train_rmse = float(np.sqrt(np.mean(train_residual**2)))
            test_rmse = float(np.sqrt(np.mean(test_residual**2)))
            if core_test_rmse is None:
                core_test_rmse = test_rmse
            lift = 0.0 if core_test_rmse <= 0.0 else (core_test_rmse - test_rmse) / core_test_rmse
            if train_rmse > 0.0:
                overfit_ratio = test_rmse / train_rmse
            else:
                overfit_ratio = 1.0 if test_rmse == 0.0 else float("inf")
            fold_rows.append(
                {
                    "fold": int(fold),
                    "basis": basis,
                    "features": int(len(indexes)),
                    "train_start": int(train_start),
                    "train_end": int(train_end),
                    "test_start": int(test_start),
                    "test_end": int(test_end),
                    "train_rows": int(train_window),
                    "test_rows": int(test_window),
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "test_rmse_lift_vs_core": float(lift),
                    "test_residual_mean": float(test_residual.mean()),
                    "test_residual_std": float(test_residual.std()),
                    "overfit_ratio": float(overfit_ratio),
                    "is_fold_winner": False,
                }
            )
        best_rmse = min(float(row["test_rmse"]) for row in fold_rows)
        for row in fold_rows:
            row["is_fold_winner"] = bool(float(row["test_rmse"]) == best_rmse)
        rows.extend(fold_rows)
        fold += 1

    if not rows:
        raise ValueError("at least one rolling fold is required; reduce train_window/test_window")
    return pd.DataFrame(rows, columns=columns)


def baseline_regime_basis_comparison(
    frame: pd.DataFrame,
    *,
    regime_col: str = "regime",
    train_window: int = 500,
    test_window: int = 250,
    step: int | None = None,
    ridge: float = 1e-3,
    lift_floor: float = 0.0,
) -> pd.DataFrame:
    """Score nonlinear baseline lift inside each out-of-sample liquidity regime.

    Aggregate rolling RMSE can hide whether nonlinear neutralization only works in
    the easy/high-liquidity state. This diagnostic repeats the rolling chronological
    baseline comparison but reduces test residuals by regime, exposing where the
    nonlinear basis consistently wins and where LCRI residualization remains fragile.
    """
    columns = [
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
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(regime_col, str) or not regime_col:
        raise ValueError("regime_col must be a non-empty string")
    for name, value in {"train_window": train_window, "test_window": test_window}.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if step is None:
        step = test_window
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    if not math.isfinite(lift_floor):
        raise ValueError("lift_floor must be finite")
    missing_required = sorted({"raw_imbalance", regime_col} - set(frame.columns))
    if missing_required:
        raise ValueError(f"missing regime basis comparison columns: {missing_required}")

    y = frame["raw_imbalance"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("raw_imbalance values must be finite")
    x = _design_matrix(frame)
    feature_names = design_feature_names()
    basis_indexes = {
        "core": list(range(len(feature_columns()))),
        "interaction": list(range(len(feature_columns()) + len(INTERACTION_FEATURES))),
        "nonlinear_liquidity": list(range(len(feature_names))),
    }

    fold_rows: list[dict[str, float | int | str]] = []
    fold = 0
    for train_start in range(0, len(frame) - train_window - test_window + 1, step):
        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + test_window
        y_train = y[train_start:train_end]
        y_test = y[test_start:test_end]
        regimes = frame[regime_col].iloc[test_start:test_end].astype(str).to_numpy()
        residuals_by_basis: dict[str, np.ndarray] = {}
        for basis, indexes in basis_indexes.items():
            x_basis = x[:, indexes]
            x_train = x_basis[train_start:train_end]
            x_test = x_basis[test_start:test_end]
            mean = x_train.mean(axis=0)
            scale = x_train.std(axis=0)
            scale[scale == 0.0] = 1.0
            train_design = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
            test_design = np.column_stack([np.ones(len(x_test)), (x_test - mean) / scale])
            penalty = np.sqrt(ridge) * np.eye(train_design.shape[1])
            penalty[0, 0] = 0.0
            coefficients = np.linalg.lstsq(
                np.vstack([train_design, penalty]),
                np.concatenate([y_train, np.zeros(train_design.shape[1])]),
                rcond=None,
            )[0]
            residuals_by_basis[basis] = y_test - test_design @ coefficients

        for regime in sorted(set(regimes)):
            mask = regimes == regime
            if not mask.any():
                continue
            rmse_by_basis = {
                basis: float(np.sqrt(np.mean(residuals[mask] ** 2)))
                for basis, residuals in residuals_by_basis.items()
            }
            core_rmse = rmse_by_basis["core"]
            best_rmse = min(rmse_by_basis.values())
            for basis, rmse in rmse_by_basis.items():
                lift = 0.0 if core_rmse <= 0.0 else (core_rmse - rmse) / core_rmse
                fold_rows.append(
                    {
                        "fold": int(fold),
                        "regime": regime,
                        "basis": basis,
                        "test_rows": int(mask.sum()),
                        "test_rmse": rmse,
                        "test_rmse_lift_vs_core": float(lift),
                        "is_fold_winner": bool(rmse == best_rmse),
                    }
                )
        fold += 1

    if not fold_rows:
        raise ValueError("at least one rolling fold is required; reduce train_window/test_window")
    detail = pd.DataFrame(fold_rows)
    rows: list[dict[str, float | int | str]] = []
    for (regime, basis), group in detail.groupby(["regime", "basis"], sort=True):
        lifts = group["test_rmse_lift_vs_core"].to_numpy(dtype=float)
        rmse = group["test_rmse"].to_numpy(dtype=float)
        winner = group["is_fold_winner"].astype(bool).to_numpy()
        if not np.isfinite(lifts).all() or not np.isfinite(rmse).all():
            raise ValueError("regime basis comparison metrics must be finite")
        positive_lift_rate = float((lifts > lift_floor).mean())
        winner_rate = float(winner.mean())
        if basis == "nonlinear_liquidity" and positive_lift_rate == 1.0 and winner_rate == 1.0:
            note = "regime_persistent_nonlinear_lift"
        elif basis == "core":
            note = "core_reference"
        elif positive_lift_rate == 1.0:
            note = "regime_positive_lift"
        else:
            note = "regime_unstable_lift"
        rows.append(
            {
                "regime": str(regime),
                "basis": str(basis),
                "folds": int(group["fold"].nunique()),
                "test_rows": int(group["test_rows"].sum()),
                "mean_test_rmse": float(rmse.mean()),
                "mean_test_rmse_lift_vs_core": float(lifts.mean()),
                "positive_lift_rate": positive_lift_rate,
                "winner_rate": winner_rate,
                "worst_fold_lift": float(lifts.min()),
                "publishability_note": note,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def baseline_regime_publishability_summary(
    regime_comparison: pd.DataFrame,
    *,
    min_regime_lift: float = 0.0,
) -> dict[str, bool | float | int | str]:
    """Gate nonlinear baseline publishability on per-regime holdout persistence.

    Aggregate nonlinear lift can be publishable-looking while failing exactly in
    the stressed or thin books that LCRI is meant to diagnose. This summary takes
    ``baseline_regime_basis_comparison`` output and requires the nonlinear basis
    to clear a minimum mean lift, positive-lift persistence, and winner persistence
    in every observed regime; it also exposes the weakest regime for review triage.
    """
    if not math.isfinite(min_regime_lift):
        raise ValueError("min_regime_lift must be finite")
    required = {
        "regime",
        "basis",
        "mean_test_rmse_lift_vs_core",
        "positive_lift_rate",
        "winner_rate",
        "worst_fold_lift",
    }
    missing = sorted(required - set(regime_comparison.columns))
    if missing:
        raise ValueError(f"missing regime publishability columns: {missing}")

    nonlinear = regime_comparison[
        regime_comparison["basis"].astype(str) == "nonlinear_liquidity"
    ].copy()
    if nonlinear.empty:
        raise ValueError("regime_comparison must include nonlinear_liquidity basis")

    metrics = nonlinear[
        [
            "mean_test_rmse_lift_vs_core",
            "positive_lift_rate",
            "winner_rate",
            "worst_fold_lift",
        ]
    ].astype(float)
    if not np.isfinite(metrics.to_numpy()).all():
        raise ValueError("regime publishability metrics must be finite")

    nonlinear = nonlinear.assign(
        mean_lift=metrics["mean_test_rmse_lift_vs_core"],
        positive_lift_rate=metrics["positive_lift_rate"],
        winner_rate=metrics["winner_rate"],
        worst_fold_lift=metrics["worst_fold_lift"],
    )
    supported = (
        (nonlinear["mean_lift"] >= min_regime_lift)
        & (nonlinear["positive_lift_rate"] >= 1.0)
        & (nonlinear["winner_rate"] >= 1.0)
    )
    weakest_index = nonlinear["mean_lift"].astype(float).idxmin()
    weakest_regime = str(nonlinear.loc[weakest_index, "regime"])
    supported_regimes = int(supported.sum())
    regimes = int(nonlinear["regime"].astype(str).nunique())
    publishable = bool(supported_regimes == len(nonlinear) == regimes)
    review_note = (
        "nonlinear_lift_regime_robust" if publishable else "nonlinear_lift_regime_fragile"
    )

    return {
        "regimes": regimes,
        "supported_regimes": supported_regimes,
        "unsupported_regimes": int(regimes - supported_regimes),
        "min_regime_mean_lift": float(nonlinear["mean_lift"].min()),
        "min_regime_worst_fold_lift": float(nonlinear["worst_fold_lift"].min()),
        "min_regime_winner_rate": float(nonlinear["winner_rate"].min()),
        "weakest_regime": weakest_regime,
        "publishable": publishable,
        "review_note": review_note,
    }


def baseline_rolling_basis_summary(
    rolling: pd.DataFrame,
    *,
    lift_floor: float = 0.0,
    max_overfit_ratio: float = 2.0,
) -> pd.DataFrame:
    """Summarize whether rolling baseline lift is persistent enough to cite.

    The row-level rolling comparison is useful for audit trails, but release and
    paper artifacts need a compact stability view: how often each basis wins,
    whether lift is positive in every fold, and whether that lift survives a
    minimum effect-size floor without obvious train/test overfit inflation.
    """
    columns = [
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
    if rolling.empty:
        return pd.DataFrame(columns=columns)
    if not math.isfinite(lift_floor):
        raise ValueError("lift_floor must be finite")
    if not math.isfinite(max_overfit_ratio) or max_overfit_ratio <= 0.0:
        raise ValueError("max_overfit_ratio must be a finite positive value")
    required = {
        "basis",
        "fold",
        "test_rmse_lift_vs_core",
        "test_residual_mean",
        "overfit_ratio",
        "is_fold_winner",
    }
    missing = sorted(required - set(rolling.columns))
    if missing:
        raise ValueError(f"missing rolling basis summary columns: {missing}")

    rows: list[dict[str, bool | float | int | str]] = []
    for basis, group in rolling.groupby("basis", sort=False):
        lifts = group["test_rmse_lift_vs_core"].to_numpy(dtype=float)
        residual_mean = group["test_residual_mean"].to_numpy(dtype=float)
        overfit = group["overfit_ratio"].to_numpy(dtype=float)
        winner = group["is_fold_winner"].astype(bool).to_numpy()
        if not np.isfinite(lifts).all() or not np.isfinite(residual_mean).all():
            raise ValueError("rolling basis summary metrics must be finite")
        if np.isnan(overfit).any():
            raise ValueError("rolling basis overfit ratios must not be NaN")

        min_lift = float(lifts.min())
        median_lift = float(np.median(lifts))
        observed_max_overfit = float(overfit.max())
        stable_lift = bool(min_lift >= lift_floor and observed_max_overfit <= max_overfit_ratio)
        if stable_lift:
            note = "persistent_out_of_sample_lift"
        elif min_lift < lift_floor:
            note = "unstable_or_insufficient_lift"
        else:
            note = "overfit_risk"
        rows.append(
            {
                "basis": str(basis),
                "folds": int(group["fold"].nunique()),
                "winner_rate": float(winner.mean()),
                "positive_lift_rate": float((lifts > 0.0).mean()),
                "median_test_rmse_lift_vs_core": median_lift,
                "min_test_rmse_lift_vs_core": min_lift,
                "max_overfit_ratio": observed_max_overfit,
                "median_test_residual_abs_mean": float(np.median(np.abs(residual_mean))),
                "stable_lift": stable_lift,
                "publishability_note": note,
            }
        )
    output = pd.DataFrame(rows, columns=columns)
    output["stable_lift"] = output["stable_lift"].astype(object)
    return output


def baseline_nonlinear_publishability_summary(
    attribution: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    *,
    coefficient_stability: pd.DataFrame | None = None,
    min_contribution_share: float = 0.50,
    min_median_lift: float = 0.0,
    min_coefficient_sign_consistency: float = 0.80,
) -> dict[str, bool | float | str]:
    """Summarize whether nonlinear liquidity neutralization is publishable.

    Nonlinear baseline claims should clear three hurdles: the fitted baseline
    should allocate meaningful prediction mass to nonlinear liquidity features,
    rolling holdout diagnostics should show stable out-of-sample lift, and (when
    supplied) rolling coefficient audits should show economically interpretable
    sign stability. This compact gate combines attribution, OOS stability, and
    optional coefficient stability so release artifacts can distinguish real
    nonlinear neutralization from an overfit or sign-flipping basis expansion.
    """
    if not math.isfinite(min_contribution_share) or not 0.0 <= min_contribution_share <= 1.0:
        raise ValueError("min_contribution_share must be finite and in [0, 1]")
    if not math.isfinite(min_median_lift):
        raise ValueError("min_median_lift must be finite")
    if (
        not math.isfinite(min_coefficient_sign_consistency)
        or not 0.0 <= min_coefficient_sign_consistency <= 1.0
    ):
        raise ValueError("min_coefficient_sign_consistency must be finite and in [0, 1]")

    attribution_required = {"component", "contribution_share"}
    missing_attribution = sorted(attribution_required - set(attribution.columns))
    if missing_attribution:
        raise ValueError(f"missing nonlinear publishability attribution columns: {missing_attribution}")
    rolling_required = {
        "basis",
        "winner_rate",
        "positive_lift_rate",
        "median_test_rmse_lift_vs_core",
        "min_test_rmse_lift_vs_core",
        "stable_lift",
    }
    missing_rolling = sorted(rolling_required - set(rolling_summary.columns))
    if missing_rolling:
        raise ValueError(f"missing nonlinear publishability rolling columns: {missing_rolling}")
    if attribution.empty:
        raise ValueError("attribution must contain at least one component row")

    contribution = attribution["contribution_share"].to_numpy(dtype=float)
    if not np.isfinite(contribution).all():
        raise ValueError("attribution contribution shares must be finite")
    nonlinear_contribution_share = float(
        attribution.loc[
            attribution["component"].astype(str) == "nonlinear_liquidity",
            "contribution_share",
        ].astype(float).sum()
    )
    nonlinear_rows = rolling_summary[rolling_summary["basis"].astype(str) == "nonlinear_liquidity"]
    if nonlinear_rows.empty:
        raise ValueError("rolling_summary must include nonlinear_liquidity basis")
    nonlinear = nonlinear_rows.iloc[0]

    numeric = {
        "nonlinear_winner_rate": float(nonlinear["winner_rate"]),
        "nonlinear_positive_lift_rate": float(nonlinear["positive_lift_rate"]),
        "nonlinear_median_test_rmse_lift_vs_core": float(
            nonlinear["median_test_rmse_lift_vs_core"]
        ),
        "nonlinear_min_test_rmse_lift_vs_core": float(nonlinear["min_test_rmse_lift_vs_core"]),
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("nonlinear publishability rolling metrics must be finite")
    nonlinear_stable_lift = bool(nonlinear["stable_lift"])

    coefficient_metrics: dict[str, bool | float] = {}
    nonlinear_coefficients_stable = True
    if coefficient_stability is not None:
        stability_required = {"component", "sign_consistency", "stability_label"}
        missing_stability = sorted(stability_required - set(coefficient_stability.columns))
        if missing_stability:
            raise ValueError(
                f"missing nonlinear publishability coefficient stability columns: {missing_stability}"
            )
        nonlinear_stability = coefficient_stability[
            coefficient_stability["component"].astype(str) == "nonlinear_liquidity"
        ]
        if nonlinear_stability.empty:
            raise ValueError("coefficient_stability must include nonlinear_liquidity rows")
        sign_consistency = nonlinear_stability["sign_consistency"].to_numpy(dtype=float)
        if not np.isfinite(sign_consistency).all():
            raise ValueError("coefficient stability sign consistency values must be finite")
        stability_labels = nonlinear_stability["stability_label"].astype(str)
        stable_mask = (sign_consistency >= min_coefficient_sign_consistency) & stability_labels.isin(
            ["sign_stable_dominant", "sign_stable", "inactive"]
        ).to_numpy()
        nonlinear_coefficients_stable = bool(stable_mask.all())
        coefficient_metrics = {
            "nonlinear_min_coefficient_sign_consistency": float(sign_consistency.min()),
            "nonlinear_stable_coefficient_rate": float(stable_mask.mean()),
            "nonlinear_coefficients_stable": nonlinear_coefficients_stable,
        }

    publishable = bool(
        nonlinear_contribution_share >= min_contribution_share
        and numeric["nonlinear_median_test_rmse_lift_vs_core"] >= min_median_lift
        and nonlinear_stable_lift
        and nonlinear_coefficients_stable
    )
    if publishable:
        review_note = "nonlinear_baseline_supported"
    elif not nonlinear_coefficients_stable:
        review_note = "nonlinear_baseline_coefficient_instability"
    else:
        review_note = "nonlinear_baseline_under_supported"
    return {
        "nonlinear_contribution_share": nonlinear_contribution_share,
        **numeric,
        "nonlinear_stable_lift": nonlinear_stable_lift,
        **coefficient_metrics,
        "publishable": publishable,
        "review_note": review_note,
    }


def baseline_nonlinear_stress_surface(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    stress_cols: tuple[str, str] = ("spread_ticks", "volatility"),
    bins: int = 3,
    ridge: float = 1e-3,
    min_lift: float = 0.10,
) -> pd.DataFrame:
    """Localize where nonlinear liquidity terms improve held-out residual control.

    The headline nonlinear baseline lift can hide a brittle improvement isolated to
    one stress corner. This surface trains core and full nonlinear bases on the
    earliest chronological slice, scores the holdout, then bins the holdout by two
    liquidity stress dimensions. Each cell reports core-vs-nonlinear RMSE lift so
    release review can verify that convex stress terms help precisely in stressed
    spread/volatility regions rather than only in benign liquidity.
    """
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 1:
        raise ValueError("bins must be an integer greater than 1")
    if len(stress_cols) != 2:
        raise ValueError("stress_cols must contain exactly two column names")
    if stress_cols[0] == stress_cols[1]:
        raise ValueError("stress_cols must contain two distinct column names")
    if not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and in (0, 1)")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    if not math.isfinite(min_lift):
        raise ValueError("min_lift must be finite")

    bin_columns = [f"{col}_bin" for col in stress_cols]
    columns = [
        "stress_cell",
        *bin_columns,
        "rows",
        "row_share",
        "core_rmse",
        "nonlinear_rmse",
        "rmse_lift_vs_core",
        "core_residual_mean",
        "nonlinear_residual_mean",
        "surface_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    required = {"raw_imbalance", *stress_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing nonlinear stress surface columns: {missing}")
    y = frame["raw_imbalance"].to_numpy(dtype=float)
    stress_values = frame[list(stress_cols)].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(stress_values).all():
        raise ValueError("nonlinear stress surface inputs must be finite")

    train_rows = int(len(frame) * train_fraction)
    if train_rows < 1 or train_rows >= len(frame):
        raise ValueError("train_fraction leaves no train or test rows")

    x = _design_matrix(frame)
    core_indexes = list(range(len(feature_columns())))
    nonlinear_indexes = list(range(len(design_feature_names())))

    def fit_predict(indexes: list[int]) -> np.ndarray:
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
        return test_design @ coefficients

    holdout = frame.iloc[train_rows:].copy()
    holdout_y = y[train_rows:]
    holdout["core_residual"] = holdout_y - fit_predict(core_indexes)
    holdout["nonlinear_residual"] = holdout_y - fit_predict(nonlinear_indexes)

    labels = ["low", "medium", "high"] if bins == 3 else [f"bin_{idx + 1}" for idx in range(bins)]
    for col, bin_col in zip(stress_cols, bin_columns, strict=True):
        pct = holdout[col].rank(method="first", pct=True).to_numpy(dtype=float)
        indexes = np.minimum(np.floor(pct * bins).astype(int), bins - 1)
        holdout[bin_col] = pd.Categorical([labels[idx] for idx in indexes], categories=labels, ordered=True)

    rows: list[dict[str, float | int | str]] = []
    total_rows = len(holdout)
    for keys, group in holdout.groupby(bin_columns, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        core_residual = group["core_residual"].to_numpy(dtype=float)
        nonlinear_residual = group["nonlinear_residual"].to_numpy(dtype=float)
        core_rmse = float(np.sqrt(np.mean(core_residual**2)))
        nonlinear_rmse = float(np.sqrt(np.mean(nonlinear_residual**2)))
        lift = 0.0 if core_rmse <= 0.0 else (core_rmse - nonlinear_rmse) / core_rmse
        if lift >= min_lift:
            label = "nonlinear_supported"
        elif lift < 0.0:
            label = "nonlinear_fragile"
        else:
            label = "neutral"
        row: dict[str, float | int | str] = {
            "stress_cell": "|".join(f"{col}={key}" for col, key in zip(stress_cols, keys, strict=True)),
            "rows": int(len(group)),
            "row_share": float(len(group) / total_rows),
            "core_rmse": core_rmse,
            "nonlinear_rmse": nonlinear_rmse,
            "rmse_lift_vs_core": float(lift),
            "core_residual_mean": float(core_residual.mean()),
            "nonlinear_residual_mean": float(nonlinear_residual.mean()),
            "surface_label": label,
        }
        for bin_col, key in zip(bin_columns, keys, strict=True):
            row[bin_col] = str(key)
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def baseline_nonlinear_stress_surface_summary(
    surface: pd.DataFrame,
    *,
    min_supported_cell_share: float = 0.50,
    min_high_stress_lift: float = 0.10,
    min_weighted_lift: float = 0.05,
) -> dict[str, bool | float | int | str]:
    """Gate nonlinear baseline lift by stressed-cell localization.

    A nonlinear liquidity basis should not earn publishability from benign-cell
    improvements alone. This summary turns the cell-level stress surface into a
    compact review gate: how broad the nonlinear support is, how much row mass it
    covers, whether the high/high stress corner improves, and whether any stress
    cell is outright fragile.
    """
    for name, value in {
        "min_supported_cell_share": min_supported_cell_share,
        "min_high_stress_lift": min_high_stress_lift,
        "min_weighted_lift": min_weighted_lift,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= min_supported_cell_share <= 1.0:
        raise ValueError("min_supported_cell_share must be in [0, 1]")

    bin_columns = [column for column in surface.columns if column.endswith("_bin")]
    required = {"stress_cell", "row_share", "rmse_lift_vs_core", "surface_label", *bin_columns}
    missing = sorted(required - set(surface.columns))
    if missing or not bin_columns:
        raise ValueError(f"missing nonlinear stress surface columns: {missing}")
    if surface.empty:
        raise ValueError("nonlinear stress surface must contain at least one cell")

    numeric = surface[["row_share", "rmse_lift_vs_core"]].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("nonlinear stress surface metrics must be finite")
    if (numeric["row_share"] < 0.0).any():
        raise ValueError("nonlinear stress surface row shares must be non-negative")
    row_share_sum = float(numeric["row_share"].sum())
    if row_share_sum <= 0.0:
        raise ValueError("nonlinear stress surface row shares must sum to a positive value")

    labels = surface["surface_label"].astype(str)
    supported_mask = labels == "nonlinear_supported"
    fragile_mask = labels == "nonlinear_fragile"
    stress_cells = int(len(surface))
    supported_cells = int(supported_mask.sum())
    fragile_cells = int(fragile_mask.sum())
    supported_cell_share = float(supported_cells / stress_cells)
    supported_row_share = float(numeric.loc[supported_mask, "row_share"].sum() / row_share_sum)
    weighted_lift = float((numeric["row_share"] * numeric["rmse_lift_vs_core"]).sum() / row_share_sum)
    worst_cell_lift = float(numeric["rmse_lift_vs_core"].min())

    high_mask = pd.Series(True, index=surface.index)
    for column in bin_columns:
        high_mask &= surface[column].astype(str).str.lower().eq("high")
    if high_mask.any():
        high_row = surface.loc[high_mask].iloc[0]
    else:
        high_row = surface.assign(_stress_rank=_stress_bin_rank(surface, bin_columns)).sort_values(
            ["_stress_rank", "row_share"], ascending=[False, False]
        ).iloc[0]
    high_stress_cell = str(high_row["stress_cell"])
    high_stress_lift = float(high_row["rmse_lift_vs_core"])
    high_stress_label = str(high_row["surface_label"])

    publishable = bool(
        supported_cell_share >= min_supported_cell_share
        and weighted_lift >= min_weighted_lift
        and high_stress_lift >= min_high_stress_lift
        and high_stress_label != "nonlinear_fragile"
        and fragile_cells == 0
    )
    if publishable:
        review_note = "nonlinear_stress_surface_supported"
    elif high_stress_lift < min_high_stress_lift or high_stress_label == "nonlinear_fragile":
        review_note = "nonlinear_high_stress_surface_fragile"
    elif fragile_cells > 0:
        review_note = "nonlinear_stress_surface_fragile_cells"
    else:
        review_note = "nonlinear_stress_surface_under_supported"

    return {
        "stress_cells": stress_cells,
        "supported_cells": supported_cells,
        "fragile_cells": fragile_cells,
        "supported_cell_share": supported_cell_share,
        "supported_row_share": supported_row_share,
        "weighted_rmse_lift_vs_core": weighted_lift,
        "worst_cell_lift": worst_cell_lift,
        "high_stress_cell": high_stress_cell,
        "high_stress_lift": high_stress_lift,
        "publishable": publishable,
        "review_note": review_note,
    }


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


def baseline_tail_lift_diagnostics(
    frame: pd.DataFrame,
    *,
    feature: str,
    train_fraction: float = 0.60,
    tail_quantile: float = 0.20,
    ridge: float = 1e-3,
    min_tail_lift: float = 0.0,
) -> pd.DataFrame:
    """Audit nonlinear baseline lift in holdout liquidity-stress tails.

    Average RMSE lift can mask exactly the failure mode reviewers care about: the
    nonlinear neutralizer may improve the body while leaving convex spread/void or
    volatility tails biased. This chronological holdout diagnostic compares a core
    baseline against the full nonlinear-liquidity basis inside low-tail, body, and
    high-tail slices of a selected design feature.
    """
    columns = [
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
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and in (0, 1)")
    if not math.isfinite(tail_quantile) or not 0.0 < tail_quantile < 0.5:
        raise ValueError("tail_quantile must be finite and in (0, 0.5)")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    if not math.isfinite(min_tail_lift):
        raise ValueError("min_tail_lift must be finite")
    if "raw_imbalance" not in frame.columns:
        raise ValueError("missing tail lift diagnostics columns: ['raw_imbalance']")

    feature_names = design_feature_names()
    if feature not in feature_names:
        raise ValueError(f"unknown design feature: {feature}")
    y = frame["raw_imbalance"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("raw_imbalance values must be finite")
    train_rows = int(len(frame) * train_fraction)
    if train_rows < 1 or train_rows >= len(frame):
        raise ValueError("train_fraction leaves no train or test rows")

    x = _design_matrix(frame)
    feature_values = x[train_rows:, feature_names.index(feature)]
    if not np.isfinite(feature_values).all():
        raise ValueError("tail feature values must be finite")

    def fit_test_residual(indexes: list[int]) -> np.ndarray:
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
        return y[train_rows:] - test_design @ coefficients

    core_residual = fit_test_residual(list(range(len(feature_columns()))))
    nonlinear_residual = fit_test_residual(list(range(len(feature_names))))
    low_cutoff = float(np.quantile(feature_values, tail_quantile))
    high_cutoff = float(np.quantile(feature_values, 1.0 - tail_quantile))
    bucket_masks = [
        ("low_tail", feature_values <= low_cutoff),
        ("body", (feature_values > low_cutoff) & (feature_values < high_cutoff)),
        ("high_tail", feature_values >= high_cutoff),
    ]

    rows: list[dict[str, float | int | str]] = []
    for bucket, mask in bucket_masks:
        if not mask.any():
            continue
        bucket_core = core_residual[mask]
        bucket_nonlinear = nonlinear_residual[mask]
        core_rmse = float(np.sqrt(np.mean(bucket_core**2)))
        nonlinear_rmse = float(np.sqrt(np.mean(bucket_nonlinear**2)))
        lift = 0.0 if core_rmse <= 0.0 else (core_rmse - nonlinear_rmse) / core_rmse
        if lift >= min_tail_lift:
            note = "nonlinear_tail_lift_supported"
        else:
            note = "nonlinear_tail_lift_fragile"
        rows.append(
            {
                "tail_bucket": bucket,
                "feature": feature,
                "test_rows": int(mask.sum()),
                "feature_min": float(feature_values[mask].min()),
                "feature_max": float(feature_values[mask].max()),
                "core_rmse": core_rmse,
                "nonlinear_rmse": nonlinear_rmse,
                "nonlinear_rmse_lift_vs_core": float(lift),
                "core_residual_mean": float(bucket_core.mean()),
                "nonlinear_residual_mean": float(bucket_nonlinear.mean()),
                "tail_publishability_note": note,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def baseline_regime_tail_lift_diagnostics(
    frame: pd.DataFrame,
    *,
    feature: str,
    regime_col: str = "regime",
    train_fraction: float = 0.60,
    tail_quantile: float = 0.20,
    ridge: float = 1e-3,
    min_regime_tail_lift: float = 0.0,
) -> pd.DataFrame:
    """Audit nonlinear baseline lift inside each liquidity regime/stress-tail pocket.

    Aggregate nonlinear lift can still fail where reviewers care most: stressed
    tails within thin or volatile regimes. This chronological holdout diagnostic
    compares core versus full nonlinear residualization after crossing the selected
    stress feature's tail buckets with the test-set liquidity regime labels.
    """
    columns = [
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
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(regime_col, str) or not regime_col:
        raise ValueError("regime_col must be a non-empty string")
    if not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and in (0, 1)")
    if not math.isfinite(tail_quantile) or not 0.0 < tail_quantile < 0.5:
        raise ValueError("tail_quantile must be finite and in (0, 0.5)")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    if not math.isfinite(min_regime_tail_lift):
        raise ValueError("min_regime_tail_lift must be finite")
    missing_required = sorted({"raw_imbalance", regime_col} - set(frame.columns))
    if missing_required:
        raise ValueError(f"missing regime tail lift diagnostics columns: {missing_required}")

    feature_names = design_feature_names()
    if feature not in feature_names:
        raise ValueError(f"unknown design feature: {feature}")
    y = frame["raw_imbalance"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("raw_imbalance values must be finite")
    train_rows = int(len(frame) * train_fraction)
    if train_rows < 1 or train_rows >= len(frame):
        raise ValueError("train_fraction leaves no train or test rows")

    x = _design_matrix(frame)
    feature_values = x[train_rows:, feature_names.index(feature)]
    regimes = frame[regime_col].iloc[train_rows:].astype(str).to_numpy()
    if not np.isfinite(feature_values).all():
        raise ValueError("regime tail feature values must be finite")

    def fit_test_residual(indexes: list[int]) -> np.ndarray:
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
        return y[train_rows:] - test_design @ coefficients

    core_residual = fit_test_residual(list(range(len(feature_columns()))))
    nonlinear_residual = fit_test_residual(list(range(len(feature_names))))
    low_cutoff = float(np.quantile(feature_values, tail_quantile))
    high_cutoff = float(np.quantile(feature_values, 1.0 - tail_quantile))
    bucket_masks = [
        ("low_tail", feature_values <= low_cutoff),
        ("body", (feature_values > low_cutoff) & (feature_values < high_cutoff)),
        ("high_tail", feature_values >= high_cutoff),
    ]

    rows: list[dict[str, float | int | str]] = []
    for regime in sorted(set(regimes)):
        regime_mask = regimes == regime
        for bucket, bucket_mask in bucket_masks:
            mask = regime_mask & bucket_mask
            if not mask.any():
                continue
            bucket_core = core_residual[mask]
            bucket_nonlinear = nonlinear_residual[mask]
            core_rmse = float(np.sqrt(np.mean(bucket_core**2)))
            nonlinear_rmse = float(np.sqrt(np.mean(bucket_nonlinear**2)))
            lift = 0.0 if core_rmse <= 0.0 else (core_rmse - nonlinear_rmse) / core_rmse
            note = (
                "regime_tail_lift_supported"
                if lift >= min_regime_tail_lift
                else "regime_tail_lift_fragile"
            )
            rows.append(
                {
                    "regime": regime,
                    "tail_bucket": bucket,
                    "feature": feature,
                    "test_rows": int(mask.sum()),
                    "feature_min": float(feature_values[mask].min()),
                    "feature_max": float(feature_values[mask].max()),
                    "core_rmse": core_rmse,
                    "nonlinear_rmse": nonlinear_rmse,
                    "nonlinear_rmse_lift_vs_core": float(lift),
                    "core_residual_mean": float(bucket_core.mean()),
                    "nonlinear_residual_mean": float(bucket_nonlinear.mean()),
                    "review_note": note,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def baseline_regime_tail_lift_summary(
    diagnostics: pd.DataFrame,
    *,
    min_regime_tail_lift: float = 0.0,
) -> pd.DataFrame:
    """Gate nonlinear baseline claims on the weakest stress tail inside each regime."""
    columns = [
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
    if diagnostics.empty:
        return pd.DataFrame(columns=columns)
    if not math.isfinite(min_regime_tail_lift):
        raise ValueError("min_regime_tail_lift must be finite")
    required = {
        "feature",
        "regime",
        "tail_bucket",
        "test_rows",
        "nonlinear_rmse_lift_vs_core",
        "core_residual_mean",
        "nonlinear_residual_mean",
    }
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"missing regime tail lift summary columns: {missing}")

    numeric_columns = [
        "test_rows",
        "nonlinear_rmse_lift_vs_core",
        "core_residual_mean",
        "nonlinear_residual_mean",
    ]
    numeric = diagnostics[numeric_columns].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("regime tail lift summary metrics must be finite")
    if (numeric["test_rows"] < 0.0).any():
        raise ValueError("regime tail lift summary test rows must be non-negative")

    rows: list[dict[str, bool | float | int | str]] = []
    for (feature, regime), group in diagnostics.groupby(["feature", "regime"], sort=False):
        lifts = group["nonlinear_rmse_lift_vs_core"].astype(float)
        worst_index = lifts.idxmin()
        supported = lifts >= min_regime_tail_lift
        publishable = bool(supported.all())
        rows.append(
            {
                "feature": str(feature),
                "regime": str(regime),
                "tail_buckets": int(group["tail_bucket"].astype(str).nunique()),
                "test_rows": int(group["test_rows"].astype(float).sum()),
                "min_regime_tail_lift": float(lifts.min()),
                "median_regime_tail_lift": float(lifts.median()),
                "worst_tail_bucket": str(group.loc[worst_index, "tail_bucket"]),
                "supported_tail_buckets": int(supported.sum()),
                "unsupported_tail_buckets": int((~supported).sum()),
                "max_core_residual_abs_mean": float(
                    group["core_residual_mean"].astype(float).abs().max()
                ),
                "max_nonlinear_residual_abs_mean": float(
                    group["nonlinear_residual_mean"].astype(float).abs().max()
                ),
                "publishable": publishable,
                "review_note": (
                    "regime_stress_tail_supported"
                    if publishable
                    else "regime_stress_tail_fragile"
                ),
            }
        )
    output = pd.DataFrame(rows, columns=columns)
    output["publishable"] = output["publishable"].astype(object)
    return output


def baseline_stress_residual_drift(
    frame: pd.DataFrame,
    *,
    feature: str,
    buckets: int = 5,
    train_fraction: float = 0.60,
    ridge: float = 1e-3,
) -> pd.DataFrame:
    """Audit holdout residual drift across liquidity-stress quantile buckets.

    Nonlinear baseline lift is only publication-worthy if it removes the systematic
    residual bias that remains in stressed liquidity states, not merely if it lowers
    aggregate RMSE. This diagnostic fits a core linear basis and the full nonlinear
    liquidity basis on an early chronological training split, buckets the holdout
    by a chosen design feature, and reports how much nonlinear neutralization
    shrinks bucket-level residual means and drift versus the lowest-stress bucket.
    """
    columns = [
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
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(buckets, int) or isinstance(buckets, bool) or buckets < 2:
        raise ValueError("buckets must be an integer greater than 1")
    if not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and in (0, 1)")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    feature_names = design_feature_names()
    if feature not in feature_names:
        raise ValueError(f"unknown design feature: {feature}")
    if "raw_imbalance" not in frame.columns:
        raise ValueError("missing stress residual drift columns: ['raw_imbalance']")

    y = frame["raw_imbalance"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("raw_imbalance values must be finite")
    train_rows = int(len(frame) * train_fraction)
    if train_rows < 1 or train_rows >= len(frame):
        raise ValueError("train_fraction leaves no train or test rows")

    x = _design_matrix(frame)
    feature_values = x[:, feature_names.index(feature)]
    core_indexes = list(range(len(feature_columns())))
    nonlinear_indexes = list(range(len(feature_names)))

    def fit_residual(indexes: list[int]) -> np.ndarray:
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
        return y[train_rows:] - test_design @ coefficients

    core_residual = fit_residual(core_indexes)
    nonlinear_residual = fit_residual(nonlinear_indexes)
    holdout_feature = pd.Series(feature_values[train_rows:]).reset_index(drop=True)
    ranks = holdout_feature.rank(method="first")
    bucket_codes = pd.qcut(ranks, q=min(buckets, len(holdout_feature)), labels=False)

    data = pd.DataFrame(
        {
            "bucket_code": bucket_codes.astype(int),
            "feature_value": holdout_feature,
            "core_residual": core_residual,
            "nonlinear_residual": nonlinear_residual,
        }
    )
    low_core_mean: float | None = None
    low_nonlinear_mean: float | None = None
    rows: list[dict[str, float | int | str]] = []
    for bucket_code, group in data.groupby("bucket_code", sort=True):
        core_mean = float(group["core_residual"].mean())
        nonlinear_mean = float(group["nonlinear_residual"].mean())
        if low_core_mean is None:
            low_core_mean = core_mean
            low_nonlinear_mean = nonlinear_mean
        core_drift = core_mean - low_core_mean
        nonlinear_drift = nonlinear_mean - float(low_nonlinear_mean)
        reduction = abs(core_mean) - abs(nonlinear_mean)
        rows.append(
            {
                "stress_bucket": f"q{int(bucket_code) + 1}",
                "feature": feature,
                "test_rows": int(len(group)),
                "feature_min": float(group["feature_value"].min()),
                "feature_max": float(group["feature_value"].max()),
                "core_residual_mean": core_mean,
                "nonlinear_residual_mean": nonlinear_mean,
                "residual_mean_abs_reduction": float(reduction),
                "core_residual_drift_vs_low_bucket": float(core_drift),
                "nonlinear_residual_drift_vs_low_bucket": float(nonlinear_drift),
                "drift_publishability_note": (
                    "nonlinear_residual_drift_neutralized"
                    if reduction >= 0.0 and abs(nonlinear_drift) <= abs(core_drift) + 1e-12
                    else "nonlinear_residual_drift_fragile"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def baseline_stress_tail_publishability_summary(
    diagnostics: pd.DataFrame,
    *,
    min_tail_lift: float = 0.0,
) -> pd.DataFrame:
    """Summarize whether nonlinear baseline lift survives every stress tail.

    ``baseline_tail_lift_diagnostics`` is intentionally feature/bucket granular.
    This release-facing companion rolls those rows up by stress feature so review
    packets can gate nonlinear neutralization claims on the weakest tail, not the
    average lift that can hide one broken high-stress pocket.
    """
    columns = [
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
    if diagnostics.empty:
        return pd.DataFrame(columns=columns)
    if not math.isfinite(min_tail_lift):
        raise ValueError("min_tail_lift must be finite")
    required = {
        "feature",
        "tail_bucket",
        "test_rows",
        "nonlinear_rmse_lift_vs_core",
        "core_residual_mean",
        "nonlinear_residual_mean",
    }
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"missing stress tail publishability columns: {missing}")

    numeric_columns = [
        "test_rows",
        "nonlinear_rmse_lift_vs_core",
        "core_residual_mean",
        "nonlinear_residual_mean",
    ]
    numeric = diagnostics[numeric_columns].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("stress tail publishability metrics must be finite")
    if (numeric["test_rows"] < 0.0).any():
        raise ValueError("stress tail publishability test rows must be non-negative")

    rows: list[dict[str, bool | float | int | str]] = []
    for feature, group in diagnostics.groupby("feature", sort=False):
        lifts = group["nonlinear_rmse_lift_vs_core"].astype(float)
        worst_index = lifts.idxmin()
        supported = lifts >= min_tail_lift
        publishable = bool(supported.all())
        review_note = (
            "nonlinear_stress_tail_supported"
            if publishable
            else "nonlinear_stress_tail_fragile"
        )
        rows.append(
            {
                "feature": str(feature),
                "tail_buckets": int(group["tail_bucket"].astype(str).nunique()),
                "test_rows": int(group["test_rows"].astype(float).sum()),
                "min_tail_lift": float(lifts.min()),
                "median_tail_lift": float(lifts.median()),
                "worst_tail_bucket": str(group.loc[worst_index, "tail_bucket"]),
                "worst_tail_lift": float(lifts.loc[worst_index]),
                "supported_tail_buckets": int(supported.sum()),
                "unsupported_tail_buckets": int((~supported).sum()),
                "max_core_residual_abs_mean": float(
                    group["core_residual_mean"].astype(float).abs().max()
                ),
                "max_nonlinear_residual_abs_mean": float(
                    group["nonlinear_residual_mean"].astype(float).abs().max()
                ),
                "publishable": publishable,
                "review_note": review_note,
            }
        )
    output = pd.DataFrame(rows, columns=columns)
    output["publishable"] = output["publishable"].astype(object)
    return output


def baseline_nonlinear_feature_ablation(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    ridge: float = 1e-3,
    ablation_features: list[str] | tuple[str, ...] = tuple(NONLINEAR_LIQUIDITY_FEATURES),
    material_drag_share: float = 0.05,
) -> pd.DataFrame:
    """Chronologically ablate nonlinear liquidity terms from the baseline basis.

    Nonlinear LCRI neutralization is stronger when reviewers can see which stress
    terms are indispensable out of sample. This diagnostic fits the full nonlinear
    basis, then drops one requested nonlinear term at a time and measures the
    holdout RMSE drag. Material ablations identify terms carrying genuine liquidity
    curvature rather than cosmetic in-sample complexity.
    """
    columns = [
        "feature",
        "component",
        "train_rows",
        "test_rows",
        "full_nonlinear_rmse",
        "ablated_rmse",
        "ablation_rmse_drag",
        "ablation_rmse_drag_share",
        "full_residual_mean",
        "ablated_residual_mean",
        "ablation_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and in (0, 1)")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    if not math.isfinite(material_drag_share) or material_drag_share < 0.0:
        raise ValueError("material_drag_share must be a finite non-negative value")
    if not ablation_features:
        raise ValueError("ablation_features must be non-empty")
    feature_names = design_feature_names()
    unknown = sorted(set(ablation_features) - set(feature_names))
    if unknown:
        raise ValueError(f"unknown ablation features: {unknown}")
    if "raw_imbalance" not in frame.columns:
        raise ValueError("missing nonlinear feature ablation columns: ['raw_imbalance']")

    y = frame["raw_imbalance"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("raw_imbalance values must be finite")
    train_rows = int(len(frame) * train_fraction)
    if train_rows < 1 or train_rows >= len(frame):
        raise ValueError("train_fraction leaves no train or test rows")

    x = _design_matrix(frame)
    all_indexes = list(range(len(feature_names)))

    def holdout_residual(indexes: list[int]) -> np.ndarray:
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
        return y[train_rows:] - test_design @ coefficients

    full_residual = holdout_residual(all_indexes)
    full_rmse = float(np.sqrt(np.mean(full_residual**2)))
    rows: list[dict[str, float | int | str]] = []
    for feature in ablation_features:
        drop_index = feature_names.index(feature)
        ablated_indexes = [index for index in all_indexes if index != drop_index]
        ablated_residual = holdout_residual(ablated_indexes)
        ablated_rmse = float(np.sqrt(np.mean(ablated_residual**2)))
        drag = float(ablated_rmse - full_rmse)
        drag_share = drag / full_rmse if full_rmse > 0.0 else (1.0 if drag > 0.0 else 0.0)
        rows.append(
            {
                "feature": str(feature),
                "component": _component_for_feature(str(feature)),
                "train_rows": int(train_rows),
                "test_rows": int(len(frame) - train_rows),
                "full_nonlinear_rmse": full_rmse,
                "ablated_rmse": ablated_rmse,
                "ablation_rmse_drag": drag,
                "ablation_rmse_drag_share": float(drag_share),
                "full_residual_mean": float(full_residual.mean()),
                "ablated_residual_mean": float(ablated_residual.mean()),
                "ablation_label": (
                    "material_nonlinear_baseline_term"
                    if drag_share >= material_drag_share
                    else "marginal_nonlinear_baseline_term"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["ablation_rmse_drag", "feature"], ascending=[False, True], ignore_index=True
    )


def baseline_nonlinear_feature_ablation_summary(
    ablation: pd.DataFrame,
    *,
    min_material_terms: int = 1,
    min_total_positive_drag_share: float = 0.10,
    max_negative_drag_share: float = 0.15,
) -> dict[str, bool | float | int | str]:
    """Summarize whether nonlinear liquidity terms are indispensable out of sample.

    ``baseline_nonlinear_feature_ablation`` exposes per-term holdout RMSE drag.
    This compact gate turns that table into a release artifact: nonlinear LCRI
    neutralization is more credible when at least one stress term is materially
    harmful to remove, total positive drag is meaningful, and no term materially
    improves the holdout after removal (a sign of cosmetic or overfit features).
    """
    if not isinstance(min_material_terms, int) or isinstance(min_material_terms, bool):
        raise ValueError("min_material_terms must be an integer")
    if min_material_terms < 0:
        raise ValueError("min_material_terms must be non-negative")
    for name, value in {
        "min_total_positive_drag_share": min_total_positive_drag_share,
        "max_negative_drag_share": max_negative_drag_share,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative value")

    required = {"feature", "ablation_rmse_drag_share", "ablation_label"}
    missing = sorted(required - set(ablation.columns))
    if missing:
        raise ValueError(f"missing nonlinear feature ablation summary columns: {missing}")
    if ablation.empty:
        return {
            "ablation_terms": 0,
            "material_terms": 0,
            "material_term_share": 0.0,
            "total_positive_drag_share": 0.0,
            "max_drag_share": 0.0,
            "max_negative_drag_share": 0.0,
            "strongest_feature": "none",
            "weakest_feature": "none",
            "publishable": False,
            "review_note": "nonlinear_feature_ablation_fragile",
        }

    drag_share = ablation["ablation_rmse_drag_share"].to_numpy(dtype=float)
    if not np.isfinite(drag_share).all():
        raise ValueError("nonlinear feature ablation drag shares must be finite")
    labels = ablation["ablation_label"].astype(str)
    valid_labels = {"material_nonlinear_baseline_term", "marginal_nonlinear_baseline_term"}
    unknown_labels = sorted(set(labels) - valid_labels)
    if unknown_labels:
        raise ValueError(f"invalid nonlinear feature ablation labels: {unknown_labels}")

    material_terms = int((labels == "material_nonlinear_baseline_term").sum())
    ablation_terms = int(len(ablation))
    material_term_share = float(material_terms / ablation_terms) if ablation_terms else 0.0
    positive_drag = np.clip(drag_share, 0.0, None)
    negative_drag = np.clip(-drag_share, 0.0, None)
    total_positive_drag_share = float(positive_drag.sum())
    max_drag_share = float(drag_share.max())
    worst_negative_drag_share = float(negative_drag.max())
    strongest_index = int(np.argmax(drag_share))
    weakest_index = int(np.argmin(drag_share))
    strongest_feature = str(ablation.iloc[strongest_index]["feature"])
    weakest_feature = str(ablation.iloc[weakest_index]["feature"])

    publishable = bool(
        material_terms >= min_material_terms
        and total_positive_drag_share >= min_total_positive_drag_share
        and worst_negative_drag_share <= max_negative_drag_share
    )
    review_note = (
        "nonlinear_feature_ablation_supported"
        if publishable
        else "nonlinear_feature_ablation_fragile"
    )
    return {
        "ablation_terms": ablation_terms,
        "material_terms": material_terms,
        "material_term_share": material_term_share,
        "total_positive_drag_share": total_positive_drag_share,
        "max_drag_share": max_drag_share,
        "max_negative_drag_share": worst_negative_drag_share,
        "strongest_feature": strongest_feature,
        "weakest_feature": weakest_feature,
        "publishable": publishable,
        "review_note": review_note,
    }


def baseline_nonlinear_regularization_path(
    frame: pd.DataFrame,
    *,
    ridges: list[float] | tuple[float, ...] = (0.0, 1e-6, 1e-4, 1e-2, 1.0),
    train_fraction: float = 0.60,
    min_lift: float = 0.0,
) -> pd.DataFrame:
    """Trace nonlinear LCRI baseline lift across ridge strengths.

    Nonlinear liquidity terms are publishable only if the claimed neutralization is
    not a one-ridge artifact. This diagnostic fits core and full nonlinear bases
    on a chronological split for each ridge, then reports out-of-sample lift plus
    coefficient shrinkage so reviewers can identify a robust regularization band.
    """
    columns = [
        "ridge",
        "basis",
        "features",
        "train_rows",
        "test_rows",
        "train_rmse",
        "test_rmse",
        "test_rmse_lift_vs_core",
        "test_residual_mean",
        "coefficient_l2_norm",
        "max_abs_coefficient",
        "support_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not ridges:
        raise ValueError("ridges must be a non-empty sequence")
    ridge_values = [float(ridge) for ridge in ridges]
    if not all(math.isfinite(ridge) and ridge >= 0.0 for ridge in ridge_values):
        raise ValueError("ridges must be finite non-negative values")
    if not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and in (0, 1)")
    if not math.isfinite(min_lift):
        raise ValueError("min_lift must be finite")
    if "raw_imbalance" not in frame.columns:
        raise ValueError("missing nonlinear regularization path columns: ['raw_imbalance']")

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
        "nonlinear_liquidity": list(range(len(feature_names))),
    }

    def fit_for(indexes: list[int], ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return train_design @ coefficients, test_design @ coefficients, coefficients[1:]

    rows: list[dict[str, float | int | str]] = []
    for ridge in sorted(ridge_values):
        fit_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        core_test_rmse: float | None = None
        for basis, indexes in basis_indexes.items():
            train_pred, test_pred, coefficients = fit_for(indexes, ridge)
            fit_results[basis] = (train_pred, test_pred, coefficients)
            if basis == "core":
                core_residual = y[train_rows:] - test_pred
                core_test_rmse = float(np.sqrt(np.mean(core_residual**2)))
        assert core_test_rmse is not None
        for basis, indexes in basis_indexes.items():
            train_pred, test_pred, coefficients = fit_results[basis]
            train_residual = y[:train_rows] - train_pred
            test_residual = y[train_rows:] - test_pred
            train_rmse = float(np.sqrt(np.mean(train_residual**2)))
            test_rmse = float(np.sqrt(np.mean(test_residual**2)))
            lift = 0.0 if core_test_rmse <= 0.0 else (core_test_rmse - test_rmse) / core_test_rmse
            if basis == "core":
                label = "core_reference"
            elif lift >= min_lift:
                label = "supported"
            else:
                label = "fragile"
            abs_coefficients = np.abs(coefficients)
            rows.append(
                {
                    "ridge": float(ridge),
                    "basis": basis,
                    "features": int(len(indexes)),
                    "train_rows": int(train_rows),
                    "test_rows": int(len(frame) - train_rows),
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "test_rmse_lift_vs_core": float(lift),
                    "test_residual_mean": float(test_residual.mean()),
                    "coefficient_l2_norm": float(np.sqrt(np.sum(coefficients**2))),
                    "max_abs_coefficient": float(abs_coefficients.max()) if len(abs_coefficients) else 0.0,
                    "support_label": label,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def baseline_nonlinear_regularization_summary(
    path: pd.DataFrame,
    *,
    min_supported_ridges: int = 2,
    min_median_lift: float = 0.0,
) -> dict[str, bool | float | int | str]:
    """Gate nonlinear baseline publishability on ridge-path robustness."""
    required = {
        "ridge",
        "basis",
        "test_rmse_lift_vs_core",
        "coefficient_l2_norm",
        "support_label",
    }
    missing = sorted(required - set(path.columns))
    if missing:
        raise ValueError(f"missing nonlinear regularization summary columns: {missing}")
    if not isinstance(min_supported_ridges, int) or isinstance(min_supported_ridges, bool):
        raise ValueError("min_supported_ridges must be an integer")
    if min_supported_ridges < 1:
        raise ValueError("min_supported_ridges must be positive")
    if not math.isfinite(min_median_lift):
        raise ValueError("min_median_lift must be finite")

    nonlinear = path[path["basis"].astype(str) == "nonlinear_liquidity"].copy()
    if nonlinear.empty:
        return {
            "ridges": 0,
            "supported_ridges": 0,
            "best_ridge": 0.0,
            "best_lift": 0.0,
            "median_lift": 0.0,
            "min_supported_lift": 0.0,
            "max_supported_coefficient_l2_norm": 0.0,
            "publishable": False,
            "review_note": "nonlinear_regularization_fragile",
        }
    numeric_columns = ["ridge", "test_rmse_lift_vs_core", "coefficient_l2_norm"]
    numeric = nonlinear[numeric_columns].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("nonlinear regularization summary metrics must be finite")
    if (numeric[["ridge", "coefficient_l2_norm"]] < 0.0).any().any():
        raise ValueError("nonlinear regularization summary metrics must be non-negative")

    supported_mask = nonlinear["support_label"].astype(str) == "supported"
    supported = nonlinear[supported_mask]
    best_index = numeric["test_rmse_lift_vs_core"].idxmax()
    supported_lifts = supported["test_rmse_lift_vs_core"].astype(float)
    supported_ridges = int(supported_mask.sum())
    median_lift = float(numeric["test_rmse_lift_vs_core"].median())
    publishable = supported_ridges >= min_supported_ridges and median_lift >= min_median_lift
    return {
        "ridges": int(nonlinear["ridge"].astype(float).nunique()),
        "supported_ridges": supported_ridges,
        "best_ridge": float(nonlinear.loc[best_index, "ridge"]),
        "best_lift": float(nonlinear.loc[best_index, "test_rmse_lift_vs_core"]),
        "median_lift": median_lift,
        "min_supported_lift": float(supported_lifts.min()) if not supported.empty else 0.0,
        "max_supported_coefficient_l2_norm": (
            float(supported["coefficient_l2_norm"].astype(float).max()) if not supported.empty else 0.0
        ),
        "publishable": bool(publishable),
        "review_note": (
            "nonlinear_regularization_supported"
            if publishable
            else "nonlinear_regularization_fragile"
        ),
    }


def baseline_residual_liquidity_orthogonality(
    frame: pd.DataFrame,
    *,
    residual_col: str = "imbalance_residual",
    feature_cols: list[str] | tuple[str, ...] | None = None,
    max_abs_correlation: float = 0.10,
) -> pd.DataFrame:
    """Audit whether post-baseline residuals still load on liquidity state variables.

    A publishable LCRI baseline should remove mechanical liquidity conditioning,
    not merely reduce RMSE. This diagnostic measures univariate residual leakage
    against core, interaction, and nonlinear liquidity features so residual alpha
    claims can be challenged before execution-aware evaluation.
    """
    columns = [
        "feature",
        "component",
        "rows",
        "residual_mean",
        "feature_mean",
        "correlation",
        "abs_correlation",
        "slope",
        "r_squared",
        "orthogonality_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(residual_col, str) or not residual_col:
        raise ValueError("residual_col must be a non-empty string")
    if feature_cols is None:
        selected_features = design_feature_names()
    else:
        selected_features = list(feature_cols)
    if not selected_features:
        raise ValueError("feature_cols must be non-empty when provided")
    if not math.isfinite(max_abs_correlation) or not 0.0 <= max_abs_correlation <= 1.0:
        raise ValueError("max_abs_correlation must be finite and in [0, 1]")

    missing_base = [feature for feature in selected_features if feature not in design_feature_names()]
    required = {residual_col, *missing_base}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing residual orthogonality columns: {missing}")

    design_values = pd.DataFrame(_design_matrix(frame), columns=design_feature_names(), index=frame.index)
    feature_values = pd.DataFrame(index=frame.index)
    for feature in selected_features:
        if feature in design_values.columns:
            feature_values[feature] = design_values[feature]
        else:
            feature_values[feature] = frame[feature].astype(float)
    residual_series = frame[residual_col].astype(float)
    values = pd.concat([residual_series.rename(residual_col), feature_values], axis=1)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("residual orthogonality inputs must be finite")

    residual = values[residual_col].to_numpy(dtype=float)
    residual_centered = residual - residual.mean()
    residual_variance = float(np.mean(residual_centered**2))
    rows: list[dict[str, float | int | str]] = []
    for feature in selected_features:
        feature_values = values[feature].to_numpy(dtype=float)
        feature_centered = feature_values - feature_values.mean()
        feature_variance = float(np.mean(feature_centered**2))
        covariance = float(np.mean(residual_centered * feature_centered))
        if residual_variance > 0.0 and feature_variance > 0.0:
            correlation = covariance / math.sqrt(residual_variance * feature_variance)
            slope = covariance / feature_variance
        else:
            correlation = 0.0
            slope = 0.0
        abs_correlation = abs(float(correlation))
        rows.append(
            {
                "feature": feature,
                "component": _component_for_feature(feature),
                "rows": int(len(frame)),
                "residual_mean": float(residual.mean()),
                "feature_mean": float(feature_values.mean()),
                "correlation": float(correlation),
                "abs_correlation": abs_correlation,
                "slope": float(slope),
                "r_squared": float(abs_correlation**2),
                "orthogonality_label": (
                    "orthogonal"
                    if abs_correlation <= max_abs_correlation
                    else "residual_liquidity_leakage"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["abs_correlation", "feature"], ascending=[False, True], ignore_index=True
    )


def baseline_residual_liquidity_orthogonality_summary(
    diagnostics: pd.DataFrame,
    *,
    max_abs_correlation: float = 0.10,
    min_orthogonal_share: float = 1.0,
) -> dict[str, bool | float | int | str]:
    """Summarize residual-liquidity orthogonality as a nonlinear baseline release gate."""
    required = {"feature", "abs_correlation", "orthogonality_label"}
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"missing residual orthogonality diagnostic columns: {missing}")
    for name, value in {
        "max_abs_correlation": max_abs_correlation,
        "min_orthogonal_share": min_orthogonal_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_abs_correlation <= 1.0:
        raise ValueError("max_abs_correlation must be in [0, 1]")
    if not 0.0 <= min_orthogonal_share <= 1.0:
        raise ValueError("min_orthogonal_share must be in [0, 1]")

    numeric = diagnostics["abs_correlation"].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("residual orthogonality correlations must be finite")
    if not numeric.between(0.0, 1.0).all():
        raise ValueError("abs_correlation must be in [0, 1]")

    features = int(len(diagnostics))
    if features == 0:
        return {
            "features": 0,
            "orthogonal_features": 0,
            "leaking_features": 0,
            "orthogonal_feature_share": 0.0,
            "max_abs_correlation": 0.0,
            "mean_abs_correlation": 0.0,
            "worst_feature": "none",
            "publishable": False,
            "review_note": "baseline_residual_liquidity_leakage",
        }

    orthogonal_mask = (numeric <= max_abs_correlation) & (
        diagnostics["orthogonality_label"].astype(str) == "orthogonal"
    )
    orthogonal_features = int(orthogonal_mask.sum())
    orthogonal_share = orthogonal_features / features
    worst_index = numeric.idxmax()
    publishable = orthogonal_share >= min_orthogonal_share
    return {
        "features": features,
        "orthogonal_features": orthogonal_features,
        "leaking_features": int(features - orthogonal_features),
        "orthogonal_feature_share": float(orthogonal_share),
        "max_abs_correlation": float(numeric.max()),
        "mean_abs_correlation": float(numeric.mean()),
        "worst_feature": str(diagnostics.loc[worst_index, "feature"]),
        "publishable": bool(publishable),
        "review_note": (
            "baseline_residual_liquidity_orthogonal"
            if publishable
            else "baseline_residual_liquidity_leakage"
        ),
    }


def baseline_regime_residual_liquidity_orthogonality(
    frame: pd.DataFrame,
    *,
    residual_col: str = "imbalance_residual",
    regime_col: str = "regime",
    feature_cols: list[str] | tuple[str, ...] | None = None,
    max_abs_correlation: float = 0.10,
) -> pd.DataFrame:
    """Audit post-baseline residual leakage separately inside each liquidity regime.

    Global residual-feature correlations can cancel when calm and stressed books
    have different liquidity-response slopes. This regime-conditioned companion to
    ``baseline_residual_liquidity_orthogonality`` exposes masked leakage so LCRI
    baseline claims are gated where execution and adverse-selection risk differ.
    """
    columns = [
        "regime",
        "feature",
        "component",
        "rows",
        "residual_mean",
        "feature_mean",
        "correlation",
        "abs_correlation",
        "slope",
        "r_squared",
        "orthogonality_label",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(residual_col, str) or not residual_col:
        raise ValueError("residual_col must be a non-empty string")
    if not isinstance(regime_col, str) or not regime_col:
        raise ValueError("regime_col must be a non-empty string")
    if feature_cols is None:
        selected_features = design_feature_names()
    else:
        selected_features = list(feature_cols)
    if not selected_features:
        raise ValueError("feature_cols must be non-empty when provided")
    if not math.isfinite(max_abs_correlation) or not 0.0 <= max_abs_correlation <= 1.0:
        raise ValueError("max_abs_correlation must be finite and in [0, 1]")

    missing_base = [feature for feature in selected_features if feature not in design_feature_names()]
    required = {residual_col, regime_col, *missing_base}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing regime residual orthogonality columns: {missing}")

    design_values = pd.DataFrame(_design_matrix(frame), columns=design_feature_names(), index=frame.index)
    feature_values = pd.DataFrame(index=frame.index)
    for feature in selected_features:
        if feature in design_values.columns:
            feature_values[feature] = design_values[feature]
        else:
            feature_values[feature] = frame[feature].astype(float)
    residual_series = frame[residual_col].astype(float)
    regimes = frame[regime_col].astype(str)
    values = pd.concat([regimes.rename(regime_col), residual_series.rename(residual_col), feature_values], axis=1)
    numeric_values = values[[residual_col, *selected_features]].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("regime residual orthogonality inputs must be finite")

    rows: list[dict[str, float | int | str]] = []
    for regime, group in values.groupby(regime_col, sort=True):
        residual = group[residual_col].to_numpy(dtype=float)
        residual_centered = residual - residual.mean()
        residual_variance = float(np.mean(residual_centered**2))
        for feature in selected_features:
            feature_array = group[feature].to_numpy(dtype=float)
            feature_centered = feature_array - feature_array.mean()
            feature_variance = float(np.mean(feature_centered**2))
            covariance = float(np.mean(residual_centered * feature_centered))
            if residual_variance > 0.0 and feature_variance > 0.0:
                correlation = covariance / math.sqrt(residual_variance * feature_variance)
                slope = covariance / feature_variance
            else:
                correlation = 0.0
                slope = 0.0
            abs_correlation = abs(float(correlation))
            rows.append(
                {
                    "regime": str(regime),
                    "feature": feature,
                    "component": _component_for_feature(feature),
                    "rows": int(len(group)),
                    "residual_mean": float(residual.mean()),
                    "feature_mean": float(feature_array.mean()),
                    "correlation": float(correlation),
                    "abs_correlation": abs_correlation,
                    "slope": float(slope),
                    "r_squared": float(abs_correlation**2),
                    "orthogonality_label": (
                        "regime_orthogonal"
                        if abs_correlation <= max_abs_correlation
                        else "regime_residual_liquidity_leakage"
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["regime", "abs_correlation", "feature"], ascending=[True, False, True], ignore_index=True
    )


def baseline_regime_residual_liquidity_orthogonality_summary(
    diagnostics: pd.DataFrame,
    *,
    max_abs_correlation: float = 0.10,
    min_orthogonal_share: float = 1.0,
) -> pd.DataFrame:
    """Summarize regime-conditioned residual leakage as a nonlinear baseline gate."""
    columns = [
        "regime",
        "features",
        "orthogonal_features",
        "leaking_features",
        "orthogonal_feature_share",
        "max_abs_correlation",
        "mean_abs_correlation",
        "worst_feature",
        "publishable",
        "review_note",
    ]
    if diagnostics.empty:
        return pd.DataFrame(columns=columns)
    required = {"regime", "feature", "abs_correlation", "orthogonality_label"}
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"missing regime residual orthogonality summary columns: {missing}")
    for name, value in {
        "max_abs_correlation": max_abs_correlation,
        "min_orthogonal_share": min_orthogonal_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_abs_correlation <= 1.0:
        raise ValueError("max_abs_correlation must be in [0, 1]")
    if not 0.0 <= min_orthogonal_share <= 1.0:
        raise ValueError("min_orthogonal_share must be in [0, 1]")

    numeric = diagnostics["abs_correlation"].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("regime residual orthogonality correlations must be finite")
    if not numeric.between(0.0, 1.0).all():
        raise ValueError("abs_correlation must be in [0, 1]")

    rows: list[dict[str, bool | float | int | str]] = []
    for regime, group in diagnostics.groupby("regime", sort=True):
        correlations = group["abs_correlation"].astype(float)
        features = int(len(group))
        orthogonal_mask = (correlations <= max_abs_correlation) & (
            group["orthogonality_label"].astype(str) == "regime_orthogonal"
        )
        orthogonal_features = int(orthogonal_mask.sum())
        orthogonal_share = orthogonal_features / features if features else 0.0
        worst_index = correlations.idxmax()
        publishable = bool(features > 0 and orthogonal_share >= min_orthogonal_share)
        rows.append(
            {
                "regime": str(regime),
                "features": features,
                "orthogonal_features": orthogonal_features,
                "leaking_features": int(features - orthogonal_features),
                "orthogonal_feature_share": float(orthogonal_share),
                "max_abs_correlation": float(correlations.max()),
                "mean_abs_correlation": float(correlations.mean()),
                "worst_feature": str(group.loc[worst_index, "feature"]),
                "publishable": publishable,
                "review_note": (
                    "regime_residual_liquidity_orthogonal"
                    if publishable
                    else "regime_residual_liquidity_leakage"
                ),
            }
        )
    output = pd.DataFrame(rows, columns=columns)
    output["publishable"] = output["publishable"].astype(object)
    return output


def baseline_nonlinear_coefficient_stability_summary(
    stability: pd.DataFrame,
    *,
    min_sign_consistency: float = 0.80,
    max_coefficient_cv: float = 1.0,
    min_stable_share: float = 0.75,
) -> dict[str, bool | float | int | str]:
    """Gate nonlinear baseline publishability on rolling coefficient stability.

    RMSE lift alone can hide a nonlinear neutralizer whose stress coefficients are
    economically incoherent across chronological windows. This summary converts
    ``baseline_nonlinear_coefficient_stability`` rows into a compact release gate:
    terms must keep a dominant sign, avoid excessive coefficient dispersion, and
    do so for enough nonlinear features to make the basis interpretable.
    """
    required = {
        "feature",
        "coefficient_cv",
        "sign_consistency",
        "stability_label",
    }
    missing = sorted(required - set(stability.columns))
    if missing:
        raise ValueError(f"missing nonlinear coefficient stability columns: {missing}")
    for name, value in {
        "min_sign_consistency": min_sign_consistency,
        "max_coefficient_cv": max_coefficient_cv,
        "min_stable_share": min_stable_share,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= min_sign_consistency <= 1.0:
        raise ValueError("min_sign_consistency must be in [0, 1]")
    if max_coefficient_cv < 0.0:
        raise ValueError("max_coefficient_cv must be non-negative")
    if not 0.0 <= min_stable_share <= 1.0:
        raise ValueError("min_stable_share must be in [0, 1]")

    columns = ["coefficient_cv", "sign_consistency"]
    numeric = stability[columns].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("nonlinear coefficient stability metrics must be finite")
    if (numeric < 0.0).any().any():
        raise ValueError("nonlinear coefficient stability metrics must be non-negative")
    if not numeric["sign_consistency"].between(0.0, 1.0).all():
        raise ValueError("sign_consistency must be in [0, 1]")

    features = int(len(stability))
    if features == 0:
        return {
            "features": 0,
            "stable_features": 0,
            "unstable_features": 0,
            "stable_feature_share": 0.0,
            "min_sign_consistency": 0.0,
            "max_coefficient_cv": 0.0,
            "weakest_feature": "none",
            "stability_label": "nonlinear_coefficient_instability",
            "publishable": False,
            "review_note": "nonlinear_coefficients_fragile",
        }

    stable_mask = (
        (numeric["sign_consistency"] >= min_sign_consistency)
        & (numeric["coefficient_cv"] <= max_coefficient_cv)
        & (stability["stability_label"].astype(str) != "sign_unstable")
    )
    stable_features = int(stable_mask.sum())
    stable_feature_share = stable_features / features
    weakest_index = numeric.sort_values(
        ["sign_consistency", "coefficient_cv"], ascending=[True, False]
    ).index[0]
    publishable = stable_feature_share >= min_stable_share

    return {
        "features": features,
        "stable_features": stable_features,
        "unstable_features": int(features - stable_features),
        "stable_feature_share": float(stable_feature_share),
        "min_sign_consistency": float(numeric["sign_consistency"].min()),
        "max_coefficient_cv": float(numeric["coefficient_cv"].max()),
        "weakest_feature": str(stability.loc[weakest_index, "feature"]),
        "stability_label": (
            "nonlinear_coefficients_stable"
            if publishable
            else "nonlinear_coefficient_instability"
        ),
        "publishable": bool(publishable),
        "review_note": (
            "nonlinear_coefficients_reliable"
            if publishable
            else "nonlinear_coefficients_fragile"
        ),
    }


def baseline_nonlinear_coefficient_stability(
    frame: pd.DataFrame,
    *,
    train_window: int = 500,
    step: int | None = None,
    ridge: float = 1e-3,
    sign_deadband: float = 1e-10,
) -> pd.DataFrame:
    """Audit rolling coefficient stability for nonlinear liquidity neutralization terms.

    Out-of-sample RMSE lift is necessary but not sufficient for a publishable
    nonlinear LCRI baseline: a flexible basis can win folds while the stress terms
    flip signs across time, making the learned neutralizer economically unstable.
    This diagnostic refits the full standardized ridge basis on rolling
    chronological windows and reports sign consistency plus coefficient dispersion
    for each nonlinear liquidity term.
    """
    columns = [
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
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(train_window, int) or isinstance(train_window, bool) or train_window < 1:
        raise ValueError("train_window must be a positive integer")
    if step is None:
        step = train_window
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    if not math.isfinite(sign_deadband) or sign_deadband < 0.0:
        raise ValueError("sign_deadband must be a finite non-negative value")
    if "raw_imbalance" not in frame.columns:
        raise ValueError("missing coefficient stability columns: ['raw_imbalance']")

    y = frame["raw_imbalance"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("raw_imbalance values must be finite")
    x = _design_matrix(frame)
    feature_names = design_feature_names()
    nonlinear_indexes = [feature_names.index(feature) for feature in NONLINEAR_LIQUIDITY_FEATURES]
    if train_window > len(frame):
        raise ValueError("at least one coefficient stability window is required")

    coefficients_by_window: list[np.ndarray] = []
    for start in range(0, len(frame) - train_window + 1, step):
        end = start + train_window
        x_train = x[start:end]
        y_train = y[start:end]
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale == 0.0] = 1.0
        train_design = np.column_stack([np.ones(train_window), (x_train - mean) / scale])
        penalty = np.sqrt(ridge) * np.eye(train_design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.lstsq(
            np.vstack([train_design, penalty]),
            np.concatenate([y_train, np.zeros(train_design.shape[1])]),
            rcond=None,
        )[0]
        coefficients_by_window.append(coefficients[1:][nonlinear_indexes])
    if not coefficients_by_window:
        raise ValueError("at least one coefficient stability window is required")

    coefficient_matrix = np.vstack(coefficients_by_window)
    rows: list[dict[str, float | int | str]] = []
    for offset, feature in enumerate(NONLINEAR_LIQUIDITY_FEATURES):
        values = coefficient_matrix[:, offset]
        abs_values = np.abs(values)
        mean_coefficient = float(values.mean())
        std_coefficient = float(values.std())
        mean_abs_coefficient = float(abs_values.mean())
        coefficient_cv = (
            float(std_coefficient / mean_abs_coefficient) if mean_abs_coefficient > 0.0 else 0.0
        )
        active = abs_values > sign_deadband
        if active.any():
            signs = np.sign(values[active])
            positive_share = float((signs > 0.0).mean())
            negative_share = float((signs < 0.0).mean())
            sign_consistency = max(positive_share, negative_share)
        else:
            sign_consistency = 0.0
        if sign_consistency >= 0.95 and coefficient_cv <= 0.50:
            label = "sign_stable_dominant"
        elif sign_consistency >= 0.80:
            label = "sign_stable"
        elif mean_abs_coefficient <= sign_deadband:
            label = "inactive"
        else:
            label = "sign_unstable"
        rows.append(
            {
                "feature": feature,
                "component": _component_for_feature(feature),
                "windows": int(coefficient_matrix.shape[0]),
                "mean_coefficient": mean_coefficient,
                "std_coefficient": std_coefficient,
                "mean_abs_coefficient": mean_abs_coefficient,
                "coefficient_cv": coefficient_cv,
                "sign_consistency": float(sign_consistency),
                "stability_label": label,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["mean_abs_coefficient", "sign_consistency", "feature"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def baseline_nonlinear_extrapolation_risk(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] | list[str] | None = None,
    train_quantile: float = 0.95,
    max_safe_out_of_support_share: float = 0.10,
) -> pd.DataFrame:
    """Audit nonlinear-baseline extrapolation risk on a chronological holdout.

    Nonlinear liquidity terms can look impressive in-sample while becoming fragile
    when the evaluation window lives outside the training support. This diagnostic
    compares evaluation nonlinear basis values against central training quantile
    support and labels terms whose holdout mass materially leaves that support.
    """
    columns = [
        "feature",
        "train_low",
        "train_high",
        "eval_min",
        "eval_max",
        "out_of_support_share",
        "mean_standardized_shift",
        "max_standardized_shift",
        "risk_label",
    ]
    if train_frame.empty or evaluation_frame.empty:
        return pd.DataFrame(columns=columns)
    if not math.isfinite(train_quantile) or not 0.0 < train_quantile < 1.0:
        raise ValueError("train_quantile must be finite and in (0, 1)")
    if (
        not math.isfinite(max_safe_out_of_support_share)
        or not 0.0 <= max_safe_out_of_support_share <= 1.0
    ):
        raise ValueError("max_safe_out_of_support_share must be finite and in [0, 1]")

    all_feature_names = design_feature_names()
    nonlinear_names = list(NONLINEAR_LIQUIDITY_FEATURES)
    if feature_names is None:
        selected_features = nonlinear_names
    else:
        selected_features = list(feature_names)
        unknown = sorted(set(selected_features) - set(nonlinear_names))
        if unknown:
            raise ValueError(f"unknown nonlinear extrapolation features: {unknown}")
        if not selected_features:
            raise ValueError("feature_names must be non-empty when provided")

    train_design = pd.DataFrame(_design_matrix(train_frame), columns=all_feature_names)
    eval_design = pd.DataFrame(_design_matrix(evaluation_frame), columns=all_feature_names)
    tail = (1.0 - train_quantile) / 2.0
    rows: list[dict[str, float | str]] = []
    for feature in selected_features:
        train_values = train_design[feature].to_numpy(dtype=float)
        eval_values = eval_design[feature].to_numpy(dtype=float)
        train_low = float(np.quantile(train_values, tail))
        train_high = float(np.quantile(train_values, 1.0 - tail))
        train_center = float(np.mean(train_values))
        train_scale = float(np.std(train_values)) or 1.0
        standardized_shift = np.abs((eval_values - train_center) / train_scale)
        out_of_support = (eval_values < train_low) | (eval_values > train_high)
        out_share = float(np.mean(out_of_support))
        rows.append(
            {
                "feature": feature,
                "train_low": train_low,
                "train_high": train_high,
                "eval_min": float(np.min(eval_values)),
                "eval_max": float(np.max(eval_values)),
                "out_of_support_share": out_share,
                "mean_standardized_shift": float(np.mean(standardized_shift)),
                "max_standardized_shift": float(np.max(standardized_shift)),
                "risk_label": "extrapolation_risk"
                if out_share > max_safe_out_of_support_share
                else "inside_train_support",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def baseline_nonlinear_extrapolation_risk_summary(
    risk: pd.DataFrame,
    *,
    max_safe_risky_terms: int = 0,
    max_safe_out_of_support_share: float = 0.10,
) -> dict[str, float | int | str | bool]:
    """Summarize nonlinear-baseline holdout support risk for release review.

    ``baseline_nonlinear_extrapolation_risk`` is feature-level; this release gate
    condenses it into a single publishability decision so demo/reporting artifacts
    can block nonlinear neutralizers whose holdout window materially leaves the
    training support.
    """
    required = {
        "feature",
        "out_of_support_share",
        "mean_standardized_shift",
        "max_standardized_shift",
        "risk_label",
    }
    missing = sorted(required - set(risk.columns))
    if missing:
        raise ValueError(f"missing nonlinear extrapolation risk summary columns: {missing}")
    if not isinstance(max_safe_risky_terms, int) or isinstance(max_safe_risky_terms, bool):
        raise ValueError("max_safe_risky_terms must be an integer")
    if max_safe_risky_terms < 0:
        raise ValueError("max_safe_risky_terms must be non-negative")
    if (
        not math.isfinite(max_safe_out_of_support_share)
        or not 0.0 <= max_safe_out_of_support_share <= 1.0
    ):
        raise ValueError("max_safe_out_of_support_share must be finite and in [0, 1]")

    if risk.empty:
        return {
            "terms": 0,
            "risky_terms": 0,
            "max_out_of_support_share": 0.0,
            "mean_out_of_support_share": 0.0,
            "max_standardized_shift": 0.0,
            "worst_feature": "",
            "publishable": False,
            "review_note": "nonlinear_extrapolation_missing",
        }

    numeric_columns = [
        "out_of_support_share",
        "mean_standardized_shift",
        "max_standardized_shift",
    ]
    numeric = risk[numeric_columns].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("nonlinear extrapolation risk summary values must be finite")
    if not numeric["out_of_support_share"].between(0.0, 1.0).all():
        raise ValueError("out_of_support_share must be in [0, 1]")
    valid_labels = {"inside_train_support", "extrapolation_risk"}
    labels = risk["risk_label"].astype(str)
    unknown_labels = sorted(set(labels) - valid_labels)
    if unknown_labels:
        raise ValueError(f"unknown nonlinear extrapolation risk labels: {unknown_labels}")

    worst_idx = numeric.sort_values(
        ["out_of_support_share", "max_standardized_shift"], ascending=[False, False]
    ).index[0]
    risky = labels.eq("extrapolation_risk") | numeric["out_of_support_share"].gt(
        max_safe_out_of_support_share
    )
    risky_terms = int(risky.sum())
    max_out_of_support = float(numeric["out_of_support_share"].max())
    publishable = bool(
        risky_terms <= max_safe_risky_terms
        and max_out_of_support <= max_safe_out_of_support_share
    )
    return {
        "terms": int(len(risk)),
        "risky_terms": risky_terms,
        "max_out_of_support_share": max_out_of_support,
        "mean_out_of_support_share": float(numeric["out_of_support_share"].mean()),
        "max_standardized_shift": float(numeric["max_standardized_shift"].max()),
        "worst_feature": str(risk.loc[worst_idx, "feature"]),
        "publishable": publishable,
        "review_note": "nonlinear_extrapolation_supported"
        if publishable
        else "nonlinear_extrapolation_fragile",
    }


def _stress_bin_rank(surface: pd.DataFrame, bin_columns: list[str]) -> pd.Series:
    rank_map = {"low": 0.0, "medium": 1.0, "high": 2.0}
    ranks = pd.Series(0.0, index=surface.index)
    for column in bin_columns:
        labels = surface[column].astype(str).str.lower()
        mapped = labels.map(rank_map)
        if mapped.isna().any():
            fallback = pd.Series(pd.factorize(labels, sort=True)[0], index=surface.index, dtype=float)
            mapped = mapped.fillna(fallback)
        ranks = ranks + mapped.astype(float)
    return ranks


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
