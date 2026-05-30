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
    min_contribution_share: float = 0.50,
    min_median_lift: float = 0.0,
) -> dict[str, bool | float | str]:
    """Summarize whether nonlinear liquidity neutralization is publishable.

    Nonlinear baseline claims should clear two hurdles: the fitted baseline should
    actually allocate meaningful prediction mass to nonlinear liquidity features,
    and rolling holdout diagnostics should show stable out-of-sample lift. This
    compact gate combines component attribution with rolling basis stability so
    release artifacts can distinguish real nonlinear neutralization from an
    overfit or decorative basis expansion.
    """
    if not math.isfinite(min_contribution_share) or not 0.0 <= min_contribution_share <= 1.0:
        raise ValueError("min_contribution_share must be finite and in [0, 1]")
    if not math.isfinite(min_median_lift):
        raise ValueError("min_median_lift must be finite")

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
    publishable = bool(
        nonlinear_contribution_share >= min_contribution_share
        and numeric["nonlinear_median_test_rmse_lift_vs_core"] >= min_median_lift
        and nonlinear_stable_lift
    )
    review_note = (
        "nonlinear_baseline_supported" if publishable else "nonlinear_baseline_under_supported"
    )
    return {
        "nonlinear_contribution_share": nonlinear_contribution_share,
        **numeric,
        "nonlinear_stable_lift": nonlinear_stable_lift,
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
