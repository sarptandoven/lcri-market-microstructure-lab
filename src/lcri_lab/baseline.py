from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from lcri_lab.features import feature_columns


@dataclass
class LiquidityBaseline:
    ridge: float = 1e-3
    coefficients: Optional[np.ndarray] = None
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    residual_scale_by_regime: Optional[dict[str, float]] = None

    def fit(self, frame: pd.DataFrame) -> "LiquidityBaseline":
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


def _design_matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = feature_columns()
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    x = frame[cols].to_numpy(dtype=float)
    interactions = np.column_stack(
        [
            frame["spread_ticks"].to_numpy(dtype=float) * frame["replenishment_rate"].to_numpy(dtype=float),
            frame["volatility"].to_numpy(dtype=float) * frame["spread_depth_ratio"].to_numpy(dtype=float),
            frame["log_total_depth"].to_numpy(dtype=float) * frame["depth_slope"].to_numpy(dtype=float),
        ]
    )
    return np.column_stack([x, interactions])
