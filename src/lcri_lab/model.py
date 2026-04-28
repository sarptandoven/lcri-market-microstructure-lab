from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lcri_lab.baseline import LiquidityBaseline, compute_lcri, design_feature_names
from lcri_lab.features import compute_features


ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ModelConfig:
    levels: int = 5
    ridge: float = 1e-3
    probability_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.levels < 1:
            raise ValueError("levels must be at least 1")
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be a finite non-negative value")
        if not math.isfinite(self.probability_scale) or self.probability_scale <= 0.0:
            raise ValueError("probability_scale must be a finite positive value")


class LCRIModel:
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self.baseline = LiquidityBaseline(ridge=self.config.ridge)
        self.is_fit = False

    def fit(self, order_books: pd.DataFrame) -> "LCRIModel":
        features = compute_features(order_books, levels=self.config.levels)
        self.baseline.fit(features)
        self.is_fit = True
        return self

    def transform(self, order_books: pd.DataFrame) -> pd.DataFrame:
        self._require_fit()
        features = compute_features(order_books, levels=self.config.levels)
        return compute_lcri(features, self.baseline)

    def score_frame(self, order_books: pd.DataFrame) -> pd.DataFrame:
        scored = self.transform(order_books)
        scored["lcri_probability"] = self.predict_proba_from_scores(scored["lcri"].to_numpy())
        scored["lcri_direction"] = (scored["lcri"] > 0.0).astype(int)
        return scored

    def predict_proba(self, order_books: pd.DataFrame) -> np.ndarray:
        scored = self.transform(order_books)
        return self.predict_proba_from_scores(scored["lcri"].to_numpy())

    def save(self, path: str | Path) -> None:
        self._require_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": ARTIFACT_VERSION,
            "config": asdict(self.config),
            "feature_names": design_feature_names(),
            "coefficients": self.baseline.coefficients.tolist(),
            "mean": self.baseline.mean_.tolist(),
            "scale": self.baseline.scale_.tolist(),
            "residual_scale_by_regime": self.baseline.residual_scale_by_regime,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "LCRIModel":
        payload = json.loads(Path(path).read_text())
        required = {
            "schema_version",
            "config",
            "feature_names",
            "coefficients",
            "mean",
            "scale",
            "residual_scale_by_regime",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"model artifact is missing keys: {missing}")
        if payload["schema_version"] != ARTIFACT_VERSION:
            raise ValueError(
                f"unsupported model artifact schema_version: {payload['schema_version']}"
            )
        model = cls(ModelConfig(**payload["config"]))
        expected_features = design_feature_names()
        if payload["feature_names"] != expected_features:
            raise ValueError("model artifact feature_names do not match this package version")
        model.baseline.coefficients = np.array(payload["coefficients"], dtype=float)
        model.baseline.mean_ = np.array(payload["mean"], dtype=float)
        model.baseline.scale_ = np.array(payload["scale"], dtype=float)
        for name, values in {
            "coefficients": model.baseline.coefficients,
            "mean": model.baseline.mean_,
            "scale": model.baseline.scale_,
        }.items():
            if not np.isfinite(values).all():
                raise ValueError(f"model artifact contains non-finite {name}")
        model.baseline.residual_scale_by_regime = {
            str(key): float(value)
            for key, value in payload["residual_scale_by_regime"].items()
        }
        model.is_fit = True
        return model

    def artifact_version(self) -> int:
        return ARTIFACT_VERSION

    def predict_proba_from_scores(self, scores: np.ndarray) -> np.ndarray:
        scaled = np.asarray(scores, dtype=float) / self.config.probability_scale
        clipped = np.clip(scaled, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _require_fit(self) -> None:
        if not self.is_fit:
            raise RuntimeError("model must be fit before scoring")
