"""Liquidity-conditioned residual imbalance research tools."""

from lcri_lab.baseline import LiquidityBaseline, compute_lcri, design_feature_names
from lcri_lab.evaluation import calibration_curve, evaluate_signals, regime_metrics, summarize_signal_lift
from lcri_lab.features import compute_features
from lcri_lab.model import ARTIFACT_VERSION, LCRIModel, ModelConfig
from lcri_lab.schema import snapshot_required_columns
from lcri_lab.simulator import SimulationConfig, simulate_order_books

__all__ = [
    "ARTIFACT_VERSION",
    "LCRIModel",
    "LiquidityBaseline",
    "ModelConfig",
    "SimulationConfig",
    "calibration_curve",
    "compute_features",
    "compute_lcri",
    "design_feature_names",
    "evaluate_signals",
    "regime_metrics",
    "summarize_signal_lift",
    "simulate_order_books",
    "snapshot_required_columns",
]
