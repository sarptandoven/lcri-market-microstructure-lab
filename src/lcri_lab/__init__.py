"""Liquidity-conditioned residual imbalance research tools."""

from lcri_lab.absorption import add_shadow_absorption
from lcri_lab.baseline import LiquidityBaseline, compute_lcri, design_feature_names
from lcri_lab.evaluation import (
    absorption_regime_metrics,
    calibration_curve,
    compare_transmission_signal,
    evaluate_signals,
    lcri_tail_diagnostics,
    regime_metrics,
    summarize_signal_lift,
)
from lcri_lab.features import compute_features, tag_liquidity_regimes
from lcri_lab.labels import add_transaction_cost_labels
from lcri_lab.memory import add_pressure_memory
from lcri_lab.model import ARTIFACT_VERSION, LCRIModel, ModelConfig
from lcri_lab.publishability import PublishabilityConfig, add_publishability_gate
from lcri_lab.reversal import add_queue_reversal_risk
from lcri_lab.schema import snapshot_required_columns
from lcri_lab.simulator import SimulationConfig, simulate_order_books

__all__ = [
    "ARTIFACT_VERSION",
    "absorption_regime_metrics",
    "add_transaction_cost_labels",
    "add_pressure_memory",
    "add_shadow_absorption",
    "add_publishability_gate",
    "add_queue_reversal_risk",
    "LCRIModel",
    "LiquidityBaseline",
    "ModelConfig",
    "PublishabilityConfig",
    "SimulationConfig",
    "calibration_curve",
    "compare_transmission_signal",
    "compute_features",
    "compute_lcri",
    "design_feature_names",
    "evaluate_signals",
    "lcri_tail_diagnostics",
    "regime_metrics",
    "summarize_signal_lift",
    "tag_liquidity_regimes",
    "simulate_order_books",
    "snapshot_required_columns",
]
