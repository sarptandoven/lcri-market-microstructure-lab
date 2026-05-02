import lcri_lab


def test_public_api_exports_calibration_curve() -> None:
    assert callable(lcri_lab.calibration_curve)
    assert "calibration_curve" in lcri_lab.__all__


def test_public_api_exports_artifact_version() -> None:
    assert lcri_lab.ARTIFACT_VERSION == 2
    assert "ARTIFACT_VERSION" in lcri_lab.__all__


def test_public_api_exports_design_feature_names() -> None:
    names = lcri_lab.design_feature_names()
    assert "spread_x_replenishment" in names
    assert "imbalance_fracture" in names
    assert "design_feature_names" in lcri_lab.__all__


def test_public_api_exports_signal_lift_summary() -> None:
    assert callable(lcri_lab.summarize_signal_lift)
    assert "summarize_signal_lift" in lcri_lab.__all__


def test_public_api_exports_signal_generalization_gap() -> None:
    assert callable(lcri_lab.signal_generalization_gap)
    assert "signal_generalization_gap" in lcri_lab.__all__


def test_public_api_exports_regime_generalization_gap() -> None:
    assert callable(lcri_lab.regime_generalization_gap)
    assert "regime_generalization_gap" in lcri_lab.__all__


def test_public_api_exports_regime_tagger() -> None:
    assert callable(lcri_lab.tag_liquidity_regimes)
    assert "tag_liquidity_regimes" in lcri_lab.__all__


def test_public_api_exports_cost_aware_labels() -> None:
    assert callable(lcri_lab.add_transaction_cost_labels)
    assert "add_transaction_cost_labels" in lcri_lab.__all__


def test_public_api_exports_cost_aware_evaluation() -> None:
    assert callable(lcri_lab.evaluate_cost_aware_signals)
    assert "evaluate_cost_aware_signals" in lcri_lab.__all__


def test_public_api_exports_transition_conditioned_metrics() -> None:
    assert callable(lcri_lab.transition_conditioned_metrics)
    assert "transition_conditioned_metrics" in lcri_lab.__all__


def test_public_api_exports_transition_signal_lift() -> None:
    assert callable(lcri_lab.transition_signal_lift)
    assert "transition_signal_lift" in lcri_lab.__all__


def test_public_api_exports_transition_robustness_summary() -> None:
    assert callable(lcri_lab.transition_robustness_summary)
    assert "transition_robustness_summary" in lcri_lab.__all__
