import lcri_lab


def test_public_api_exports_calibration_curve() -> None:
    assert callable(lcri_lab.calibration_curve)
    assert "calibration_curve" in lcri_lab.__all__


def test_public_api_exports_artifact_version() -> None:
    assert lcri_lab.ARTIFACT_VERSION == 1
    assert "ARTIFACT_VERSION" in lcri_lab.__all__


def test_public_api_exports_design_feature_names() -> None:
    assert "spread_x_replenishment" in lcri_lab.design_feature_names()
    assert "design_feature_names" in lcri_lab.__all__
