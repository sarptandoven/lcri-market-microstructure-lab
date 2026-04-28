import lcri_lab


def test_public_api_exports_calibration_curve() -> None:
    assert callable(lcri_lab.calibration_curve)
    assert "calibration_curve" in lcri_lab.__all__
