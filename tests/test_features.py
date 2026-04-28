from lcri_lab.features import compute_features
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def test_compute_features_adds_expected_columns() -> None:
    books = simulate_order_books(SimulationConfig(rows=100, seed=1))
    features = compute_features(books)

    for column in [
        "raw_imbalance",
        "top_imbalance",
        "total_depth",
        "log_total_depth",
        "depth_slope",
        "liquidity_score",
    ]:
        assert column in features.columns

    assert features["raw_imbalance"].between(-1, 1).all()
    assert features["total_depth"].gt(0).all()


def test_compute_features_rejects_incomplete_snapshots() -> None:
    books = simulate_order_books(SimulationConfig(rows=10, seed=4)).drop(columns=["bid_sz_3"])

    try:
        compute_features(books)
    except ValueError as exc:
        assert "bid_sz_3" in str(exc)
    else:
        raise AssertionError("expected missing column validation")


def test_compute_features_rejects_non_finite_inputs() -> None:
    books = simulate_order_books(SimulationConfig(rows=10, seed=5))
    books.loc[0, "volatility"] = float("nan")

    try:
        compute_features(books)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected finite-value validation")


def test_compute_features_rejects_negative_sizes() -> None:
    books = simulate_order_books(SimulationConfig(rows=10, seed=6))
    books.loc[0, "ask_sz_1"] = -1.0

    try:
        compute_features(books)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected non-negative size validation")
