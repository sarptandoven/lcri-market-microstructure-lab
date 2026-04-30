import pytest

from lcri_lab.features import compute_features, tag_liquidity_regimes
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
        "imbalance_fracture",
        "liquidity_void_ratio",
        "depth_convexity",
        "resilience_asymmetry",
        "liquidity_score",
    ]:
        assert column in features.columns

    assert features["raw_imbalance"].between(-1, 1).all()
    assert features["liquidity_void_ratio"].between(0, 1).all()
    assert features["total_depth"].gt(0).all()


def test_liquidity_fracture_features_capture_depth_cliffs() -> None:
    books = simulate_order_books(SimulationConfig(rows=1, levels=3, seed=11))
    books.loc[0, ["bid_sz_1", "ask_sz_1"]] = [12.0, 8.0]
    books.loc[0, ["bid_sz_2", "ask_sz_2"]] = [5.0, 5.0]
    books.loc[0, ["bid_sz_3", "ask_sz_3"]] = [3.0, 2.0]

    features = compute_features(books, levels=3)

    assert features.loc[0, "liquidity_void_ratio"] == pytest.approx(10.0 / 35.0)
    assert features.loc[0, "depth_convexity"] == pytest.approx(5.0 / 35.0)
    assert features.loc[0, "imbalance_fracture"] == pytest.approx(
        features.loc[0, "top_imbalance"] - features.loc[0, "raw_imbalance"]
    )


def test_tag_liquidity_regimes_overwrites_unclassified_regime() -> None:
    books = simulate_order_books(SimulationConfig(rows=100, seed=2))
    features = compute_features(books).assign(regime="unclassified")

    tagged = tag_liquidity_regimes(features)

    assert set(tagged["regime"]).issubset({"thick", "thin", "stressed", "replenishing"})
    assert tagged["regime"].nunique() > 1


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


def test_compute_features_rejects_invalid_spreads() -> None:
    books = simulate_order_books(SimulationConfig(rows=10, seed=7))
    books.loc[0, "spread"] = 0.0

    try:
        compute_features(books)
    except ValueError as exc:
        assert "spread" in str(exc)
    else:
        raise AssertionError("expected spread validation")
