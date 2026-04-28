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
