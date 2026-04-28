import pytest
import numpy as np

from lcri_lab.model import ARTIFACT_VERSION, LCRIModel, ModelConfig
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def test_model_scores_and_persists(tmp_path) -> None:
    books = simulate_order_books(SimulationConfig(rows=800, seed=12))
    train = books.iloc[:500]
    test = books.iloc[500:]

    model = LCRIModel().fit(train)
    scored = model.score_frame(test)

    assert "lcri" in scored.columns
    assert "lcri_probability" in scored.columns
    assert scored["lcri_probability"].between(0, 1).all()

    path = tmp_path / "model.json"
    model.save(path)
    payload = path.read_text()
    assert f'"schema_version": {ARTIFACT_VERSION}' in payload
    loaded = LCRIModel.load(path)

    original = model.predict_proba(test)
    restored = loaded.predict_proba(test)
    assert np.allclose(original, restored)


def test_model_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="levels"):
        ModelConfig(levels=0)
    with pytest.raises(ValueError, match="ridge"):
        ModelConfig(ridge=-1.0)
    with pytest.raises(ValueError, match="probability_scale"):
        ModelConfig(probability_scale=0.0)


def test_model_load_rejects_incomplete_artifact(tmp_path) -> None:
    path = tmp_path / "model.json"
    path.write_text('{"config": {"levels": 5}}')

    with pytest.raises(ValueError, match="missing keys"):
        LCRIModel.load(path)


def test_model_load_rejects_unsupported_artifact_version(tmp_path) -> None:
    books = simulate_order_books(SimulationConfig(rows=120, seed=15))
    path = tmp_path / "model.json"
    LCRIModel().fit(books).save(path)
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 99'))

    with pytest.raises(ValueError, match="schema_version"):
        LCRIModel.load(path)


def test_model_load_rejects_non_finite_artifact_values(tmp_path) -> None:
    books = simulate_order_books(SimulationConfig(rows=120, seed=14))
    path = tmp_path / "model.json"
    LCRIModel().fit(books).save(path)
    payload = path.read_text().replace('"coefficients": [', '"coefficients": [NaN, ', 1)
    path.write_text(payload)

    with pytest.raises(ValueError, match="non-finite coefficients"):
        LCRIModel.load(path)
