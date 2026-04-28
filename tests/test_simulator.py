import pytest

from lcri_lab.simulator import SimulationConfig


def test_simulation_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="rows"):
        SimulationConfig(rows=0)
    with pytest.raises(ValueError, match="levels"):
        SimulationConfig(levels=0)
    with pytest.raises(ValueError, match="tick_size"):
        SimulationConfig(tick_size=0.0)
    with pytest.raises(ValueError, match="initial_mid"):
        SimulationConfig(initial_mid=float("nan"))
