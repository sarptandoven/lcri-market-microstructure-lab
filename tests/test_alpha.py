import pandas as pd
import pytest

from lcri_lab.alpha import (
    add_microstructure_alpha_stack,
    alpha_research_gate,
    microstructure_alpha_regime_summary,
)


def test_microstructure_alpha_stack_scores_toxic_resonance() -> None:
    frame = pd.DataFrame(
        {
            "pressure_memory": [0.2, 0.4, 0.8, 1.1],
            "latent_liquidity_fracture": [0.1, 0.3, 0.7, 0.9],
            "gross_return_ticks": [1.0, -1.0, -1.0, -2.0],
            "spread_ticks": [1.0, 1.0, 2.0, 2.0],
            "top_depth": [100.0, 80.0, 60.0, 45.0],
        }
    )

    scored = add_microstructure_alpha_stack(frame, window=2, depth_col="top_depth")

    assert scored["toxic_pressure_resonance"].iloc[-1] > scored["toxic_pressure_resonance"].iloc[0]
    assert scored["phase_shift_alpha"].iloc[-1] > 0.0
    assert scored["microstructure_alpha_score"].iloc[-1] > 0.0


def test_microstructure_alpha_regime_summary_concentrates_alpha() -> None:
    scored = pd.DataFrame(
        {
            "pressure_memory_decay_state": ["fast_decay", "fast_decay", "persistent"],
            "microstructure_alpha_score": [2.0, 1.0, 1.0],
            "toxic_pressure_resonance": [1.0, 0.8, 0.2],
            "phase_shift_alpha": [0.5, 0.2, 0.0],
            "resiliency_adjusted_alpha": [1.0, 0.7, 0.1],
        }
    )

    summary = microstructure_alpha_regime_summary(scored)
    fast = summary.set_index("pressure_memory_decay_state").loc["fast_decay"]

    assert fast["alpha_share"] == pytest.approx(0.75)
    assert fast["mean_phase_shift_alpha"] == pytest.approx(0.35)


def test_alpha_research_gate_blocks_toxic_selected_regime() -> None:
    summary = pd.DataFrame(
        {
            "pressure_memory_decay_state": ["fast_decay", "persistent"],
            "alpha_share": [0.70, 0.30],
            "mean_microstructure_alpha_score": [3.0, 1.0],
            "mean_phase_shift_alpha": [1.5, 0.1],
        }
    )

    gate = alpha_research_gate(summary, min_alpha_share=0.20, max_phase_shift_alpha=1.0)

    assert gate["alpha_gate"] == "review"
    assert gate["selected_regime"] == "fast_decay"
    assert gate["investable"] is False


def test_alpha_research_gate_rejects_bad_alpha_share() -> None:
    summary = pd.DataFrame(
        {
            "pressure_memory_decay_state": ["fast_decay"],
            "alpha_share": [1.2],
            "mean_microstructure_alpha_score": [1.0],
            "mean_phase_shift_alpha": [0.0],
        }
    )

    with pytest.raises(ValueError, match="alpha_share"):
        alpha_research_gate(summary)
