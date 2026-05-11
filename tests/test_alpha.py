import pandas as pd
import pytest

from lcri_lab.alpha import (
    add_microstructure_alpha_stack,
    alpha_event_regime_summary,
    alpha_event_window_diagnostics,
    alpha_event_window_summary,
    alpha_research_gate,
    alpha_toxicity_review_summary,
    alpha_toxicity_review_table,
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


def test_alpha_toxicity_review_table_ranks_toxic_concentration() -> None:
    summary = pd.DataFrame(
        {
            "pressure_memory_decay_state": ["persistent", "fast_decay", "neutral"],
            "observations": [20, 12, 5],
            "alpha_share": [0.55, 0.30, 0.15],
            "mean_microstructure_alpha_score": [2.0, 1.1, 0.2],
            "mean_phase_shift_alpha": [1.4, 0.4, 1.3],
            "mean_toxic_pressure_resonance": [0.8, 0.2, 0.1],
        }
    )

    review = alpha_toxicity_review_table(
        summary,
        min_alpha_share=0.20,
        max_phase_shift_alpha=1.0,
        min_score=0.5,
    )

    assert review.iloc[0]["pressure_memory_decay_state"] == "persistent"
    assert review.iloc[0]["review_label"] == "toxic_concentration"
    assert review.iloc[-1]["review_label"] == "phase_shift_watch"
    assert review["toxicity_score"].is_monotonic_decreasing


def test_alpha_toxicity_review_summary_reports_top_regime() -> None:
    review = pd.DataFrame(
        {
            "pressure_memory_decay_state": ["persistent", "fast_decay"],
            "review_label": ["toxic_concentration", "clear"],
            "toxicity_score": [1.5, 0.0],
        }
    )

    summary = alpha_toxicity_review_summary(review)

    assert summary == {
        "rows": 2,
        "review_rows": 1,
        "top_regime": "persistent",
        "top_review_label": "toxic_concentration",
        "top_toxicity_score": 1.5,
    }


def test_alpha_event_window_diagnostics_measures_post_event_drift() -> None:
    frame = pd.DataFrame(
        {
            "phase_shift_alpha": [0.0, 0.2, 1.4, 0.0, 0.3, 1.1],
            "microstructure_alpha_score": [0.1, 0.4, 2.0, 0.6, 0.5, 1.7],
            "gross_return_ticks": [1.0, 1.0, -2.0, -1.0, -1.0, -3.0],
            "pressure_memory_decay_state": ["calm", "calm", "toxic", "toxic", "calm", "toxic"],
        }
    )

    diagnostics = alpha_event_window_diagnostics(
        frame,
        regime_col="pressure_memory_decay_state",
        window=2,
        threshold=1.0,
    )

    assert diagnostics["event_index"].tolist() == [2, 5]
    assert diagnostics["event_regime"].tolist() == ["toxic", "toxic"]
    first = diagnostics.iloc[0]
    assert first["pre_return_sum"] == pytest.approx(2.0)
    assert first["post_return_sum"] == pytest.approx(-2.0)
    assert first["post_minus_pre_return"] == pytest.approx(-4.0)
    assert first["window_rows"] == 5


def test_alpha_event_window_diagnostics_rejects_nonfinite_threshold() -> None:
    frame = pd.DataFrame(
        {
            "phase_shift_alpha": [0.0],
            "microstructure_alpha_score": [1.0],
            "gross_return_ticks": [0.0],
        }
    )

    with pytest.raises(ValueError, match="threshold"):
        alpha_event_window_diagnostics(frame, threshold=float("nan"))


def test_alpha_event_regime_summary_ranks_adverse_regimes() -> None:
    events = pd.DataFrame(
        {
            "event_regime": ["toxic", "toxic", "calm"],
            "event_score": [2.0, 1.0, 0.5],
            "post_minus_pre_return": [-3.0, -1.0, 2.0],
        }
    )

    summary = alpha_event_regime_summary(events)

    assert summary.iloc[0]["event_regime"] == "toxic"
    assert summary.iloc[0]["events"] == 2
    assert summary.iloc[0]["adverse_post_drift_share"] == pytest.approx(1.0)
    assert summary.iloc[0]["mean_post_minus_pre_return"] == pytest.approx(-2.0)
    assert summary.iloc[0]["mean_event_score"] == pytest.approx(1.5)


def test_alpha_event_window_summary_surfaces_adverse_drift() -> None:
    events = pd.DataFrame(
        {
            "event_index": ["t2", "t5", "t8"],
            "event_score": [2.0, 1.7, 0.9],
            "post_minus_pre_return": [-4.0, 1.0, -2.0],
        }
    )

    summary = alpha_event_window_summary(events)

    assert summary == {
        "events": 3,
        "adverse_post_drift_events": 2,
        "adverse_post_drift_share": pytest.approx(2.0 / 3.0),
        "mean_post_minus_pre_return": pytest.approx(-5.0 / 3.0),
        "worst_event_index": "t2",
        "worst_post_minus_pre_return": -4.0,
        "max_event_score": 2.0,
    }


def test_alpha_event_window_summary_handles_empty_events() -> None:
    summary = alpha_event_window_summary(pd.DataFrame())

    assert summary["events"] == 0
    assert summary["worst_event_index"] == "none"
    assert summary["adverse_post_drift_share"] == 0.0
    assert summary["max_event_score"] == 0.0
