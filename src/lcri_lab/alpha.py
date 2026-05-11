from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_microstructure_alpha_stack(
    frame: pd.DataFrame,
    *,
    window: int = 20,
    signal_col: str = "pressure_memory",
    fracture_col: str = "latent_liquidity_fracture",
    return_col: str = "gross_return_ticks",
    spread_col: str = "spread_ticks",
    depth_col: str | None = None,
) -> pd.DataFrame:
    """Add path-dependent microstructure alpha diagnostics.

    The stack treats LCRI as a residual pressure process instead of a single
    cross-sectional score. It combines four effects that matter for short-horizon
    order-book alpha research:

    - toxic pressure resonance: same-signed pressure that survives while latent
      liquidity fracture is elevated;
    - resiliency-adjusted alpha: pressure that remains predictive after charging
      spread and depth-fracture penalties;
    - phase-shift alpha: pressure that flips into adverse-selection returns;
    - crowding exhaustion: high pressure whose incremental slope stalls.
    """
    _validate_window(window)
    required = [signal_col, fracture_col, return_col, spread_col]
    if depth_col is not None:
        required.append(depth_col)
    _require_columns(frame, required, label="microstructure alpha")

    output = frame.copy()
    signal = output[signal_col].astype(float)
    fracture = output[fracture_col].astype(float)
    returns = output[return_col].astype(float)
    spread = output[spread_col].astype(float)
    arrays = [signal, fracture, returns, spread]
    if depth_col is None:
        depth_penalty = pd.Series(1.0, index=output.index, dtype=float)
    else:
        depth = output[depth_col].astype(float)
        arrays.append(depth)
        depth_penalty = 1.0 / (1.0 + _safe_zscore(depth.rolling(window, min_periods=2).std().fillna(0.0)))
    _require_finite(arrays, label="microstructure alpha inputs")

    signed_alignment = np.sign(signal).replace(0.0, np.nan) * np.sign(returns).replace(0.0, np.nan)
    adverse_phase = signed_alignment.lt(0.0).fillna(False).astype(float)
    pressure_slope = signal.diff().fillna(0.0)
    pressure_accel = pressure_slope.diff().fillna(0.0)
    resonance = signal.ewm(span=window, adjust=False, min_periods=1).mean() * fracture.abs()
    resonance_persistence = resonance.abs().rolling(window, min_periods=2).mean().fillna(resonance.abs())
    spread_cost = spread.clip(lower=1.0) / spread.clip(lower=1.0).rolling(window, min_periods=1).median()

    output["toxic_pressure_resonance"] = resonance.astype(float)
    output["resonance_persistence"] = resonance_persistence.astype(float)
    output["phase_shift_alpha"] = (adverse_phase * signal.abs() * fracture.abs()).astype(float)
    output["resiliency_adjusted_alpha"] = (
        signal * (1.0 + fracture.abs()) * depth_penalty / (1.0 + spread_cost)
    ).astype(float)
    output["crowding_exhaustion"] = (
        signal.abs().rolling(window, min_periods=2).mean().fillna(0.0)
        * pressure_slope.abs().rolling(window, min_periods=2).mean().fillna(0.0)
        / (1.0 + pressure_accel.abs().rolling(window, min_periods=2).mean().fillna(0.0))
    ).astype(float)
    output["microstructure_alpha_score"] = (
        output["resiliency_adjusted_alpha"].abs()
        + output["toxic_pressure_resonance"].abs()
        + output["phase_shift_alpha"]
        - output["crowding_exhaustion"].clip(lower=0.0)
    ).clip(lower=0.0).astype(float)
    return output


def microstructure_alpha_regime_summary(
    frame: pd.DataFrame,
    *,
    regime_col: str = "pressure_memory_decay_state",
    score_col: str = "microstructure_alpha_score",
    resonance_col: str = "toxic_pressure_resonance",
    phase_col: str = "phase_shift_alpha",
    adjusted_col: str = "resiliency_adjusted_alpha",
) -> pd.DataFrame:
    """Summarize alpha concentration by pressure-memory regime."""
    required = [regime_col, score_col, resonance_col, phase_col, adjusted_col]
    _require_columns(frame, required, label="microstructure alpha regime summary")
    data = frame[required].copy()
    for column in [score_col, resonance_col, phase_col, adjusted_col]:
        data[column] = data[column].astype(float)
    _require_finite([data[score_col], data[resonance_col], data[phase_col], data[adjusted_col]], label="alpha summary inputs")

    total_score = float(data[score_col].clip(lower=0.0).sum())
    rows = []
    for regime, group in data.groupby(regime_col, sort=True):
        positive_score = float(group[score_col].clip(lower=0.0).sum())
        rows.append(
            {
                "pressure_memory_decay_state": regime,
                "observations": int(len(group)),
                "alpha_share": _safe_divide(positive_score, total_score),
                "mean_microstructure_alpha_score": _finite_mean(group[score_col]),
                "mean_toxic_pressure_resonance": _finite_mean(group[resonance_col]),
                "mean_phase_shift_alpha": _finite_mean(group[phase_col]),
                "mean_resiliency_adjusted_alpha": _finite_mean(group[adjusted_col]),
            }
        )
    return pd.DataFrame(rows)


def alpha_research_gate(
    summary: pd.DataFrame,
    *,
    min_alpha_share: float = 0.20,
    max_phase_shift_alpha: float = 1.00,
) -> dict[str, float | int | str | bool]:
    """Gate whether the strongest alpha pocket is investable or toxic."""
    if not np.isfinite(min_alpha_share) or not 0.0 <= min_alpha_share <= 1.0:
        raise ValueError("min_alpha_share must be finite and between 0 and 1")
    if not np.isfinite(max_phase_shift_alpha) or max_phase_shift_alpha < 0.0:
        raise ValueError("max_phase_shift_alpha must be finite and non-negative")
    required = [
        "pressure_memory_decay_state",
        "alpha_share",
        "mean_microstructure_alpha_score",
        "mean_phase_shift_alpha",
    ]
    _require_columns(summary, required, label="alpha research gate")
    if summary.empty:
        return {
            "alpha_gate": "block",
            "selected_regime": "none",
            "selected_alpha_share": 0.0,
            "selected_phase_shift_alpha": 0.0,
            "review_rows": 0,
            "investable": False,
        }

    data = summary.copy()
    for column in ["alpha_share", "mean_microstructure_alpha_score", "mean_phase_shift_alpha"]:
        data[column] = data[column].astype(float)
    _require_finite(
        [data["alpha_share"], data["mean_microstructure_alpha_score"], data["mean_phase_shift_alpha"]],
        label="alpha gate inputs",
    )
    if not data["alpha_share"].between(0.0, 1.0).all():
        raise ValueError("alpha_share must be between 0 and 1")

    ranked = data.sort_values(
        ["alpha_share", "mean_microstructure_alpha_score"], ascending=[False, False]
    ).reset_index(drop=True)
    selected = ranked.iloc[0]
    toxic = selected["mean_phase_shift_alpha"] > max_phase_shift_alpha
    concentrated = selected["alpha_share"] >= min_alpha_share
    gate = "pass" if concentrated and not toxic else "review" if concentrated else "block"
    return {
        "alpha_gate": gate,
        "selected_regime": str(selected["pressure_memory_decay_state"]),
        "selected_alpha_share": float(selected["alpha_share"]),
        "selected_phase_shift_alpha": float(selected["mean_phase_shift_alpha"]),
        "review_rows": int((data["mean_phase_shift_alpha"] > max_phase_shift_alpha).sum()),
        "investable": bool(gate == "pass"),
    }


def alpha_toxicity_review_table(
    summary: pd.DataFrame,
    *,
    min_alpha_share: float = 0.20,
    max_phase_shift_alpha: float = 1.00,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """Rank alpha regimes that need toxicity review before release.

    The table separates investable concentration from toxic concentration and
    points reviewers to pressure-memory regimes where alpha looks like adverse
    selection rather than usable signal.
    """
    _validate_alpha_review_thresholds(min_alpha_share, max_phase_shift_alpha, min_score)
    required = [
        "pressure_memory_decay_state",
        "observations",
        "alpha_share",
        "mean_microstructure_alpha_score",
        "mean_phase_shift_alpha",
        "mean_toxic_pressure_resonance",
    ]
    _require_columns(summary, required, label="alpha toxicity review")
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "pressure_memory_decay_state",
                "observations",
                "alpha_share",
                "mean_microstructure_alpha_score",
                "mean_phase_shift_alpha",
                "mean_toxic_pressure_resonance",
                "toxicity_score",
                "review_label",
            ]
        )

    data = summary[required].copy()
    numeric_columns = [
        "observations",
        "alpha_share",
        "mean_microstructure_alpha_score",
        "mean_phase_shift_alpha",
        "mean_toxic_pressure_resonance",
    ]
    for column in numeric_columns:
        data[column] = data[column].astype(float)
    _require_finite([data[column] for column in numeric_columns], label="alpha toxicity inputs")
    if not data["alpha_share"].between(0.0, 1.0).all():
        raise ValueError("alpha_share must be between 0 and 1")
    if (data["observations"] < 0.0).any():
        raise ValueError("observations must be non-negative")

    concentration_excess = (data["alpha_share"] - min_alpha_share).clip(lower=0.0)
    phase_excess = (data["mean_phase_shift_alpha"] - max_phase_shift_alpha).clip(lower=0.0)
    score_excess = (data["mean_microstructure_alpha_score"] - min_score).clip(lower=0.0)
    data["toxicity_score"] = (
        concentration_excess
        * (1.0 + phase_excess)
        * (1.0 + data["mean_toxic_pressure_resonance"].abs())
        * (1.0 + score_excess)
    ).astype(float)
    data["review_label"] = np.select(
        [
            (concentration_excess > 0.0) & (phase_excess > 0.0),
            concentration_excess > 0.0,
            phase_excess > 0.0,
        ],
        ["toxic_concentration", "concentrated_alpha", "phase_shift_watch"],
        default="clear",
    )
    data["observations"] = data["observations"].astype(int)
    return data.sort_values(
        ["toxicity_score", "alpha_share", "mean_phase_shift_alpha"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def alpha_event_window_diagnostics(
    frame: pd.DataFrame,
    *,
    event_col: str = "phase_shift_alpha",
    score_col: str = "microstructure_alpha_score",
    return_col: str = "gross_return_ticks",
    regime_col: str | None = None,
    window: int = 3,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Measure pre/post return drift around alpha-toxicity events.

    Events are rows where ``event_col`` exceeds ``threshold``. The diagnostic
    keeps a compact, deterministic event table so researchers can inspect
    whether alpha pockets are preceded by pressure buildup, followed by adverse
    returns, or both.
    """
    _validate_window(window)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    required = [event_col, score_col, return_col]
    if regime_col is not None:
        required.append(regime_col)
    _require_columns(frame, required, label="alpha event window")
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "event_index",
                "event_value",
                "event_score",
                "event_regime",
                "pre_return_sum",
                "post_return_sum",
                "post_minus_pre_return",
                "window_rows",
            ]
        )

    data = frame[required].copy()
    numeric_columns = [event_col, score_col, return_col]
    for column in numeric_columns:
        data[column] = data[column].astype(float)
    _require_finite([data[column] for column in numeric_columns], label="alpha event window inputs")

    returns = data[return_col].reset_index(drop=True)
    events = data.index[data[event_col] > threshold].tolist()
    rows = []
    for event_index in events:
        position = data.index.get_loc(event_index)
        pre_start = max(0, position - window)
        post_end = min(len(data), position + window + 1)
        pre = returns.iloc[pre_start:position]
        post = returns.iloc[position + 1 : post_end]
        pre_sum = float(pre.sum()) if len(pre) else 0.0
        post_sum = float(post.sum()) if len(post) else 0.0
        rows.append(
            {
                "event_index": event_index,
                "event_value": float(data.at[event_index, event_col]),
                "event_score": float(data.at[event_index, score_col]),
                "event_regime": str(data.at[event_index, regime_col]) if regime_col is not None else "all",
                "pre_return_sum": pre_sum,
                "post_return_sum": post_sum,
                "post_minus_pre_return": float(post_sum - pre_sum),
                "window_rows": int(len(pre) + len(post) + 1),
            }
        )
    return pd.DataFrame(rows)


def alpha_event_regime_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate alpha event-window drift by event regime."""
    columns = [
        "event_regime",
        "events",
        "adverse_post_drift_events",
        "adverse_post_drift_share",
        "mean_post_minus_pre_return",
        "worst_post_minus_pre_return",
        "mean_event_score",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    required = ["event_regime", "event_score", "post_minus_pre_return"]
    _require_columns(events, required, label="alpha event regime summary")
    data = events[required].copy()
    for column in ["event_score", "post_minus_pre_return"]:
        data[column] = data[column].astype(float)
    _require_finite(
        [data["event_score"], data["post_minus_pre_return"]],
        label="alpha event regime summary inputs",
    )

    rows = []
    for regime, group in data.groupby("event_regime", sort=True):
        drift = group["post_minus_pre_return"]
        adverse = int((drift < 0.0).sum())
        rows.append(
            {
                "event_regime": str(regime),
                "events": int(len(group)),
                "adverse_post_drift_events": adverse,
                "adverse_post_drift_share": _safe_divide(float(adverse), float(len(group))),
                "mean_post_minus_pre_return": _finite_mean(drift),
                "worst_post_minus_pre_return": float(drift.min()),
                "mean_event_score": _finite_mean(group["event_score"]),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values(
        ["adverse_post_drift_share", "worst_post_minus_pre_return"],
        ascending=[False, True],
    ).reset_index(drop=True)


def alpha_event_window_summary(events: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize alpha event-window drift for release review."""
    if events.empty:
        return {
            "events": 0,
            "adverse_post_drift_events": 0,
            "adverse_post_drift_share": 0.0,
            "mean_post_minus_pre_return": 0.0,
            "worst_event_index": "none",
            "worst_post_minus_pre_return": 0.0,
            "max_event_score": 0.0,
        }
    required = [
        "event_index",
        "event_score",
        "post_minus_pre_return",
    ]
    _require_columns(events, required, label="alpha event window summary")
    data = events[required].copy()
    for column in ["event_score", "post_minus_pre_return"]:
        data[column] = data[column].astype(float)
    _require_finite(
        [data["event_score"], data["post_minus_pre_return"]],
        label="alpha event window summary inputs",
    )

    drift = data["post_minus_pre_return"]
    worst = data.loc[drift.idxmin()]
    return {
        "events": int(len(data)),
        "adverse_post_drift_events": int((drift < 0.0).sum()),
        "adverse_post_drift_share": _safe_divide(float((drift < 0.0).sum()), float(len(data))),
        "mean_post_minus_pre_return": _finite_mean(drift),
        "worst_event_index": str(worst["event_index"]),
        "worst_post_minus_pre_return": float(worst["post_minus_pre_return"]),
        "max_event_score": float(data["event_score"].max()),
    }


def alpha_event_score_weighted_drift(events: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize event drift weighted by alpha-event score.

    The unweighted event gate treats every threshold crossing equally. This
    companion diagnostic asks whether the largest alpha events carry the
    adverse drift, which is the failure mode that matters most for release
    review when low-score events are noisy but high-score events are toxic.
    """
    if events.empty:
        return {
            "events": 0,
            "total_event_score": 0.0,
            "score_weighted_post_minus_pre_return": 0.0,
            "score_weighted_adverse_share": 0.0,
            "top_weighted_event_index": "none",
            "top_weighted_adverse_drift": 0.0,
        }
    required = ["event_index", "event_score", "post_minus_pre_return"]
    _require_columns(events, required, label="alpha event score-weighted drift")
    data = events[required].copy()
    for column in ["event_score", "post_minus_pre_return"]:
        data[column] = data[column].astype(float)
    _require_finite(
        [data["event_score"], data["post_minus_pre_return"]],
        label="alpha event score-weighted drift inputs",
    )
    if (data["event_score"] < 0.0).any():
        raise ValueError("event_score must be non-negative")

    weights = data["event_score"]
    drift = data["post_minus_pre_return"]
    total_weight = float(weights.sum())
    adverse_weight = float(weights[drift < 0.0].sum())
    weighted_drift = _safe_divide(float((weights * drift).sum()), total_weight)
    adverse_drift = (-drift).clip(lower=0.0)
    weighted_adverse_drift = weights * adverse_drift
    top_position = weighted_adverse_drift.idxmax()
    top = data.loc[top_position]
    return {
        "events": int(len(data)),
        "total_event_score": total_weight,
        "score_weighted_post_minus_pre_return": weighted_drift,
        "score_weighted_adverse_share": _safe_divide(adverse_weight, total_weight),
        "top_weighted_event_index": str(top["event_index"]),
        "top_weighted_adverse_drift": float(weighted_adverse_drift.loc[top_position]),
    }


def alpha_event_drift_gate(
    summary: dict[str, float | int | str],
    *,
    max_adverse_share: float = 0.50,
    min_mean_post_minus_pre_return: float = 0.0,
    max_worst_post_minus_pre_return: float = -2.0,
) -> dict[str, float | int | str | bool]:
    """Gate alpha event windows by adverse post-event drift.

    This converts compact event-window summaries into a release decision. A
    concentrated alpha pocket can look strong in cross-section while still
    becoming adverse immediately after the event. The gate blocks that failure
    mode when adverse drift is common, mean post-minus-pre drift is negative, or
    the worst event breaches the tail threshold.
    """
    _validate_alpha_event_drift_thresholds(
        max_adverse_share,
        min_mean_post_minus_pre_return,
        max_worst_post_minus_pre_return,
    )
    required = {
        "events",
        "adverse_post_drift_share",
        "mean_post_minus_pre_return",
        "worst_event_index",
        "worst_post_minus_pre_return",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"missing alpha event drift summary keys: {missing}")

    events = int(summary["events"])
    adverse_share = float(summary["adverse_post_drift_share"])
    mean_drift = float(summary["mean_post_minus_pre_return"])
    worst_drift = float(summary["worst_post_minus_pre_return"])
    if events < 0:
        raise ValueError("events must be non-negative")
    if not np.isfinite([adverse_share, mean_drift, worst_drift]).all():
        raise ValueError("alpha event drift summary values must be finite")
    if not 0.0 <= adverse_share <= 1.0:
        raise ValueError("adverse_post_drift_share must be between 0 and 1")

    if events == 0:
        decision = "pass"
        reason = "no alpha events crossed the event threshold"
    elif adverse_share > max_adverse_share:
        decision = "block"
        reason = "adverse post-event drift share breached threshold"
    elif mean_drift < min_mean_post_minus_pre_return:
        decision = "block"
        reason = "mean post-event drift breached threshold"
    elif worst_drift < max_worst_post_minus_pre_return:
        decision = "review"
        reason = "worst post-event drift breached review threshold"
    else:
        decision = "pass"
        reason = "alpha event drift stayed within release thresholds"

    return {
        "passes": decision == "pass",
        "decision": decision,
        "events": events,
        "adverse_post_drift_share": adverse_share,
        "mean_post_minus_pre_return": mean_drift,
        "worst_event_index": str(summary["worst_event_index"]),
        "worst_post_minus_pre_return": worst_drift,
        "max_adverse_share": float(max_adverse_share),
        "min_mean_post_minus_pre_return": float(min_mean_post_minus_pre_return),
        "max_worst_post_minus_pre_return": float(max_worst_post_minus_pre_return),
        "reason": reason,
    }


def alpha_event_release_review_packet(
    drift_gate: dict[str, float | int | str | bool],
    weighted_drift: dict[str, float | int | str],
    regime_summary: pd.DataFrame | None = None,
    *,
    max_score_weighted_adverse_share: float = 0.50,
) -> pd.DataFrame:
    """Build a one-row release-review packet for alpha event drift.

    The packet joins the deterministic drift gate with score-weighted drift and
    the worst event regime so a release reviewer can see whether the decision is
    driven by frequent drift, high-score toxic drift, or localized regime stress.
    """
    if not np.isfinite(max_score_weighted_adverse_share) or not 0.0 <= max_score_weighted_adverse_share <= 1.0:
        raise ValueError("max_score_weighted_adverse_share must be finite and between 0 and 1")
    missing_gate = sorted({"decision", "passes", "events", "reason", "adverse_post_drift_share"} - set(drift_gate))
    if missing_gate:
        raise ValueError(f"missing alpha event drift gate keys: {missing_gate}")
    missing_weighted = sorted(
        {"score_weighted_post_minus_pre_return", "score_weighted_adverse_share", "top_weighted_event_index"}
        - set(weighted_drift)
    )
    if missing_weighted:
        raise ValueError(f"missing alpha event score-weighted drift keys: {missing_weighted}")

    decision = str(drift_gate["decision"])
    weighted_adverse_share = float(weighted_drift["score_weighted_adverse_share"])
    if not np.isfinite(weighted_adverse_share) or not 0.0 <= weighted_adverse_share <= 1.0:
        raise ValueError("score_weighted_adverse_share must be finite and between 0 and 1")
    weighted_review = weighted_adverse_share > max_score_weighted_adverse_share
    events = int(drift_gate["events"])
    priority = 3 if decision == "block" else 2 if decision == "review" or weighted_review else 1 if events else 0
    worst_regime = "none"
    if regime_summary is not None and not regime_summary.empty:
        _require_columns(
            regime_summary,
            ["event_regime", "adverse_post_drift_share", "worst_post_minus_pre_return"],
            label="alpha event regime release review",
        )
        regimes = regime_summary[
            ["event_regime", "adverse_post_drift_share", "worst_post_minus_pre_return"]
        ].copy()
        for column in ["adverse_post_drift_share", "worst_post_minus_pre_return"]:
            regimes[column] = regimes[column].astype(float)
        _require_finite(
            [regimes["adverse_post_drift_share"], regimes["worst_post_minus_pre_return"]],
            label="alpha event regime release review inputs",
        )
        worst = regimes.sort_values(
            ["adverse_post_drift_share", "worst_post_minus_pre_return"], ascending=[False, True]
        ).iloc[0]
        worst_regime = str(worst["event_regime"])
    if decision == "block":
        note = f"release blocked by alpha event drift; inspect {worst_regime} regime"
    elif decision == "review":
        note = f"owner review needed for alpha event drift; inspect {worst_regime} regime"
    elif weighted_review:
        note = f"owner review needed because high-score alpha events carry adverse drift; inspect {worst_regime} regime"
    else:
        note = "alpha event drift evidence is release-aligned"

    return pd.DataFrame(
        [
            {
                "decision": "review" if decision == "pass" and weighted_review else decision,
                "passes": bool(drift_gate["passes"]) and not weighted_review,
                "review_priority": priority,
                "events": events,
                "adverse_post_drift_share": float(drift_gate["adverse_post_drift_share"]),
                "score_weighted_adverse_share": weighted_adverse_share,
                "score_weighted_post_minus_pre_return": float(weighted_drift["score_weighted_post_minus_pre_return"]),
                "top_weighted_event_index": str(weighted_drift["top_weighted_event_index"]),
                "worst_event_regime": worst_regime,
                "release_note": note,
                "gate_reason": str(drift_gate["reason"]),
            }
        ]
    )


def alpha_toxicity_review_summary(review_table: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize the highest-priority alpha toxicity review row."""
    if review_table.empty:
        return {
            "rows": 0,
            "review_rows": 0,
            "top_regime": "none",
            "top_review_label": "clear",
            "top_toxicity_score": 0.0,
        }
    _require_columns(
        review_table,
        ["pressure_memory_decay_state", "review_label", "toxicity_score"],
        label="alpha toxicity review summary",
    )
    scores = review_table["toxicity_score"].astype(float)
    _require_finite([scores], label="alpha toxicity review summary inputs")
    top = review_table.loc[scores.idxmax()]
    labels = review_table["review_label"].astype(str)
    return {
        "rows": int(len(review_table)),
        "review_rows": int((labels != "clear").sum()),
        "top_regime": str(top["pressure_memory_decay_state"]),
        "top_review_label": str(top["review_label"]),
        "top_toxicity_score": float(top["toxicity_score"]),
    }


def _validate_window(window: int) -> None:
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValueError("window must be an integer")
    if window < 2:
        raise ValueError("window must be at least 2")


def _validate_alpha_review_thresholds(
    min_alpha_share: float,
    max_phase_shift_alpha: float,
    min_score: float,
) -> None:
    if not np.isfinite(min_alpha_share) or not 0.0 <= min_alpha_share <= 1.0:
        raise ValueError("min_alpha_share must be finite and between 0 and 1")
    if not np.isfinite(max_phase_shift_alpha) or max_phase_shift_alpha < 0.0:
        raise ValueError("max_phase_shift_alpha must be finite and non-negative")
    if not np.isfinite(min_score) or min_score < 0.0:
        raise ValueError("min_score must be finite and non-negative")


def _validate_alpha_event_drift_thresholds(
    max_adverse_share: float,
    min_mean_post_minus_pre_return: float,
    max_worst_post_minus_pre_return: float,
) -> None:
    if not np.isfinite(max_adverse_share) or not 0.0 <= max_adverse_share <= 1.0:
        raise ValueError("max_adverse_share must be finite and between 0 and 1")
    if not np.isfinite(min_mean_post_minus_pre_return):
        raise ValueError("min_mean_post_minus_pre_return must be finite")
    if not np.isfinite(max_worst_post_minus_pre_return):
        raise ValueError("max_worst_post_minus_pre_return must be finite")


def _require_columns(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _require_finite(series: list[pd.Series], *, label: str) -> None:
    if not np.isfinite(np.column_stack([item.to_numpy(dtype=float) for item in series])).all():
        raise ValueError(f"{label} must be finite")


def _safe_zscore(value: pd.Series) -> pd.Series:
    scale = value.rolling(max(2, min(20, len(value))), min_periods=2).std().replace(0.0, np.nan)
    zscore = (value - value.rolling(max(2, min(20, len(value))), min_periods=1).mean()) / scale
    return zscore.replace([math.inf, -math.inf], np.nan).fillna(0.0).abs()


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _finite_mean(value: pd.Series) -> float:
    mean = float(value.mean()) if len(value) else 0.0
    return mean if math.isfinite(mean) else 0.0
