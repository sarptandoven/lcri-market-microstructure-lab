from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_signals(frame: pd.DataFrame, signals: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    signals = signals or ["raw_imbalance", "lcri"]
    _require_columns(frame, [*signals, "future_direction"])
    rows = []
    for signal in signals:
        score = frame[signal].to_numpy(dtype=float)
        target = frame["future_direction"].to_numpy(dtype=float)
        probability = _logistic(_standardize(score))
        rows.append(
            {
                "signal": signal,
                "directional_accuracy": _directional_accuracy(score, target),
                "brier_score": float(np.mean((probability - target) ** 2)),
                "rank_correlation": _spearman(score, target),
                "mean_abs_score": float(np.mean(np.abs(score))),
            }
        )
    return pd.DataFrame(rows)


def compare_transmission_signal(frame: pd.DataFrame) -> dict[str, float]:
    """Compare LCRI and transmission pressure as directional signals."""
    _require_columns(frame, ["lcri", "transmission_pressure", "future_direction"])
    metrics = evaluate_signals(frame, signals=["lcri", "transmission_pressure"]).set_index("signal")
    lcri = metrics.loc["lcri"]
    transmission = metrics.loc["transmission_pressure"]
    return {
        "directional_accuracy_delta": float(
            transmission["directional_accuracy"] - lcri["directional_accuracy"]
        ),
        "brier_score_delta": float(transmission["brier_score"] - lcri["brier_score"]),
        "rank_correlation_delta": float(
            transmission["rank_correlation"] - lcri["rank_correlation"]
        ),
    }


def absorption_regime_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate LCRI and transmission pressure inside each absorption regime."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    _require_columns(
        frame,
        ["absorption_regime", "lcri", "transmission_pressure", "future_direction"],
    )

    rows = []
    for regime, group in frame.groupby("absorption_regime", sort=True):
        metrics = evaluate_signals(group, signals=["lcri", "transmission_pressure"])
        for row in metrics.to_dict("records"):
            row["absorption_regime"] = regime
            row["rows"] = len(group)
            rows.append(row)
    return pd.DataFrame(rows)[
        [
            "absorption_regime",
            "signal",
            "rows",
            "directional_accuracy",
            "brier_score",
            "rank_correlation",
            "mean_abs_score",
        ]
    ]


def generalization_gap_leaderboard(
    signal_gap: pd.DataFrame,
    regime_gap: pd.DataFrame,
    transition_gap: pd.DataFrame,
    *,
    limit: int = 10,
) -> pd.DataFrame:
    """Rank the largest directional accuracy gaps across generalization tables."""
    rows = [
        *_gap_rows(signal_gap, scope="signal", context_column=None),
        *_gap_rows(regime_gap, scope="regime", context_column="regime"),
        *_gap_rows(transition_gap, scope="transition", context_column="segment"),
    ]
    if not rows:
        return pd.DataFrame(columns=["scope", "context", "signal", "directional_accuracy_gap"])
    return (
        pd.DataFrame(rows)
        .sort_values("directional_accuracy_gap", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def lcri_generalization_gap_leaderboard(
    signal_gap: pd.DataFrame,
    regime_gap: pd.DataFrame,
    transition_gap: pd.DataFrame,
    *,
    limit: int = 10,
) -> pd.DataFrame:
    """Rank LCRI-only directional accuracy gaps across generalization tables."""
    leaderboard = generalization_gap_leaderboard(
        signal_gap,
        regime_gap,
        transition_gap,
        limit=max(limit * 3, limit),
    )
    if leaderboard.empty:
        return leaderboard
    return leaderboard.loc[leaderboard["signal"] == "lcri"].head(limit).reset_index(drop=True)


def lcri_generalization_scope_summary(lcri_leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Summarize LCRI generalization gaps by scope."""
    if lcri_leaderboard.empty:
        return pd.DataFrame(columns=["scope", "rows", "mean_directional_accuracy_gap", "max_directional_accuracy_gap"])
    _require_columns(lcri_leaderboard, ["scope", "directional_accuracy_gap"])
    return (
        lcri_leaderboard.groupby("scope", sort=True)["directional_accuracy_gap"]
        .agg(
            rows="count",
            mean_directional_accuracy_gap="mean",
            max_directional_accuracy_gap="max",
        )
        .reset_index()
    )


def lcri_worst_generalization_context(lcri_leaderboard: pd.DataFrame) -> dict[str, float | str]:
    """Return the LCRI context with the largest directional accuracy gap."""
    if lcri_leaderboard.empty:
        return {
            "scope": "none",
            "context": "none",
            "directional_accuracy_gap": 0.0,
        }
    _require_columns(lcri_leaderboard, ["scope", "context", "directional_accuracy_gap"])
    row = lcri_leaderboard.loc[lcri_leaderboard["directional_accuracy_gap"].astype(float).idxmax()]
    return {
        "scope": str(row["scope"]),
        "context": str(row["context"]),
        "directional_accuracy_gap": float(row["directional_accuracy_gap"]),
    }


def lcri_generalization_severity(
    lcri_leaderboard: pd.DataFrame,
    *,
    warning_gap: float = 0.02,
    critical_gap: float = 0.05,
) -> pd.DataFrame:
    """Attach severity labels to LCRI generalization gap rows."""
    if warning_gap < 0.0 or critical_gap < warning_gap:
        raise ValueError("severity thresholds must be non-negative and ordered")
    if lcri_leaderboard.empty:
        return pd.DataFrame(columns=[*lcri_leaderboard.columns, "severity"])
    _require_columns(lcri_leaderboard, ["directional_accuracy_gap"])

    output = lcri_leaderboard.copy()
    gaps = output["directional_accuracy_gap"].astype(float)
    output["severity"] = np.select(
        [gaps >= critical_gap, gaps >= warning_gap],
        ["critical", "warning"],
        default="stable",
    )
    return output


def lcri_generalization_severity_summary(severity: pd.DataFrame) -> dict[str, bool | int]:
    """Count LCRI generalization severity labels for report gating."""
    if severity.empty:
        return {
            "rows": 0,
            "stable_rows": 0,
            "warning_rows": 0,
            "critical_rows": 0,
            "passes_lcri_generalization_gate": True,
        }
    _require_columns(severity, ["severity"])

    counts = severity["severity"].value_counts()
    critical_rows = int(counts.get("critical", 0))
    return {
        "rows": len(severity),
        "stable_rows": int(counts.get("stable", 0)),
        "warning_rows": int(counts.get("warning", 0)),
        "critical_rows": critical_rows,
        "passes_lcri_generalization_gate": bool(critical_rows == 0),
    }


def lcri_generalization_critical_contexts(severity: pd.DataFrame) -> pd.DataFrame:
    """Return critical LCRI generalization rows ordered by largest gap."""
    if severity.empty:
        return pd.DataFrame(columns=list(severity.columns))
    _require_columns(severity, ["severity", "directional_accuracy_gap"])

    critical = severity.loc[severity["severity"] == "critical"].copy()
    if critical.empty:
        return critical.reset_index(drop=True)
    return critical.sort_values(
        "directional_accuracy_gap",
        ascending=False,
    ).reset_index(drop=True)


def lcri_generalization_severity_by_scope(severity: pd.DataFrame) -> pd.DataFrame:
    """Count LCRI severity labels within each generalization scope."""
    columns = ["scope", "rows", "stable_rows", "warning_rows", "critical_rows"]
    if severity.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(severity, ["scope", "severity"])

    counts = (
        severity.groupby(["scope", "severity"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ["stable", "warning", "critical"]:
        if label not in counts.columns:
            counts[label] = 0
    counts["rows"] = counts[["stable", "warning", "critical"]].sum(axis=1)
    return counts.rename(
        columns={
            "stable": "stable_rows",
            "warning": "warning_rows",
            "critical": "critical_rows",
        }
    )[columns]


def lcri_generalization_scope_risk(severity_by_scope: pd.DataFrame) -> pd.DataFrame:
    """Convert severity scope counts into warning and critical risk rates."""
    columns = ["scope", "rows", "warning_or_critical_share", "critical_share"]
    if severity_by_scope.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        severity_by_scope,
        ["scope", "rows", "warning_rows", "critical_rows"],
    )

    output = severity_by_scope.copy()
    rows = output["rows"].astype(float).replace(0.0, np.nan)
    output["warning_or_critical_share"] = (
        (output["warning_rows"].astype(float) + output["critical_rows"].astype(float))
        / rows
    ).fillna(0.0)
    output["critical_share"] = (output["critical_rows"].astype(float) / rows).fillna(0.0)
    return output[columns]


def lcri_generalization_blocker_summary(critical_contexts: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize critical LCRI gate blockers for concise release notes."""
    if critical_contexts.empty:
        return {
            "critical_rows": 0,
            "critical_scopes": "none",
            "max_critical_gap": 0.0,
            "max_critical_context": "none",
        }
    _require_columns(critical_contexts, ["scope", "context", "directional_accuracy_gap"])

    gaps = critical_contexts["directional_accuracy_gap"].astype(float)
    worst = critical_contexts.loc[gaps.idxmax()]
    scopes = sorted({str(scope) for scope in critical_contexts["scope"]})
    return {
        "critical_rows": len(critical_contexts),
        "critical_scopes": ",".join(scopes),
        "max_critical_gap": float(gaps.max()),
        "max_critical_context": f"{worst['scope']}:{worst['context']}",
    }


def lcri_generalization_scope_gate_decisions(scope_risk: pd.DataFrame) -> pd.DataFrame:
    """Assign pass/warn/block decisions to each LCRI generalization scope."""
    columns = ["scope", "rows", "decision", "reason"]
    if scope_risk.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(scope_risk, ["scope", "rows", "warning_or_critical_share", "critical_share"])

    output = scope_risk.copy()
    output["decision"] = np.select(
        [output["critical_share"].astype(float) > 0.0, output["warning_or_critical_share"].astype(float) > 0.0],
        ["block", "warn"],
        default="pass",
    )
    output["reason"] = [
        _lcri_scope_gate_reason(row) for row in output.to_dict("records")
    ]
    return output[columns]


def lcri_generalization_gate_decision(
    severity_summary: dict[str, bool | int],
    worst_context: dict[str, float | str],
) -> dict[str, bool | float | int | str]:
    """Build a compact LCRI generalization gate decision payload."""
    required_summary = {
        "rows",
        "warning_rows",
        "critical_rows",
        "passes_lcri_generalization_gate",
    }
    missing_summary = sorted(required_summary - set(severity_summary))
    if missing_summary:
        raise ValueError(f"incomplete severity summary: {missing_summary}")
    _require_mapping_keys(
        worst_context,
        ["scope", "context", "directional_accuracy_gap"],
        label="worst context",
    )

    passes = bool(severity_summary["passes_lcri_generalization_gate"])
    critical_rows = int(severity_summary["critical_rows"])
    warning_rows = int(severity_summary["warning_rows"])
    return {
        "passes": passes,
        "decision": "pass" if passes else "block",
        "rows_evaluated": int(severity_summary["rows"]),
        "warning_rows": warning_rows,
        "critical_rows": critical_rows,
        "worst_scope": str(worst_context["scope"]),
        "worst_context": str(worst_context["context"]),
        "worst_directional_accuracy_gap": float(worst_context["directional_accuracy_gap"]),
        "reason": _lcri_gate_reason(passes, warning_rows, critical_rows, worst_context),
    }


def generalization_overview(
    signal_gap: pd.DataFrame,
    regime_gap: pd.DataFrame,
    transition_gap: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarize generated generalization gap tables for quick audit checks."""
    return {
        "signal_rows": len(signal_gap),
        "regime_rows": len(regime_gap),
        "transition_rows": len(transition_gap),
        "max_signal_directional_accuracy_gap": _max_gap(signal_gap),
        "max_regime_directional_accuracy_gap": _max_gap(regime_gap),
        "max_transition_directional_accuracy_gap": _max_gap(transition_gap),
    }


def lcri_generalization_gap_delta(
    signal_gap: pd.DataFrame,
    regime_gap: pd.DataFrame,
    transition_gap: pd.DataFrame,
) -> pd.DataFrame:
    """Compare LCRI gap stability against raw imbalance across all gap tables."""
    rows = [
        *_gap_delta_rows(signal_gap, scope="signal", context_column=None),
        *_gap_delta_rows(regime_gap, scope="regime", context_column="regime"),
        *_gap_delta_rows(transition_gap, scope="transition", context_column="segment"),
    ]
    if not rows:
        return pd.DataFrame(
            columns=[
                "scope",
                "context",
                "raw_imbalance_directional_accuracy_gap",
                "lcri_directional_accuracy_gap",
                "raw_minus_lcri_directional_accuracy_gap",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        "raw_minus_lcri_directional_accuracy_gap",
        ascending=False,
    ).reset_index(drop=True)


def lcri_gap_delta_summary(gap_delta: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize where LCRI generalizes better or worse than raw imbalance."""
    column = "raw_minus_lcri_directional_accuracy_gap"
    if gap_delta.empty:
        return {
            "rows": 0,
            "lcri_more_stable_rows": 0,
            "lcri_less_stable_rows": 0,
            "lcri_equal_stability_rows": 0,
            "max_lcri_stability_edge": 0.0,
            "max_lcri_stability_edge_context": "none",
            "max_lcri_instability_edge": 0.0,
            "max_lcri_instability_edge_context": "none",
        }
    _require_columns(gap_delta, ["scope", "context", column])

    values = gap_delta[column].astype(float)
    best = gap_delta.loc[values.idxmax()]
    worst = gap_delta.loc[values.idxmin()]
    return {
        "rows": len(gap_delta),
        "lcri_more_stable_rows": int((values > 0.0).sum()),
        "lcri_less_stable_rows": int((values < 0.0).sum()),
        "lcri_equal_stability_rows": int((values == 0.0).sum()),
        "max_lcri_stability_edge": float(values.max()),
        "max_lcri_stability_edge_context": f"{best['scope']}:{best['context']}",
        "max_lcri_instability_edge": float(values.min()),
        "max_lcri_instability_edge_context": f"{worst['scope']}:{worst['context']}",
    }


def lcri_gap_delta_improvements(gap_delta: pd.DataFrame) -> pd.DataFrame:
    """Return scopes where LCRI generalizes better than raw imbalance."""
    column = "raw_minus_lcri_directional_accuracy_gap"
    if gap_delta.empty:
        return pd.DataFrame(columns=list(gap_delta.columns))
    _require_columns(gap_delta, [column])

    improvements = gap_delta.loc[gap_delta[column].astype(float) > 0.0].copy()
    if improvements.empty:
        return improvements.reset_index(drop=True)
    return improvements.sort_values(column, ascending=False).reset_index(drop=True)


def lcri_gap_delta_regressions(gap_delta: pd.DataFrame) -> pd.DataFrame:
    """Return scopes where LCRI generalizes worse than raw imbalance."""
    column = "raw_minus_lcri_directional_accuracy_gap"
    if gap_delta.empty:
        return pd.DataFrame(columns=list(gap_delta.columns))
    _require_columns(gap_delta, [column])

    regressions = gap_delta.loc[gap_delta[column].astype(float) < 0.0].copy()
    if regressions.empty:
        return regressions.reset_index(drop=True)
    return regressions.sort_values(column, ascending=True).reset_index(drop=True)


def lcri_gap_delta_scope_summary(gap_delta: pd.DataFrame) -> pd.DataFrame:
    """Summarize LCRI-vs-raw stability deltas by generalization scope."""
    column = "raw_minus_lcri_directional_accuracy_gap"
    columns = [
        "scope",
        "rows",
        "mean_raw_minus_lcri_gap",
        "min_raw_minus_lcri_gap",
        "max_raw_minus_lcri_gap",
    ]
    if gap_delta.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(gap_delta, ["scope", column])

    return (
        gap_delta.groupby("scope", sort=True)[column]
        .agg(
            rows="count",
            mean_raw_minus_lcri_gap="mean",
            min_raw_minus_lcri_gap="min",
            max_raw_minus_lcri_gap="max",
        )
        .reset_index()[columns]
    )


def lcri_gap_delta_scorecard(gap_delta: pd.DataFrame) -> dict[str, float | int]:
    """Score how often LCRI reduces generalization gaps versus raw imbalance."""
    column = "raw_minus_lcri_directional_accuracy_gap"
    if gap_delta.empty:
        return {
            "rows": 0,
            "mean_raw_minus_lcri_directional_accuracy_gap": 0.0,
            "median_raw_minus_lcri_directional_accuracy_gap": 0.0,
            "lcri_more_stable_share": 0.0,
            "lcri_less_stable_share": 0.0,
        }
    _require_columns(gap_delta, [column])

    values = gap_delta[column].astype(float)
    rows = len(values)
    return {
        "rows": rows,
        "mean_raw_minus_lcri_directional_accuracy_gap": float(values.mean()),
        "median_raw_minus_lcri_directional_accuracy_gap": float(values.median()),
        "lcri_more_stable_share": float((values > 0.0).sum() / rows),
        "lcri_less_stable_share": float((values < 0.0).sum() / rows),
    }


def lcri_gap_delta_flags(gap_delta: pd.DataFrame) -> pd.DataFrame:
    """Attach categorical stability flags to LCRI gap delta rows."""
    column = "raw_minus_lcri_directional_accuracy_gap"
    if gap_delta.empty:
        return pd.DataFrame(columns=[*gap_delta.columns, "stability_flag"])
    _require_columns(gap_delta, ["scope", "context", column])

    output = gap_delta.copy()
    values = output[column].astype(float)
    output["stability_flag"] = np.select(
        [values > 0.0, values < 0.0],
        ["lcri_more_stable", "lcri_less_stable"],
        default="lcri_equal_stability",
    )
    return output


def regime_generalization_gap(metrics: pd.DataFrame, heldout_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare full-sample and heldout metrics by regime and signal."""
    required = ["regime", "signal", "directional_accuracy", "brier_score", "rank_correlation"]
    _require_columns(metrics, required)
    _require_columns(heldout_metrics, required)

    full = metrics.set_index(["regime", "signal"])
    heldout = heldout_metrics.set_index(["regime", "signal"])
    keys = [key for key in full.index if key in heldout.index]
    rows = []
    for regime, signal in keys:
        rows.append(
            {
                "regime": regime,
                "signal": signal,
                "directional_accuracy_gap": float(
                    full.loc[(regime, signal), "directional_accuracy"]
                    - heldout.loc[(regime, signal), "directional_accuracy"]
                ),
                "brier_score_gap": float(
                    heldout.loc[(regime, signal), "brier_score"]
                    - full.loc[(regime, signal), "brier_score"]
                ),
                "rank_correlation_gap": float(
                    full.loc[(regime, signal), "rank_correlation"]
                    - heldout.loc[(regime, signal), "rank_correlation"]
                ),
            }
        )
    return pd.DataFrame(rows)


def regime_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    _require_columns(frame, ["regime", "raw_imbalance", "lcri", "future_direction"])
    rows = []
    for regime, group in frame.groupby("regime", sort=True):
        metrics = evaluate_signals(group)
        for row in metrics.to_dict("records"):
            row["regime"] = regime
            row["rows"] = len(group)
            rows.append(row)
    return pd.DataFrame(rows)[
        [
            "regime",
            "signal",
            "rows",
            "directional_accuracy",
            "brier_score",
            "rank_correlation",
            "mean_abs_score",
        ]
    ]


def signal_generalization_gap(metrics: pd.DataFrame, heldout_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare full-sample and heldout signal metrics by signal."""
    required = ["signal", "directional_accuracy", "brier_score", "rank_correlation"]
    _require_columns(metrics, required)
    _require_columns(heldout_metrics, required)

    full = metrics.set_index("signal")
    heldout = heldout_metrics.set_index("signal")
    signals = [signal for signal in full.index if signal in heldout.index]
    rows = []
    for signal in signals:
        rows.append(
            {
                "signal": signal,
                "directional_accuracy_gap": float(
                    full.loc[signal, "directional_accuracy"]
                    - heldout.loc[signal, "directional_accuracy"]
                ),
                "brier_score_gap": float(
                    heldout.loc[signal, "brier_score"] - full.loc[signal, "brier_score"]
                ),
                "rank_correlation_gap": float(
                    full.loc[signal, "rank_correlation"]
                    - heldout.loc[signal, "rank_correlation"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_signal_lift(frame: pd.DataFrame) -> dict[str, float]:
    metrics = evaluate_signals(frame).set_index("signal")
    raw = metrics.loc["raw_imbalance"]
    lcri = metrics.loc["lcri"]
    return {
        "directional_accuracy_lift": float(
            lcri["directional_accuracy"] - raw["directional_accuracy"]
        ),
        "brier_score_reduction": float(raw["brier_score"] - lcri["brier_score"]),
        "rank_correlation_lift": float(lcri["rank_correlation"] - raw["rank_correlation"]),
    }


def transition_generalization_gap(
    metrics: pd.DataFrame,
    heldout_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare full-sample and heldout metrics by transition segment and signal."""
    required = ["segment", "signal", "directional_accuracy", "brier_score", "rank_correlation"]
    _require_columns(metrics, required)
    _require_columns(heldout_metrics, required)

    full = metrics.set_index(["segment", "signal"])
    heldout = heldout_metrics.set_index(["segment", "signal"])
    rows = []
    for segment, signal in [key for key in full.index if key in heldout.index]:
        rows.append(
            {
                "segment": segment,
                "signal": signal,
                "directional_accuracy_gap": float(
                    full.loc[(segment, signal), "directional_accuracy"]
                    - heldout.loc[(segment, signal), "directional_accuracy"]
                ),
                "brier_score_gap": float(
                    heldout.loc[(segment, signal), "brier_score"]
                    - full.loc[(segment, signal), "brier_score"]
                ),
                "rank_correlation_gap": float(
                    full.loc[(segment, signal), "rank_correlation"]
                    - heldout.loc[(segment, signal), "rank_correlation"]
                ),
            }
        )
    return pd.DataFrame(rows)


def transition_signal_lift(
    frame: pd.DataFrame,
    *,
    transition_col: str = "regime_changed",
) -> pd.DataFrame:
    """Summarize LCRI lift over raw imbalance during stable and transition periods."""
    metrics = transition_conditioned_metrics(frame, transition_col=transition_col)
    rows = []
    for segment, group in metrics.groupby("segment", sort=True):
        by_signal = group.set_index("signal")
        raw = by_signal.loc["raw_imbalance"]
        lcri = by_signal.loc["lcri"]
        rows.append(
            {
                "segment": segment,
                "rows": int(lcri["rows"]),
                "directional_accuracy_lift": float(
                    lcri["directional_accuracy"] - raw["directional_accuracy"]
                ),
                "brier_score_reduction": float(raw["brier_score"] - lcri["brier_score"]),
                "rank_correlation_lift": float(
                    lcri["rank_correlation"] - raw["rank_correlation"]
                ),
            }
        )
    return pd.DataFrame(rows)


def transition_robustness_summary(
    frame: pd.DataFrame,
    *,
    transition_col: str = "regime_changed",
    min_accuracy_lift: float = 0.0,
) -> dict[str, float | bool]:
    """Summarize whether LCRI lift survives both stable and transition periods."""
    lift = transition_signal_lift(frame, transition_col=transition_col).set_index("segment")
    stable = lift.loc["stable"] if "stable" in lift.index else None
    transition = lift.loc["transition"] if "transition" in lift.index else None

    stable_accuracy_lift = (
        float(stable["directional_accuracy_lift"]) if stable is not None else 0.0
    )
    transition_accuracy_lift = (
        float(transition["directional_accuracy_lift"]) if transition is not None else 0.0
    )
    stable_rows = int(stable["rows"]) if stable is not None else 0
    transition_rows = int(transition["rows"]) if transition is not None else 0

    return {
        "stable_rows": stable_rows,
        "transition_rows": transition_rows,
        "stable_directional_accuracy_lift": stable_accuracy_lift,
        "transition_directional_accuracy_lift": transition_accuracy_lift,
        "minimum_directional_accuracy_lift": min(
            stable_accuracy_lift,
            transition_accuracy_lift,
        ),
        "transition_to_stable_lift_ratio": _safe_scalar_divide(
            transition_accuracy_lift,
            stable_accuracy_lift,
        ),
        "passes_transition_robustness": bool(
            stable_rows > 0
            and transition_rows > 0
            and stable_accuracy_lift >= min_accuracy_lift
            and transition_accuracy_lift >= min_accuracy_lift
        ),
    }


def transition_conditioned_metrics(
    frame: pd.DataFrame,
    signals: list[str] | None = None,
    *,
    transition_col: str = "regime_changed",
) -> pd.DataFrame:
    """Evaluate signals separately around stable and transitioning liquidity states."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    signals = signals or ["raw_imbalance", "lcri"]
    _require_columns(frame, [*signals, "future_direction", transition_col])

    transition = frame[transition_col].to_numpy(dtype=float) > 0.0
    segments = [
        ("stable", ~transition),
        ("transition", transition),
    ]

    rows = []
    for segment, mask in segments:
        if not np.any(mask):
            continue
        metrics = evaluate_signals(frame.loc[mask], signals=signals)
        for row in metrics.to_dict("records"):
            row["segment"] = segment
            row["rows"] = int(np.sum(mask))
            rows.append(row)

    if not rows:
        raise ValueError("transition-conditioned evaluation has no rows")
    return pd.DataFrame(rows)[
        [
            "segment",
            "signal",
            "rows",
            "directional_accuracy",
            "brier_score",
            "rank_correlation",
            "mean_abs_score",
        ]
    ]


def feature_stability_report(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Report finite rates and distribution stability for selected features."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    if not columns:
        raise ValueError("columns must be non-empty")
    _require_columns(frame, ["regime", *columns])

    rows = []
    for regime, group in frame.groupby("regime", sort=True):
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            finite = np.isfinite(values)
            finite_values = values[finite]
            rows.append(
                {
                    "regime": regime,
                    "feature": column,
                    "rows": len(group),
                    "finite_rate": float(np.mean(finite)),
                    "mean": float(np.mean(finite_values)) if len(finite_values) else 0.0,
                    "std": float(np.std(finite_values)) if len(finite_values) else 0.0,
                    "p05": float(np.quantile(finite_values, 0.05)) if len(finite_values) else 0.0,
                    "p95": float(np.quantile(finite_values, 0.95)) if len(finite_values) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def lcri_tail_diagnostics(
    frame: pd.DataFrame,
    thresholds: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Summarize future behavior in positive and negative LCRI tails."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    _require_columns(frame, ["lcri", "future_direction"])

    future_ticks = None
    if "future_return_ticks" in frame.columns:
        future_ticks = frame["future_return_ticks"].to_numpy(dtype=float)
    score = frame["lcri"].to_numpy(dtype=float)
    target = frame["future_direction"].to_numpy(dtype=float)

    rows = []
    for threshold in thresholds:
        if threshold <= 0.0:
            raise ValueError("thresholds must be positive")
        for side, mask, expected_direction in [
            ("positive", score >= threshold, 1.0),
            ("negative", score <= -threshold, 0.0),
        ]:
            count = int(np.sum(mask))
            row = {
                "threshold": float(threshold),
                "side": side,
                "rows": count,
                "hit_rate": 0.0,
                "mean_future_return_ticks": 0.0,
            }
            if count:
                row["hit_rate"] = float(np.mean(target[mask] == expected_direction))
                if future_ticks is not None:
                    row["mean_future_return_ticks"] = float(np.mean(future_ticks[mask]))
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_cost_aware_signals(
    frame: pd.DataFrame,
    signals: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate signals only where transaction-cost labels choose a side."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    signals = signals or ["raw_imbalance", "lcri"]
    _require_columns(frame, [*signals, "cost_aware_direction"])

    tradable = frame["cost_aware_direction"].to_numpy(dtype=float) != -1.0
    if not np.any(tradable):
        raise ValueError("cost-aware evaluation has no tradable rows")

    rows = []
    target = frame.loc[tradable, "cost_aware_direction"].to_numpy(dtype=float)
    for signal in signals:
        score = frame.loc[tradable, signal].to_numpy(dtype=float)
        probability = _logistic(_standardize(score))
        rows.append(
            {
                "signal": signal,
                "rows": int(np.sum(tradable)),
                "abstained_rows": int(len(frame) - np.sum(tradable)),
                "directional_accuracy": _directional_accuracy(score, target),
                "brier_score": float(np.mean((probability - target) ** 2)),
                "rank_correlation": _spearman(score, target),
            }
        )
    return pd.DataFrame(rows)


def calibration_curve(frame: pd.DataFrame, signal: str, bins: int = 10) -> pd.DataFrame:
    if bins < 1:
        raise ValueError("bins must be at least 1")
    _require_columns(frame, [signal, "future_direction"])
    probability = _logistic(frame[signal].to_numpy(dtype=float))
    target = frame["future_direction"].to_numpy(dtype=float)
    bucket = np.clip(np.floor(probability * bins).astype(int), 0, bins - 1)
    rows = []
    for idx in range(bins):
        mask = bucket == idx
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": idx,
                "predicted_probability": float(np.mean(probability[mask])),
                "observed_frequency": float(np.mean(target[mask])),
                "rows": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


def _gap_rows(
    frame: pd.DataFrame,
    *,
    scope: str,
    context_column: str | None,
) -> list[dict[str, float | str]]:
    if frame.empty or "directional_accuracy_gap" not in frame.columns:
        return []

    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "scope": scope,
                "context": row[context_column] if context_column else "all",
                "signal": row["signal"],
                "directional_accuracy_gap": float(row["directional_accuracy_gap"]),
            }
        )
    return rows


def _gap_delta_rows(
    frame: pd.DataFrame,
    *,
    scope: str,
    context_column: str | None,
) -> list[dict[str, float | str]]:
    if frame.empty:
        return []
    _require_columns(frame, ["signal", "directional_accuracy_gap"])
    index_columns = [context_column] if context_column else []
    indexed = frame.set_index([*index_columns, "signal"])
    contexts = sorted(set(indexed.index.get_level_values(0))) if context_column else ["all"]

    rows = []
    for context in contexts:
        raw_key = (context, "raw_imbalance") if context_column else "raw_imbalance"
        lcri_key = (context, "lcri") if context_column else "lcri"
        if raw_key not in indexed.index or lcri_key not in indexed.index:
            continue
        raw_gap = float(indexed.loc[raw_key, "directional_accuracy_gap"])
        lcri_gap = float(indexed.loc[lcri_key, "directional_accuracy_gap"])
        rows.append(
            {
                "scope": scope,
                "context": context,
                "raw_imbalance_directional_accuracy_gap": raw_gap,
                "lcri_directional_accuracy_gap": lcri_gap,
                "raw_minus_lcri_directional_accuracy_gap": raw_gap - lcri_gap,
            }
        )
    return rows


def _max_gap(frame: pd.DataFrame) -> float:
    if frame.empty or "directional_accuracy_gap" not in frame.columns:
        return 0.0
    return float(frame["directional_accuracy_gap"].max())


def _require_mapping_keys(payload: dict[str, object], keys: list[str], *, label: str) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        raise ValueError(f"incomplete {label}: {missing}")


def _lcri_scope_gate_reason(row: dict[str, object]) -> str:
    scope = row["scope"]
    critical_share = float(row["critical_share"])
    warning_share = float(row["warning_or_critical_share"])
    if critical_share > 0.0:
        return f"{scope} blocked with {critical_share:.2%} critical LCRI rows"
    if warning_share > 0.0:
        return f"{scope} warned with {warning_share:.2%} warning-or-critical LCRI rows"
    return f"{scope} passed with no warning or critical LCRI rows"


def _lcri_gate_reason(
    passes: bool,
    warning_rows: int,
    critical_rows: int,
    worst_context: dict[str, float | str],
) -> str:
    scope = worst_context["scope"]
    context = worst_context["context"]
    gap = float(worst_context["directional_accuracy_gap"])
    if not passes:
        return (
            f"blocked by {critical_rows} critical LCRI generalization rows; "
            f"worst gap is {gap:.4f} in {scope}:{context}"
        )
    if warning_rows:
        return (
            f"passed with {warning_rows} warning LCRI generalization rows; "
            f"worst gap is {gap:.4f} in {scope}:{context}"
        )
    return f"passed with no warning or critical LCRI generalization rows; worst gap is {gap:.4f}"


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing evaluation columns: {missing}")


def _directional_accuracy(score: np.ndarray, target: np.ndarray) -> float:
    prediction = (score > 0.0).astype(float)
    return float(np.mean(prediction == target))


def _standardize(score: np.ndarray) -> np.ndarray:
    scale = float(np.std(score))
    if scale == 0.0:
        return score
    return score / scale


def _logistic(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(score, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _safe_scalar_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def _spearman(score: np.ndarray, target: np.ndarray) -> float:
    score_rank = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    target_rank = pd.Series(target).rank(method="average").to_numpy(dtype=float)
    if np.std(score_rank) == 0.0 or np.std(target_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(score_rank, target_rank)[0, 1])
