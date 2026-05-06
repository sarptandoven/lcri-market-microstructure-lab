from __future__ import annotations

from math import erf, sqrt

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
        if not np.isfinite(score).all() or not np.isfinite(target).all():
            raise ValueError("signal evaluation inputs must be finite")
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
    _validate_limit(limit)
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
    _validate_limit(limit)
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


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")
    if limit < 1:
        raise ValueError("limit must be at least 1")



def generalization_fragility_diagnostics(
    metrics: pd.DataFrame,
    heldout_metrics: pd.DataFrame,
    regime_metric_table: pd.DataFrame,
    heldout_regime_metric_table: pd.DataFrame,
    transition_metric_table: pd.DataFrame,
    heldout_transition_metric_table: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate whether full-to-heldout accuracy gaps exceed heldout uncertainty.

    The diagnostics use the heldout directional-accuracy binomial standard error
    as a lightweight uncertainty scale for each signal/context row. It is not a
    formal acceptance test, but it makes small heldout slices visible before a
    large gap is overinterpreted.
    """
    rows = [
        *_fragility_rows(metrics, heldout_metrics, scope="signal", context_column=None),
        *_fragility_rows(
            regime_metric_table,
            heldout_regime_metric_table,
            scope="regime",
            context_column="regime",
        ),
        *_fragility_rows(
            transition_metric_table,
            heldout_transition_metric_table,
            scope="transition",
            context_column="segment",
        ),
    ]
    columns = [
        "scope",
        "context",
        "signal",
        "full_rows",
        "heldout_rows",
        "full_directional_accuracy",
        "heldout_directional_accuracy",
        "directional_accuracy_gap",
        "heldout_directional_accuracy_se",
        "abs_gap_to_se_ratio",
        "fragility_label",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values(
        ["abs_gap_to_se_ratio", "heldout_rows"],
        ascending=[False, True],
    ).reset_index(drop=True)


def generalization_fragility_summary(diagnostics: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize heldout fragility labels for release-review dashboards."""
    if diagnostics.empty:
        return {
            "rows": 0,
            "stable_rows": 0,
            "watch_rows": 0,
            "fragile_rows": 0,
            "max_abs_gap_to_se_ratio": 0.0,
            "most_fragile_context": "none",
        }
    _require_columns(diagnostics, ["scope", "context", "signal", "abs_gap_to_se_ratio", "fragility_label"])
    labels = diagnostics["fragility_label"].astype(str)
    ratio = diagnostics["abs_gap_to_se_ratio"].astype(float)
    worst = diagnostics.loc[ratio.idxmax()]
    return {
        "rows": len(diagnostics),
        "stable_rows": int((labels == "stable").sum()),
        "watch_rows": int((labels == "watch").sum()),
        "fragile_rows": int((labels == "fragile").sum()),
        "max_abs_gap_to_se_ratio": float(ratio.max()),
        "most_fragile_context": f"{worst['scope']}:{worst['context']}:{worst['signal']}",
    }


def generalization_stability_confidence_intervals(
    fragility_diagnostics: pd.DataFrame,
    *,
    z_score: float = 1.96,
) -> pd.DataFrame:
    """Attach heldout directional-accuracy confidence intervals to fragility rows.

    Fragility ratios are useful for ranking suspect gaps, but reviewers also need
    the raw heldout interval that produced the uncertainty scale. This artifact
    keeps the interval math next to each full-to-heldout gap so dashboards can
    distinguish narrow, stable slices from wide, sample-limited ones.
    """
    columns = [
        "scope",
        "context",
        "signal",
        "heldout_rows",
        "heldout_directional_accuracy",
        "heldout_directional_accuracy_se",
        "confidence_level",
        "heldout_directional_accuracy_ci_lower",
        "heldout_directional_accuracy_ci_upper",
        "heldout_directional_accuracy_ci_width",
        "directional_accuracy_gap",
        "gap_exceeds_ci_half_width",
    ]
    if not np.isfinite(z_score) or z_score <= 0.0:
        raise ValueError("z_score must be finite and positive")
    if fragility_diagnostics.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        fragility_diagnostics,
        [
            "scope",
            "context",
            "signal",
            "heldout_rows",
            "heldout_directional_accuracy",
            "heldout_directional_accuracy_se",
            "directional_accuracy_gap",
        ],
    )

    output = fragility_diagnostics.copy()
    heldout_accuracy = output["heldout_directional_accuracy"].astype(float)
    se = output["heldout_directional_accuracy_se"].astype(float)
    half_width = z_score * se
    lower = (heldout_accuracy - half_width).clip(lower=0.0)
    upper = (heldout_accuracy + half_width).clip(upper=1.0)
    output["confidence_level"] = float(_normal_confidence_level(z_score))
    output["heldout_directional_accuracy_ci_lower"] = lower
    output["heldout_directional_accuracy_ci_upper"] = upper
    output["heldout_directional_accuracy_ci_width"] = upper - lower
    output["gap_exceeds_ci_half_width"] = output["directional_accuracy_gap"].astype(float).abs() > half_width
    return output[columns].sort_values(
        ["gap_exceeds_ci_half_width", "heldout_directional_accuracy_ci_width", "heldout_rows"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def generalization_stability_confidence_summary(
    intervals: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize heldout confidence interval coverage for release review."""
    if intervals.empty:
        return {
            "rows": 0,
            "gap_exceeds_ci_half_width_rows": 0,
            "mean_ci_width": 0.0,
            "max_ci_width": 0.0,
            "widest_interval_context": "none",
        }
    _require_columns(
        intervals,
        [
            "scope",
            "context",
            "signal",
            "heldout_directional_accuracy_ci_width",
            "gap_exceeds_ci_half_width",
        ],
    )
    widths = intervals["heldout_directional_accuracy_ci_width"].astype(float)
    widest = intervals.loc[widths.idxmax()]
    return {
        "rows": len(intervals),
        "gap_exceeds_ci_half_width_rows": int(intervals["gap_exceeds_ci_half_width"].astype(bool).sum()),
        "mean_ci_width": float(widths.mean()),
        "max_ci_width": float(widths.max()),
        "widest_interval_context": f"{widest['scope']}:{widest['context']}:{widest['signal']}",
    }


def lcri_ci_gate_contradiction_diagnostics(
    lcri_severity: pd.DataFrame,
    stability_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """Compare deterministic LCRI gate labels with heldout CI evidence.

    The severity gate uses fixed directional-accuracy gap thresholds. The heldout
    CI table asks whether that gap is larger than the heldout sampling half-width.
    This diagnostic makes CI/gate disagreements row-addressable: critical or
    warning gates whose gap sits inside the CI half-width, plus stable rows whose
    gap still exceeds that uncertainty band.
    """
    columns = [
        "scope",
        "context",
        "severity",
        "directional_accuracy_gap",
        "heldout_rows",
        "confidence_level",
        "heldout_directional_accuracy_ci_width",
        "ci_half_width",
        "gap_exceeds_ci_half_width",
        "ci_gate_label",
        "review_priority",
        "review_note",
    ]
    if lcri_severity.empty or stability_intervals.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        lcri_severity,
        ["scope", "context", "signal", "directional_accuracy_gap", "severity"],
    )
    _require_columns(
        stability_intervals,
        [
            "scope",
            "context",
            "signal",
            "heldout_rows",
            "confidence_level",
            "heldout_directional_accuracy_ci_width",
            "gap_exceeds_ci_half_width",
        ],
    )

    severity = lcri_severity.loc[
        lcri_severity["signal"].astype(str) == "lcri",
        ["scope", "context", "directional_accuracy_gap", "severity"],
    ].copy()
    intervals = stability_intervals.loc[
        stability_intervals["signal"].astype(str) == "lcri",
        [
            "scope",
            "context",
            "heldout_rows",
            "confidence_level",
            "heldout_directional_accuracy_ci_width",
            "gap_exceeds_ci_half_width",
        ],
    ].copy()
    output = severity.merge(intervals, on=["scope", "context"], how="inner")
    if output.empty:
        return pd.DataFrame(columns=columns)
    output["ci_half_width"] = output["heldout_directional_accuracy_ci_width"].astype(float) / 2.0
    output["ci_gate_label"] = [
        _ci_gate_contradiction_label(severity_label, exceeds)
        for severity_label, exceeds in zip(output["severity"], output["gap_exceeds_ci_half_width"])
    ]
    output["review_priority"] = [
        _ci_gate_review_priority(label, severity_label, ci_width)
        for label, severity_label, ci_width in zip(
            output["ci_gate_label"],
            output["severity"],
            output["heldout_directional_accuracy_ci_width"],
        )
    ]
    output["review_note"] = [
        _ci_gate_review_note(row) for row in output.to_dict("records")
    ]
    return output[columns].sort_values(
        ["review_priority", "ci_gate_label", "directional_accuracy_gap"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def lcri_ci_gate_contradiction_summary(
    diagnostics: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize CI-versus-gate contradiction diagnostics."""
    if diagnostics.empty:
        return {
            "rows": 0,
            "aligned_rows": 0,
            "contradiction_rows": 0,
            "gate_blocks_inside_ci_rows": 0,
            "gate_warns_inside_ci_rows": 0,
            "stable_gap_outside_ci_rows": 0,
            "max_review_priority": 0,
            "worst_ci_gate_context": "none",
        }
    _require_columns(
        diagnostics,
        [
            "scope",
            "context",
            "ci_gate_label",
            "review_priority",
            "directional_accuracy_gap",
        ],
    )
    labels = diagnostics["ci_gate_label"].astype(str)
    priority = diagnostics["review_priority"].astype(int)
    contradiction_rows = diagnostics.loc[labels != "aligned"]
    if contradiction_rows.empty:
        worst_context = "none"
    else:
        worst = contradiction_rows.sort_values(
            ["review_priority", "directional_accuracy_gap"], ascending=[False, False]
        ).iloc[0]
        worst_context = f"{worst['scope']}:{worst['context']}:{worst['ci_gate_label']}"
    return {
        "rows": len(diagnostics),
        "aligned_rows": int((labels == "aligned").sum()),
        "contradiction_rows": int((labels != "aligned").sum()),
        "gate_blocks_inside_ci_rows": int((labels == "gate_blocks_inside_ci").sum()),
        "gate_warns_inside_ci_rows": int((labels == "gate_warns_inside_ci").sum()),
        "stable_gap_outside_ci_rows": int((labels == "stable_gap_outside_ci").sum()),
        "max_review_priority": int(priority.max()),
        "worst_ci_gate_context": worst_context,
    }


def lcri_ci_confidence_coverage_scorecard(
    stability_intervals: pd.DataFrame,
    ci_gate_diagnostics: pd.DataFrame,
    *,
    wide_ci_threshold: float = 0.20,
) -> pd.DataFrame:
    """Aggregate heldout CI coverage and gate contradictions by review scope.

    Row-level CI/gate diagnostics identify individual disagreements. This
    scorecard makes the owner-facing question explicit: which scopes combine wide
    heldout intervals, gaps outside the interval band, and non-aligned gate
    labels often enough to deserve first review.
    """
    columns = [
        "scope",
        "rows",
        "mean_ci_width",
        "max_ci_width",
        "wide_ci_rows",
        "wide_ci_share",
        "gap_exceeds_ci_half_width_rows",
        "gap_exceeds_ci_half_width_share",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
        "max_ci_gate_review_priority",
        "coverage_label",
        "review_note",
    ]
    if wide_ci_threshold <= 0.0:
        raise ValueError("wide_ci_threshold must be positive")
    if stability_intervals.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        stability_intervals,
        [
            "scope",
            "context",
            "signal",
            "heldout_directional_accuracy_ci_width",
            "gap_exceeds_ci_half_width",
        ],
    )
    _require_columns(
        ci_gate_diagnostics,
        ["scope", "context", "ci_gate_label", "review_priority"],
    )

    intervals = stability_intervals.loc[
        stability_intervals["signal"].astype(str) == "lcri",
        [
            "scope",
            "context",
            "heldout_directional_accuracy_ci_width",
            "gap_exceeds_ci_half_width",
        ],
    ].copy()
    if intervals.empty:
        return pd.DataFrame(columns=columns)
    intervals["heldout_directional_accuracy_ci_width"] = intervals[
        "heldout_directional_accuracy_ci_width"
    ].astype(float)
    intervals["gap_exceeds_ci_half_width"] = intervals["gap_exceeds_ci_half_width"].astype(bool)
    intervals["wide_ci"] = intervals["heldout_directional_accuracy_ci_width"] >= wide_ci_threshold

    diagnostics = ci_gate_diagnostics.loc[
        :, ["scope", "context", "ci_gate_label", "review_priority"]
    ].copy()
    diagnostics["ci_gate_label"] = diagnostics["ci_gate_label"].astype(str)
    diagnostics["review_priority"] = diagnostics["review_priority"].astype(int)

    rows = []
    for scope, scope_intervals in intervals.groupby("scope", dropna=False):
        scope_text = str(scope)
        scope_diagnostics = diagnostics.loc[diagnostics["scope"].astype(str) == scope_text]
        row_count = len(scope_intervals)
        ci_width = scope_intervals["heldout_directional_accuracy_ci_width"].astype(float)
        wide_rows = int(scope_intervals["wide_ci"].sum())
        outside_rows = int(scope_intervals["gap_exceeds_ci_half_width"].sum())
        contradiction_rows = int((scope_diagnostics["ci_gate_label"] != "aligned").sum())
        high_priority_rows = int((scope_diagnostics["review_priority"] >= 3).sum())
        max_priority = int(scope_diagnostics["review_priority"].max()) if not scope_diagnostics.empty else 0
        label = _ci_confidence_coverage_label(
            wide_rows=wide_rows,
            outside_rows=outside_rows,
            contradiction_rows=contradiction_rows,
            high_priority_rows=high_priority_rows,
        )
        rows.append(
            {
                "scope": scope_text,
                "rows": row_count,
                "mean_ci_width": float(ci_width.mean()),
                "max_ci_width": float(ci_width.max()),
                "wide_ci_rows": wide_rows,
                "wide_ci_share": wide_rows / row_count,
                "gap_exceeds_ci_half_width_rows": outside_rows,
                "gap_exceeds_ci_half_width_share": outside_rows / row_count,
                "ci_gate_contradiction_rows": contradiction_rows,
                "high_priority_ci_gate_rows": high_priority_rows,
                "max_ci_gate_review_priority": max_priority,
                "coverage_label": label,
                "review_note": _ci_confidence_coverage_note(label),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        [
            "high_priority_ci_gate_rows",
            "ci_gate_contradiction_rows",
            "wide_ci_share",
            "gap_exceeds_ci_half_width_share",
            "max_ci_width",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def lcri_ci_confidence_coverage_summary(
    scorecard: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize scope-level heldout CI coverage risk."""
    if scorecard.empty:
        return {
            "scopes": 0,
            "review_scopes": 0,
            "blocking_review_scopes": 0,
            "contradiction_review_scopes": 0,
            "wide_ci_review_scopes": 0,
            "total_ci_gate_contradiction_rows": 0,
            "total_wide_ci_rows": 0,
            "worst_ci_confidence_scope": "none",
        }
    _require_columns(
        scorecard,
        [
            "scope",
            "coverage_label",
            "ci_gate_contradiction_rows",
            "wide_ci_rows",
            "high_priority_ci_gate_rows",
            "wide_ci_share",
            "max_ci_width",
        ],
    )
    labels = scorecard["coverage_label"].astype(str)
    review_mask = labels != "adequate_ci_coverage"
    worst = scorecard.loc[review_mask].iloc[0] if review_mask.any() else scorecard.iloc[0]
    return {
        "scopes": len(scorecard),
        "review_scopes": int(review_mask.sum()),
        "blocking_review_scopes": int((labels == "blocking_ci_gate_review").sum()),
        "contradiction_review_scopes": int((labels == "ci_gate_contradiction_review").sum()),
        "wide_ci_review_scopes": int((labels == "wide_ci_review").sum()),
        "total_ci_gate_contradiction_rows": int(scorecard["ci_gate_contradiction_rows"].astype(int).sum()),
        "total_wide_ci_rows": int(scorecard["wide_ci_rows"].astype(int).sum()),
        "worst_ci_confidence_scope": f"{worst['scope']}:{worst['coverage_label']}",
    }


def lcri_fragility_gate_alignment(
    fragility_diagnostics: pd.DataFrame,
    lcri_severity: pd.DataFrame,
) -> pd.DataFrame:
    """Compare LCRI gate severity with heldout uncertainty fragility.

    Severity labels are deterministic full-to-heldout gap thresholds, while the
    fragility diagnostics scale the same gaps by heldout sampling uncertainty.
    This table makes contradictions explicit for reviewers: rows where a gate
    blocks on a statistically stable slice, or rows where a non-blocking gap is
    uncertainty-fragile and should not be overinterpreted.
    """
    columns = [
        "scope",
        "context",
        "directional_accuracy_gap",
        "severity",
        "heldout_rows",
        "heldout_directional_accuracy_se",
        "abs_gap_to_se_ratio",
        "fragility_label",
        "alignment_label",
        "review_note",
    ]
    if fragility_diagnostics.empty or lcri_severity.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        fragility_diagnostics,
        [
            "scope",
            "context",
            "signal",
            "heldout_rows",
            "heldout_directional_accuracy_se",
            "abs_gap_to_se_ratio",
            "fragility_label",
        ],
    )
    _require_columns(lcri_severity, ["scope", "context", "directional_accuracy_gap", "severity"])

    fragility = fragility_diagnostics.loc[
        fragility_diagnostics["signal"].astype(str) == "lcri",
        [
            "scope",
            "context",
            "heldout_rows",
            "heldout_directional_accuracy_se",
            "abs_gap_to_se_ratio",
            "fragility_label",
        ],
    ]
    joined = lcri_severity.merge(fragility, on=["scope", "context"], how="left")
    if joined.empty:
        return pd.DataFrame(columns=columns)
    joined["alignment_label"] = [
        _fragility_gate_alignment_label(severity, fragility)
        for severity, fragility in zip(joined["severity"], joined["fragility_label"])
    ]
    joined["review_note"] = [
        _fragility_gate_review_note(row) for row in joined.to_dict("records")
    ]
    return joined[columns].sort_values(
        ["alignment_label", "abs_gap_to_se_ratio", "directional_accuracy_gap"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def lcri_fragility_gate_scorecard(alignment: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize agreement between deterministic LCRI gates and heldout fragility."""
    if alignment.empty:
        return {
            "rows": 0,
            "aligned_rows": 0,
            "review_required_rows": 0,
            "gate_blocks_stable_slice_rows": 0,
            "uncertainty_fragile_noncritical_rows": 0,
            "uncertainty_watch_stable_gap_rows": 0,
            "critical_rows": 0,
            "critical_stable_slice_share": 0.0,
            "max_abs_gap_to_se_ratio": 0.0,
            "worst_review_context": "none",
        }
    _require_columns(
        alignment,
        ["scope", "context", "severity", "abs_gap_to_se_ratio", "alignment_label"],
    )

    labels = alignment["alignment_label"].astype(str)
    severity = alignment["severity"].astype(str)
    critical = severity == "critical"
    critical_rows = int(critical.sum())
    gate_blocks_stable = labels == "gate_blocks_stable_slice"
    review_required = labels != "aligned"
    ratio = alignment["abs_gap_to_se_ratio"].astype(float)
    review_rows = alignment.loc[review_required].copy()
    if review_rows.empty:
        worst_review_context = "none"
    else:
        review_ratio = review_rows["abs_gap_to_se_ratio"].astype(float)
        worst = review_rows.loc[review_ratio.idxmax()]
        worst_review_context = f"{worst['scope']}:{worst['context']}:{worst['alignment_label']}"

    return {
        "rows": len(alignment),
        "aligned_rows": int((labels == "aligned").sum()),
        "review_required_rows": int(review_required.sum()),
        "gate_blocks_stable_slice_rows": int(gate_blocks_stable.sum()),
        "uncertainty_fragile_noncritical_rows": int((labels == "uncertainty_fragile_noncritical").sum()),
        "uncertainty_watch_stable_gap_rows": int((labels == "uncertainty_watch_stable_gap").sum()),
        "critical_rows": critical_rows,
        "critical_stable_slice_share": float(gate_blocks_stable.sum() / critical_rows) if critical_rows else 0.0,
        "max_abs_gap_to_se_ratio": float(ratio.max()),
        "worst_review_context": worst_review_context,
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




def lcri_scope_gate_decision_summary(scope_decisions: pd.DataFrame) -> dict[str, int | str]:
    """Summarize pass/warn/block decisions across LCRI generalization scopes."""
    if scope_decisions.empty:
        return {
            "scopes": 0,
            "pass_scopes": 0,
            "warn_scopes": 0,
            "block_scopes": 0,
            "blocked_scope_names": "none",
            "warn_scope_names": "none",
        }
    _require_columns(scope_decisions, ["scope", "decision"])

    decisions = scope_decisions["decision"].astype(str)
    blocked = sorted(scope_decisions.loc[decisions == "block", "scope"].astype(str))
    warned = sorted(scope_decisions.loc[decisions == "warn", "scope"].astype(str))
    return {
        "scopes": len(scope_decisions),
        "pass_scopes": int((decisions == "pass").sum()),
        "warn_scopes": int((decisions == "warn").sum()),
        "block_scopes": int((decisions == "block").sum()),
        "blocked_scope_names": ",".join(blocked) if blocked else "none",
        "warn_scope_names": ",".join(warned) if warned else "none",
    }


def lcri_scope_stability_contradictions(
    scope_decisions: pd.DataFrame,
    gap_delta_scope_summary: pd.DataFrame,
    fragility_gate_alignment: pd.DataFrame,
) -> pd.DataFrame:
    """Flag scope-level contradictions between gates, stability deltas, and fragility review rows.

    The LCRI gate is driven by absolute heldout gap severity. The gap-delta
    dashboard is relative to raw imbalance. This audit catches scopes where the
    gate posture and relative-stability story can read as contradictory in a
    report, then attaches fragility-review counts as a second diagnostic lens.
    """
    columns = [
        "scope",
        "decision",
        "rows",
        "lcri_more_stable_share",
        "lcri_less_stable_share",
        "fragility_review_required_rows",
        "contradiction_label",
        "review_note",
    ]
    if scope_decisions.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(scope_decisions, ["scope", "rows", "decision"])
    _require_columns(
        gap_delta_scope_summary,
        ["scope", "lcri_more_stable_share", "lcri_less_stable_share"],
    )
    _require_columns(fragility_gate_alignment, ["scope", "alignment_label"])

    delta = gap_delta_scope_summary[
        ["scope", "lcri_more_stable_share", "lcri_less_stable_share"]
    ].copy()
    review_counts = (
        fragility_gate_alignment.assign(
            fragility_review_required=lambda frame: frame["alignment_label"].astype(str) != "aligned"
        )
        .groupby("scope", sort=True)["fragility_review_required"]
        .sum()
        .reset_index()
        .rename(columns={"fragility_review_required": "fragility_review_required_rows"})
    )
    output = scope_decisions[["scope", "rows", "decision"]].copy()
    output = output.merge(delta, on="scope", how="left")
    output = output.merge(review_counts, on="scope", how="left")
    output[["lcri_more_stable_share", "lcri_less_stable_share"]] = output[
        ["lcri_more_stable_share", "lcri_less_stable_share"]
    ].fillna(0.0)
    output["fragility_review_required_rows"] = (
        output["fragility_review_required_rows"].fillna(0).astype(int)
    )
    output["contradiction_label"] = [
        _lcri_scope_stability_contradiction_label(row) for row in output.to_dict("records")
    ]
    output["review_note"] = [
        _lcri_scope_stability_contradiction_note(row) for row in output.to_dict("records")
    ]
    return output[columns].sort_values(
        ["contradiction_label", "fragility_review_required_rows", "lcri_less_stable_share"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def lcri_scope_stability_contradiction_summary(
    contradictions: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize cross-scope gate-versus-stability contradiction labels."""
    if contradictions.empty:
        return {
            "scopes": 0,
            "aligned_scopes": 0,
            "contradiction_scopes": 0,
            "gate_blocks_despite_relative_stability_scopes": 0,
            "pass_scope_with_relative_regressions_scopes": 0,
            "warning_scope_with_broad_relative_regression_scopes": 0,
            "fragility_review_required_rows": 0,
            "worst_contradiction_scope": "none",
        }
    _require_columns(
        contradictions,
        [
            "scope",
            "contradiction_label",
            "fragility_review_required_rows",
            "lcri_less_stable_share",
        ],
    )

    labels = contradictions["contradiction_label"].astype(str)
    contradiction_rows = contradictions.loc[labels != "aligned"].copy()
    if contradiction_rows.empty:
        worst_scope = "none"
    else:
        worst = contradiction_rows.sort_values(
            ["fragility_review_required_rows", "lcri_less_stable_share"],
            ascending=[False, False],
        ).iloc[0]
        worst_scope = f"{worst['scope']}:{worst['contradiction_label']}"
    return {
        "scopes": len(contradictions),
        "aligned_scopes": int((labels == "aligned").sum()),
        "contradiction_scopes": int((labels != "aligned").sum()),
        "gate_blocks_despite_relative_stability_scopes": int(
            (labels == "gate_blocks_despite_relative_stability").sum()
        ),
        "pass_scope_with_relative_regressions_scopes": int(
            (labels == "pass_scope_with_relative_regressions").sum()
        ),
        "warning_scope_with_broad_relative_regression_scopes": int(
            (labels == "warning_scope_with_broad_relative_regression").sum()
        ),
        "fragility_review_required_rows": int(
            contradictions["fragility_review_required_rows"].astype(int).sum()
        ),
        "worst_contradiction_scope": worst_scope,
    }


def lcri_contradiction_review_packet(
    contradictions: pd.DataFrame,
    lcri_severity: pd.DataFrame,
    gap_delta: pd.DataFrame,
    fragility_gate_alignment: pd.DataFrame,
) -> pd.DataFrame:
    """Build an evidence-linked scope review packet for gate/stability contradictions.

    The scope contradiction table is intentionally compact. This packet attaches
    the worst deterministic gate row, worst relative LCRI-vs-raw delta row, and
    worst fragility/gate disagreement for each scope so reviewers can trace a
    contradictory scope directly back to row-level evidence.
    """
    columns = [
        "scope",
        "contradiction_label",
        "decision",
        "scope_rows",
        "lcri_less_stable_share",
        "fragility_review_required_rows",
        "worst_gate_context",
        "worst_gate_severity",
        "worst_gate_directional_accuracy_gap",
        "worst_delta_context",
        "worst_raw_minus_lcri_directional_accuracy_gap",
        "worst_fragility_context",
        "worst_fragility_alignment_label",
        "worst_fragility_abs_gap_to_se_ratio",
        "review_priority",
    ]
    if contradictions.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        contradictions,
        [
            "scope",
            "contradiction_label",
            "decision",
            "rows",
            "lcri_less_stable_share",
            "fragility_review_required_rows",
        ],
    )
    _require_columns(lcri_severity, ["scope", "context", "directional_accuracy_gap", "severity"])
    _require_columns(gap_delta, ["scope", "context", "raw_minus_lcri_directional_accuracy_gap"])
    _require_columns(
        fragility_gate_alignment,
        ["scope", "context", "alignment_label", "abs_gap_to_se_ratio"],
    )

    rows = []
    for contradiction in contradictions.to_dict("records"):
        scope = str(contradiction["scope"])
        gate = _worst_scope_gate_row(lcri_severity, scope)
        delta = _worst_scope_delta_row(gap_delta, scope)
        fragility = _worst_scope_fragility_row(fragility_gate_alignment, scope)
        priority = _contradiction_review_priority(
            str(contradiction["contradiction_label"]),
            int(contradiction["fragility_review_required_rows"]),
        )
        rows.append(
            {
                "scope": scope,
                "contradiction_label": str(contradiction["contradiction_label"]),
                "decision": str(contradiction["decision"]),
                "scope_rows": int(contradiction["rows"]),
                "lcri_less_stable_share": float(contradiction["lcri_less_stable_share"]),
                "fragility_review_required_rows": int(
                    contradiction["fragility_review_required_rows"]
                ),
                "worst_gate_context": gate["context"],
                "worst_gate_severity": gate["severity"],
                "worst_gate_directional_accuracy_gap": gate["directional_accuracy_gap"],
                "worst_delta_context": delta["context"],
                "worst_raw_minus_lcri_directional_accuracy_gap": delta[
                    "raw_minus_lcri_directional_accuracy_gap"
                ],
                "worst_fragility_context": fragility["context"],
                "worst_fragility_alignment_label": fragility["alignment_label"],
                "worst_fragility_abs_gap_to_se_ratio": fragility["abs_gap_to_se_ratio"],
                "review_priority": priority,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        [
            "review_priority",
            "fragility_review_required_rows",
            "worst_gate_directional_accuracy_gap",
            "lcri_less_stable_share",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def lcri_contradiction_review_packet_summary(
    packet: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize the evidence-linked contradiction packet for release gates."""
    if packet.empty:
        return {
            "scopes": 0,
            "high_priority_scopes": 0,
            "medium_priority_scopes": 0,
            "low_priority_scopes": 0,
            "fragility_review_required_rows": 0,
            "max_review_priority": 0,
            "worst_review_scope": "none",
            "worst_fragility_scope": "none",
        }
    _require_columns(
        packet,
        [
            "scope",
            "contradiction_label",
            "fragility_review_required_rows",
            "worst_fragility_abs_gap_to_se_ratio",
            "review_priority",
        ],
    )
    priority = packet["review_priority"].astype(int)
    worst_review = packet.sort_values(
        ["review_priority", "fragility_review_required_rows", "worst_fragility_abs_gap_to_se_ratio"],
        ascending=[False, False, False],
    ).iloc[0]
    worst_fragility = packet.sort_values(
        ["worst_fragility_abs_gap_to_se_ratio", "review_priority"],
        ascending=[False, False],
    ).iloc[0]
    return {
        "scopes": len(packet),
        "high_priority_scopes": int((priority >= 3).sum()),
        "medium_priority_scopes": int((priority == 2).sum()),
        "low_priority_scopes": int((priority == 1).sum()),
        "fragility_review_required_rows": int(
            packet["fragility_review_required_rows"].astype(int).sum()
        ),
        "max_review_priority": int(priority.max()),
        "worst_review_scope": f"{worst_review['scope']}:{worst_review['contradiction_label']}",
        "worst_fragility_scope": (
            f"{worst_fragility['scope']}:{worst_fragility['worst_fragility_abs_gap_to_se_ratio']:.6f}"
        ),
    }


def lcri_uncertainty_weighted_review_priority(
    review_packet: pd.DataFrame,
    ci_confidence_scorecard: pd.DataFrame,
) -> pd.DataFrame:
    """Rank owner review scopes by contradiction evidence scaled by uncertainty.

    The contradiction packet gives a deterministic review priority, while the CI
    coverage scorecard describes how sample-limited each scope is. This table
    combines both signals so reviewers start with scopes whose apparent gate or
    stability contradiction is also backed by wide or conflicting heldout CI
    evidence.
    """
    columns = [
        "scope",
        "contradiction_label",
        "base_review_priority",
        "fragility_review_required_rows",
        "worst_fragility_abs_gap_to_se_ratio",
        "coverage_label",
        "mean_ci_width",
        "max_ci_width",
        "wide_ci_share",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
        "uncertainty_weighted_priority",
        "priority_label",
        "review_note",
    ]
    if review_packet.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        review_packet,
        [
            "scope",
            "contradiction_label",
            "fragility_review_required_rows",
            "worst_fragility_abs_gap_to_se_ratio",
            "review_priority",
        ],
    )
    _require_columns(
        ci_confidence_scorecard,
        [
            "scope",
            "coverage_label",
            "mean_ci_width",
            "max_ci_width",
            "wide_ci_share",
            "ci_gate_contradiction_rows",
            "high_priority_ci_gate_rows",
        ],
    )

    packet = review_packet.loc[
        :,
        [
            "scope",
            "contradiction_label",
            "fragility_review_required_rows",
            "worst_fragility_abs_gap_to_se_ratio",
            "review_priority",
        ],
    ].copy()
    scorecard = ci_confidence_scorecard.loc[
        :,
        [
            "scope",
            "coverage_label",
            "mean_ci_width",
            "max_ci_width",
            "wide_ci_share",
            "ci_gate_contradiction_rows",
            "high_priority_ci_gate_rows",
        ],
    ].copy()
    output = packet.merge(scorecard, on="scope", how="left")
    output["coverage_label"] = output["coverage_label"].fillna("missing_ci_coverage")
    for column in ["mean_ci_width", "max_ci_width", "wide_ci_share"]:
        output[column] = output[column].fillna(0.0).astype(float)
    for column in ["ci_gate_contradiction_rows", "high_priority_ci_gate_rows"]:
        output[column] = output[column].fillna(0).astype(int)
    output = output.rename(columns={"review_priority": "base_review_priority"})
    output["base_review_priority"] = output["base_review_priority"].astype(int)
    output["fragility_review_required_rows"] = output["fragility_review_required_rows"].astype(int)
    output["worst_fragility_abs_gap_to_se_ratio"] = output[
        "worst_fragility_abs_gap_to_se_ratio"
    ].astype(float)

    output["uncertainty_weighted_priority"] = [
        _uncertainty_weighted_priority(row) for row in output.to_dict("records")
    ]
    output["priority_label"] = [
        _uncertainty_weighted_priority_label(score)
        for score in output["uncertainty_weighted_priority"]
    ]
    output["review_note"] = [
        _uncertainty_weighted_priority_note(row) for row in output.to_dict("records")
    ]
    return output[columns].sort_values(
        [
            "uncertainty_weighted_priority",
            "high_priority_ci_gate_rows",
            "ci_gate_contradiction_rows",
            "worst_fragility_abs_gap_to_se_ratio",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def lcri_uncertainty_weighted_review_priority_summary(
    priorities: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize uncertainty-weighted owner review priority labels."""
    if priorities.empty:
        return {
            "scopes": 0,
            "critical_priority_scopes": 0,
            "high_priority_scopes": 0,
            "medium_priority_scopes": 0,
            "low_priority_scopes": 0,
            "max_uncertainty_weighted_priority": 0.0,
            "worst_uncertainty_weighted_scope": "none",
        }
    _require_columns(
        priorities,
        ["scope", "priority_label", "uncertainty_weighted_priority"],
    )
    labels = priorities["priority_label"].astype(str)
    scores = priorities["uncertainty_weighted_priority"].astype(float)
    worst = priorities.loc[scores.idxmax()]
    return {
        "scopes": len(priorities),
        "critical_priority_scopes": int((labels == "critical").sum()),
        "high_priority_scopes": int((labels == "high").sum()),
        "medium_priority_scopes": int((labels == "medium").sum()),
        "low_priority_scopes": int((labels == "low").sum()),
        "max_uncertainty_weighted_priority": float(scores.max()),
        "worst_uncertainty_weighted_scope": f"{worst['scope']}:{worst['priority_label']}",
    }


def lcri_cross_artifact_evidence_index(
    severity_by_scope: pd.DataFrame,
    scope_gate_decisions: pd.DataFrame,
    gap_delta_scope_summary: pd.DataFrame,
    scope_stability_contradictions: pd.DataFrame,
    ci_confidence_scorecard: pd.DataFrame,
    uncertainty_weighted_priorities: pd.DataFrame,
) -> pd.DataFrame:
    """Build one scope-level index across gate, stability, CI, and owner-review artifacts.

    The report now emits several owner-facing diagnostics. This index keeps the
    release review workflow from becoming a scavenger hunt by joining the core
    scope signals into a single sortable table and assigning a deterministic
    evidence score.
    """
    columns = [
        "scope",
        "gate_decision",
        "severity_rows",
        "warning_rows",
        "critical_rows",
        "lcri_more_stable_share",
        "lcri_less_stable_share",
        "fragility_review_required_rows",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
        "contradiction_label",
        "priority_label",
        "uncertainty_weighted_priority",
        "evidence_score",
        "evidence_label",
        "review_note",
    ]
    if severity_by_scope.empty and scope_gate_decisions.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(severity_by_scope, ["scope", "rows", "warning_rows", "critical_rows"])
    _require_columns(scope_gate_decisions, ["scope", "decision"])
    _require_columns(
        gap_delta_scope_summary,
        ["scope", "lcri_more_stable_share", "lcri_less_stable_share"],
    )
    _require_columns(
        scope_stability_contradictions,
        ["scope", "fragility_review_required_rows", "contradiction_label"],
    )
    _require_columns(
        ci_confidence_scorecard,
        ["scope", "ci_gate_contradiction_rows", "high_priority_ci_gate_rows"],
    )
    _require_columns(
        uncertainty_weighted_priorities,
        ["scope", "priority_label", "uncertainty_weighted_priority"],
    )

    output = severity_by_scope[
        ["scope", "rows", "warning_rows", "critical_rows"]
    ].rename(columns={"rows": "severity_rows"})
    output = output.merge(
        scope_gate_decisions[["scope", "decision"]].rename(columns={"decision": "gate_decision"}),
        on="scope",
        how="left",
    )
    output = output.merge(
        gap_delta_scope_summary[["scope", "lcri_more_stable_share", "lcri_less_stable_share"]],
        on="scope",
        how="left",
    )
    output = output.merge(
        scope_stability_contradictions[
            ["scope", "fragility_review_required_rows", "contradiction_label"]
        ],
        on="scope",
        how="left",
    )
    output = output.merge(
        ci_confidence_scorecard[
            ["scope", "ci_gate_contradiction_rows", "high_priority_ci_gate_rows"]
        ],
        on="scope",
        how="left",
    )
    output = output.merge(
        uncertainty_weighted_priorities[
            ["scope", "priority_label", "uncertainty_weighted_priority"]
        ],
        on="scope",
        how="left",
    )

    output["gate_decision"] = output["gate_decision"].fillna("missing_gate_decision")
    output["contradiction_label"] = output["contradiction_label"].fillna("missing_contradiction_review")
    output["priority_label"] = output["priority_label"].fillna("none")
    for column in [
        "severity_rows",
        "warning_rows",
        "critical_rows",
        "fragility_review_required_rows",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
    ]:
        output[column] = output[column].fillna(0).astype(int)
    for column in [
        "lcri_more_stable_share",
        "lcri_less_stable_share",
        "uncertainty_weighted_priority",
    ]:
        output[column] = output[column].fillna(0.0).astype(float)

    output["evidence_score"] = [
        _cross_artifact_evidence_score(row) for row in output.to_dict("records")
    ]
    output["evidence_label"] = [
        _cross_artifact_evidence_label(score) for score in output["evidence_score"]
    ]
    output["review_note"] = [
        _cross_artifact_evidence_note(row) for row in output.to_dict("records")
    ]
    return output[columns].sort_values(
        [
            "evidence_score",
            "critical_rows",
            "high_priority_ci_gate_rows",
            "ci_gate_contradiction_rows",
            "lcri_less_stable_share",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def lcri_cross_artifact_evidence_index_summary(
    evidence_index: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize cross-artifact scope evidence for release review."""
    if evidence_index.empty:
        return {
            "scopes": 0,
            "urgent_scopes": 0,
            "review_scopes": 0,
            "monitor_scopes": 0,
            "aligned_scopes": 0,
            "max_evidence_score": 0.0,
            "worst_evidence_scope": "none",
        }
    _require_columns(evidence_index, ["scope", "evidence_label", "evidence_score"])
    labels = evidence_index["evidence_label"].astype(str)
    scores = evidence_index["evidence_score"].astype(float)
    worst = evidence_index.loc[scores.idxmax()]
    return {
        "scopes": len(evidence_index),
        "urgent_scopes": int((labels == "urgent").sum()),
        "review_scopes": int((labels == "review").sum()),
        "monitor_scopes": int((labels == "monitor").sum()),
        "aligned_scopes": int((labels == "aligned").sum()),
        "max_evidence_score": float(scores.max()),
        "worst_evidence_scope": f"{worst['scope']}:{worst['evidence_label']}",
    }


def lcri_evidence_release_checklist(evidence_index: pd.DataFrame) -> pd.DataFrame:
    """Convert the cross-artifact evidence index into release-owner checklist rows.

    The evidence index is optimized for diagnosis. The checklist is optimized for
    go/no-go review: one row per scope, with a deterministic status and the next
    owner action derived from the joined gate, stability, CI, and priority evidence.
    """
    columns = [
        "scope",
        "check_status",
        "checklist_item",
        "gate_decision",
        "evidence_label",
        "evidence_score",
        "priority_label",
        "required_action",
        "source_artifact",
    ]
    if evidence_index.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(
        evidence_index,
        [
            "scope",
            "gate_decision",
            "evidence_label",
            "evidence_score",
            "priority_label",
        ],
    )
    output = evidence_index[
        ["scope", "gate_decision", "evidence_label", "evidence_score", "priority_label"]
    ].copy()
    output["evidence_score"] = output["evidence_score"].astype(float)
    output["check_status"] = [_release_check_status(row) for row in output.to_dict("records")]
    output["checklist_item"] = [
        f"{row['scope']} release evidence reconciliation" for row in output.to_dict("records")
    ]
    output["required_action"] = [
        _release_check_required_action(row) for row in output.to_dict("records")
    ]
    output["source_artifact"] = "lcri_cross_artifact_evidence_index.csv"
    return output[columns].sort_values(
        ["check_status", "evidence_score", "scope"],
        ascending=[True, False, True],
        key=lambda values: values.map(_release_check_status_rank)
        if values.name == "check_status"
        else values,
    ).reset_index(drop=True)


def lcri_evidence_release_checklist_summary(
    checklist: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize evidence-derived release checklist status counts."""
    if checklist.empty:
        return {
            "items": 0,
            "blocked_items": 0,
            "review_items": 0,
            "monitor_items": 0,
            "ready_items": 0,
            "max_evidence_score": 0.0,
            "worst_check_scope": "none",
            "release_ready": True,
        }
    _require_columns(checklist, ["scope", "check_status", "evidence_score"])
    statuses = checklist["check_status"].astype(str)
    scores = checklist["evidence_score"].astype(float)
    worst = checklist.loc[scores.idxmax()]
    blocked = int((statuses == "blocked").sum())
    review = int((statuses == "needs_review").sum())
    return {
        "items": len(checklist),
        "blocked_items": blocked,
        "review_items": review,
        "monitor_items": int((statuses == "monitor").sum()),
        "ready_items": int((statuses == "ready").sum()),
        "max_evidence_score": float(scores.max()),
        "worst_check_scope": f"{worst['scope']}:{worst['check_status']}",
        "release_ready": blocked == 0 and review == 0,
    }


def lcri_owner_handoff_packet(
    evidence_index: pd.DataFrame,
    release_checklist: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact owner handoff queue from evidence and release checklist artifacts.

    The release checklist says what blocks sign-off. This handoff packet adds the
    highest-value supporting evidence fields so an owner can review one sorted
    CSV without reopening the full cross-artifact index.
    """
    columns = [
        "scope",
        "handoff_rank",
        "handoff_status",
        "owner_queue",
        "check_status",
        "gate_decision",
        "evidence_label",
        "evidence_score",
        "priority_label",
        "critical_rows",
        "warning_rows",
        "fragility_review_required_rows",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
        "lcri_less_stable_share",
        "required_action",
        "evidence_source_artifact",
        "checklist_source_artifact",
    ]
    if evidence_index.empty and release_checklist.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(
        evidence_index,
        [
            "scope",
            "critical_rows",
            "warning_rows",
            "fragility_review_required_rows",
            "ci_gate_contradiction_rows",
            "high_priority_ci_gate_rows",
            "lcri_less_stable_share",
        ],
    )
    _require_columns(
        release_checklist,
        [
            "scope",
            "check_status",
            "gate_decision",
            "evidence_label",
            "evidence_score",
            "priority_label",
            "required_action",
            "source_artifact",
        ],
    )

    output = release_checklist[
        [
            "scope",
            "check_status",
            "gate_decision",
            "evidence_label",
            "evidence_score",
            "priority_label",
            "required_action",
            "source_artifact",
        ]
    ].rename(columns={"source_artifact": "evidence_source_artifact"})
    output = output.merge(
        evidence_index[
            [
                "scope",
                "critical_rows",
                "warning_rows",
                "fragility_review_required_rows",
                "ci_gate_contradiction_rows",
                "high_priority_ci_gate_rows",
                "lcri_less_stable_share",
            ]
        ],
        on="scope",
        how="left",
    )
    output["evidence_score"] = output["evidence_score"].astype(float)
    for column in [
        "critical_rows",
        "warning_rows",
        "fragility_review_required_rows",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
    ]:
        output[column] = output[column].fillna(0).astype(int)
    output["lcri_less_stable_share"] = output["lcri_less_stable_share"].fillna(0.0).astype(float)
    output["handoff_status"] = [_owner_handoff_status(row) for row in output.to_dict("records")]
    output["owner_queue"] = [_owner_handoff_queue(row) for row in output.to_dict("records")]
    output["checklist_source_artifact"] = "lcri_evidence_release_checklist.csv"
    output = output.sort_values(
        [
            "handoff_status",
            "evidence_score",
            "critical_rows",
            "high_priority_ci_gate_rows",
            "scope",
        ],
        ascending=[True, False, False, False, True],
        key=lambda values: values.map(_owner_handoff_status_rank)
        if values.name == "handoff_status"
        else values,
    ).reset_index(drop=True)
    output["handoff_rank"] = range(1, len(output) + 1)
    return output[columns]


def lcri_owner_handoff_packet_summary(
    packet: pd.DataFrame,
) -> dict[str, float | int | str | bool]:
    """Summarize owner handoff queue readiness."""
    if packet.empty:
        return {
            "items": 0,
            "immediate_items": 0,
            "review_items": 0,
            "monitor_items": 0,
            "signoff_items": 0,
            "max_evidence_score": 0.0,
            "top_handoff_scope": "none",
            "handoff_clear": True,
        }
    _require_columns(packet, ["scope", "handoff_status", "evidence_score"])
    statuses = packet["handoff_status"].astype(str)
    scores = packet["evidence_score"].astype(float)
    top = packet.loc[scores.idxmax()]
    immediate = int((statuses == "immediate_owner_decision").sum())
    review = int((statuses == "owner_review").sum())
    return {
        "items": len(packet),
        "immediate_items": immediate,
        "review_items": review,
        "monitor_items": int((statuses == "release_note_monitor").sum()),
        "signoff_items": int((statuses == "signoff_ready").sum()),
        "max_evidence_score": float(scores.max()),
        "top_handoff_scope": f"{top['scope']}:{top['handoff_status']}",
        "handoff_clear": immediate == 0 and review == 0,
    }


def lcri_evidence_lineage_map(
    evidence_index: pd.DataFrame,
    release_checklist: pd.DataFrame,
    owner_handoff_packet: pd.DataFrame,
) -> pd.DataFrame:
    """Trace each release-review scope from evidence index to owner handoff.

    The evidence index, release checklist, and handoff packet intentionally repeat
    scope-level fields for different owner surfaces. This lineage map is a
    compact audit layer that records whether every scope still has an intact
    source chain from cross-artifact evidence through final handoff.
    """
    columns = [
        "scope",
        "evidence_label",
        "check_status",
        "handoff_status",
        "evidence_score",
        "evidence_source_artifact",
        "checklist_source_artifact",
        "handoff_source_artifact",
        "lineage_status",
        "lineage_note",
    ]
    if evidence_index.empty and release_checklist.empty and owner_handoff_packet.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(evidence_index, ["scope", "evidence_label", "evidence_score"])
    _require_columns(release_checklist, ["scope", "check_status", "source_artifact"])
    _require_columns(
        owner_handoff_packet,
        ["scope", "handoff_status", "evidence_source_artifact", "checklist_source_artifact"],
    )

    output = evidence_index[["scope", "evidence_label", "evidence_score"]].copy()
    output["evidence_source_artifact"] = "lcri_cross_artifact_evidence_index.csv"
    output = output.merge(
        release_checklist[["scope", "check_status", "source_artifact"]].rename(
            columns={"source_artifact": "checklist_source_artifact"}
        ),
        on="scope",
        how="outer",
    )
    output = output.merge(
        owner_handoff_packet[
            ["scope", "handoff_status", "evidence_source_artifact", "checklist_source_artifact"]
        ].rename(
            columns={
                "evidence_source_artifact": "handoff_source_artifact",
                "checklist_source_artifact": "handoff_checklist_source_artifact",
            }
        ),
        on="scope",
        how="outer",
    )

    output["evidence_score"] = output["evidence_score"].fillna(0.0).astype(float)
    for column in ["evidence_label", "check_status", "handoff_status"]:
        output[column] = output[column].fillna("missing")
    output["evidence_source_artifact"] = output["evidence_source_artifact"].fillna(
        "missing_evidence_index"
    )
    output["checklist_source_artifact"] = output["checklist_source_artifact"].fillna(
        "missing_release_checklist"
    )
    output["handoff_source_artifact"] = output["handoff_source_artifact"].fillna(
        "missing_owner_handoff"
    )
    output["handoff_checklist_source_artifact"] = output[
        "handoff_checklist_source_artifact"
    ].fillna("missing_owner_handoff")

    output["lineage_status"] = [
        _evidence_lineage_status(row) for row in output.to_dict("records")
    ]
    output["lineage_note"] = [
        _evidence_lineage_note(row) for row in output.to_dict("records")
    ]
    return output[columns].sort_values(
        ["lineage_status", "evidence_score", "scope"],
        ascending=[True, False, True],
        key=lambda values: values.map(_evidence_lineage_status_rank)
        if values.name == "lineage_status"
        else values,
    ).reset_index(drop=True)


def lcri_evidence_lineage_map_summary(
    lineage_map: pd.DataFrame,
) -> dict[str, float | int | str | bool]:
    """Summarize source-chain completeness for owner-facing LCRI evidence."""
    if lineage_map.empty:
        return {
            "scopes": 0,
            "complete_scopes": 0,
            "source_mismatch_scopes": 0,
            "incomplete_scopes": 0,
            "max_evidence_score": 0.0,
            "worst_lineage_scope": "none",
            "lineage_clear": True,
        }
    _require_columns(lineage_map, ["scope", "lineage_status", "evidence_score"])
    statuses = lineage_map["lineage_status"].astype(str)
    scores = lineage_map["evidence_score"].astype(float)
    worst = lineage_map.loc[scores.idxmax()]
    mismatch = int((statuses == "source_mismatch").sum())
    incomplete = int((statuses == "incomplete_lineage").sum())
    return {
        "scopes": len(lineage_map),
        "complete_scopes": int((statuses == "complete").sum()),
        "source_mismatch_scopes": mismatch,
        "incomplete_scopes": incomplete,
        "max_evidence_score": float(scores.max()),
        "worst_lineage_scope": f"{worst['scope']}:{worst['lineage_status']}",
        "lineage_clear": mismatch == 0 and incomplete == 0,
    }


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
        "lcri_more_stable_share",
        "lcri_less_stable_share",
    ]
    if gap_delta.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(gap_delta, ["scope", column])

    grouped = (
        gap_delta.groupby("scope", sort=True)[column]
        .agg(
            rows="count",
            mean_raw_minus_lcri_gap="mean",
            min_raw_minus_lcri_gap="min",
            max_raw_minus_lcri_gap="max",
        )
        .reset_index()
    )
    shares = (
        gap_delta.assign(
            lcri_more_stable=lambda frame: frame[column].astype(float) > 0.0,
            lcri_less_stable=lambda frame: frame[column].astype(float) < 0.0,
        )
        .groupby("scope", sort=True)[["lcri_more_stable", "lcri_less_stable"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "lcri_more_stable": "lcri_more_stable_share",
                "lcri_less_stable": "lcri_less_stable_share",
            }
        )
    )
    return grouped.merge(shares, on="scope", how="left")[columns]


def lcri_gap_delta_dominant_scopes(scope_summary: pd.DataFrame) -> dict[str, float | str]:
    """Identify the scopes where LCRI has the strongest relative edge and drag."""
    if scope_summary.empty:
        return {
            "best_scope": "none",
            "best_mean_raw_minus_lcri_gap": 0.0,
            "worst_scope": "none",
            "worst_mean_raw_minus_lcri_gap": 0.0,
        }
    _require_columns(scope_summary, ["scope", "mean_raw_minus_lcri_gap"])

    values = scope_summary["mean_raw_minus_lcri_gap"].astype(float)
    best = scope_summary.loc[values.idxmax()]
    worst = scope_summary.loc[values.idxmin()]
    return {
        "best_scope": str(best["scope"]),
        "best_mean_raw_minus_lcri_gap": float(best["mean_raw_minus_lcri_gap"]),
        "worst_scope": str(worst["scope"]),
        "worst_mean_raw_minus_lcri_gap": float(worst["mean_raw_minus_lcri_gap"]),
    }


def lcri_gap_delta_scope_extremes(gap_delta: pd.DataFrame) -> pd.DataFrame:
    """Return the strongest LCRI stability and instability edge per scope."""
    column = "raw_minus_lcri_directional_accuracy_gap"
    columns = [
        "scope",
        "best_context",
        "best_raw_minus_lcri_gap",
        "worst_context",
        "worst_raw_minus_lcri_gap",
    ]
    if gap_delta.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(gap_delta, ["scope", "context", column])

    rows = []
    for scope, group in gap_delta.groupby("scope", sort=True):
        values = group[column].astype(float)
        best = group.loc[values.idxmax()]
        worst = group.loc[values.idxmin()]
        rows.append(
            {
                "scope": scope,
                "best_context": str(best["context"]),
                "best_raw_minus_lcri_gap": float(best[column]),
                "worst_context": str(worst["context"]),
                "worst_raw_minus_lcri_gap": float(worst[column]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


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


def signal_quantile_monotonicity(
    frame: pd.DataFrame,
    signal: str,
    *,
    quantiles: int = 10,
) -> pd.DataFrame:
    """Audit whether larger signal quantiles map to higher future-up frequency.

    Directional accuracy can hide a brittle score shape. This diagnostic bins the
    signal by empirical quantile and reports the observed target rate, adjacent
    target-rate slope, and monotonicity violation flag for each populated bucket.
    """
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    _require_columns(frame, [signal, "future_direction"])
    score = frame[signal].to_numpy(dtype=float)
    target = frame["future_direction"].to_numpy(dtype=float)
    if not np.isfinite(np.column_stack([score, target])).all():
        raise ValueError("monotonicity inputs must be finite")
    if np.std(score) == 0.0:
        raise ValueError("signal must vary across rows")

    bucket = pd.qcut(score, q=quantiles, labels=False, duplicates="drop")
    rows = []
    previous_rate: float | None = None
    for quantile in sorted(pd.Series(bucket).dropna().unique()):
        mask = bucket == quantile
        target_rate = float(np.mean(target[mask]))
        slope = 0.0 if previous_rate is None else target_rate - previous_rate
        rows.append(
            {
                "signal": signal,
                "quantile": int(quantile),
                "rows": int(np.sum(mask)),
                "mean_score": float(np.mean(score[mask])),
                "observed_frequency": target_rate,
                "adjacent_frequency_slope": slope,
                "monotonicity_violation": bool(previous_rate is not None and slope < 0.0),
            }
        )
        previous_rate = target_rate
    return pd.DataFrame(rows)



def signal_quantile_monotonicity_summary(
    monotonicity: pd.DataFrame,
) -> dict[str, float | int | str | bool]:
    """Summarize quantile monotonicity violations for model validation gates."""
    if monotonicity.empty:
        return {
            "quantiles": 0,
            "monotonicity_violation_rows": 0,
            "worst_negative_slope": 0.0,
            "worst_negative_slope_quantile": "none",
            "mean_observed_frequency": 0.0,
            "passes_monotonicity_gate": True,
        }
    _require_columns(
        monotonicity,
        [
            "quantile",
            "observed_frequency",
            "adjacent_frequency_slope",
            "monotonicity_violation",
        ],
    )
    slope = monotonicity["adjacent_frequency_slope"].astype(float)
    observed = monotonicity["observed_frequency"].astype(float)
    if not np.isfinite(np.column_stack([slope, observed])).all():
        raise ValueError("monotonicity diagnostics must be finite")
    violations = monotonicity["monotonicity_violation"].astype(bool)
    worst = monotonicity.loc[slope.idxmin()]
    return {
        "quantiles": len(monotonicity),
        "monotonicity_violation_rows": int(violations.sum()),
        "worst_negative_slope": float(min(slope.min(), 0.0)),
        "worst_negative_slope_quantile": str(worst["quantile"]),
        "mean_observed_frequency": float(observed.mean()),
        "passes_monotonicity_gate": bool(not violations.any()),
    }



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



def _fragility_rows(
    metrics: pd.DataFrame,
    heldout_metrics: pd.DataFrame,
    *,
    scope: str,
    context_column: str | None,
) -> list[dict[str, float | int | str]]:
    required = ["signal", "rows", "directional_accuracy"]
    _require_columns(metrics, [*( [context_column] if context_column else [] ), *required])
    _require_columns(heldout_metrics, [*( [context_column] if context_column else [] ), *required])
    index_columns = [context_column] if context_column else []
    full = metrics.set_index([*index_columns, "signal"])
    heldout = heldout_metrics.set_index([*index_columns, "signal"])
    keys = [key for key in full.index if key in heldout.index]
    rows = []
    for key in keys:
        context = key[0] if context_column else "all"
        signal = key[1] if context_column else key
        full_row = full.loc[key]
        heldout_row = heldout.loc[key]
        full_accuracy = float(full_row["directional_accuracy"])
        heldout_accuracy = float(heldout_row["directional_accuracy"])
        heldout_rows = int(heldout_row["rows"])
        se = _binomial_standard_error(heldout_accuracy, heldout_rows)
        gap = full_accuracy - heldout_accuracy
        ratio = abs(gap) / se if se > 0.0 else 0.0
        rows.append(
            {
                "scope": scope,
                "context": str(context),
                "signal": str(signal),
                "full_rows": int(full_row["rows"]),
                "heldout_rows": heldout_rows,
                "full_directional_accuracy": full_accuracy,
                "heldout_directional_accuracy": heldout_accuracy,
                "directional_accuracy_gap": float(gap),
                "heldout_directional_accuracy_se": float(se),
                "abs_gap_to_se_ratio": float(ratio),
                "fragility_label": _fragility_label(ratio),
            }
        )
    return rows


def _binomial_standard_error(probability: float, rows: int) -> float:
    if rows <= 0:
        return 0.0
    clipped = min(max(float(probability), 0.0), 1.0)
    return float(np.sqrt(clipped * (1.0 - clipped) / rows))


def _normal_confidence_level(z_score: float) -> float:
    return float(erf(z_score / sqrt(2.0)))


def _fragility_label(abs_gap_to_se_ratio: float) -> str:
    if abs_gap_to_se_ratio >= 3.0:
        return "fragile"
    if abs_gap_to_se_ratio >= 1.96:
        return "watch"
    return "stable"


def _ci_gate_contradiction_label(severity: object, gap_exceeds_ci_half_width: object) -> str:
    severity_label = str(severity)
    exceeds = bool(gap_exceeds_ci_half_width)
    if severity_label == "critical" and not exceeds:
        return "gate_blocks_inside_ci"
    if severity_label == "warning" and not exceeds:
        return "gate_warns_inside_ci"
    if severity_label == "stable" and exceeds:
        return "stable_gap_outside_ci"
    return "aligned"


def _ci_gate_review_priority(label: object, severity: object, ci_width: object) -> int:
    label_text = str(label)
    severity_text = str(severity)
    width = float(ci_width)
    if label_text == "gate_blocks_inside_ci":
        return 3
    if label_text == "stable_gap_outside_ci":
        return 2
    if label_text == "gate_warns_inside_ci":
        return 2 if width >= 0.20 else 1
    if severity_text == "critical":
        return 2
    return 1


def _ci_gate_review_note(row: dict[str, object]) -> str:
    label = str(row.get("ci_gate_label", "aligned"))
    if label == "gate_blocks_inside_ci":
        return "critical gate blocker is inside heldout CI half-width; inspect sample uncertainty before release block"
    if label == "gate_warns_inside_ci":
        return "warning gate row is inside heldout CI half-width; treat as uncertainty-qualified warning"
    if label == "stable_gap_outside_ci":
        return "stable deterministic row still exceeds heldout CI half-width; review threshold margin"
    return "deterministic gate label and heldout CI evidence agree"


def _ci_confidence_coverage_label(
    *,
    wide_rows: int,
    outside_rows: int,
    contradiction_rows: int,
    high_priority_rows: int,
) -> str:
    if high_priority_rows:
        return "blocking_ci_gate_review"
    if contradiction_rows:
        return "ci_gate_contradiction_review"
    if wide_rows and outside_rows:
        return "wide_ci_review"
    return "adequate_ci_coverage"


def _ci_confidence_coverage_note(label: str) -> str:
    if label == "blocking_ci_gate_review":
        return "scope has high-priority CI/gate disagreement; review before accepting gate owner decision"
    if label == "ci_gate_contradiction_review":
        return "scope has CI/gate disagreements; inspect uncertainty-qualified gate rows"
    if label == "wide_ci_review":
        return "scope has wide heldout intervals with gaps outside CI half-width; prioritize more coverage"
    return "scope CI coverage is aligned with current gate evidence"


def _fragility_gate_alignment_label(severity: object, fragility: object) -> str:
    severity_label = str(severity)
    fragility_label = str(fragility)
    if severity_label == "critical" and fragility_label == "stable":
        return "gate_blocks_stable_slice"
    if severity_label in {"stable", "warning"} and fragility_label == "fragile":
        return "uncertainty_fragile_noncritical"
    if severity_label == "stable" and fragility_label == "watch":
        return "uncertainty_watch_stable_gap"
    return "aligned"


def _fragility_gate_review_note(row: dict[str, object]) -> str:
    label = str(row.get("alignment_label", "aligned"))
    if label == "gate_blocks_stable_slice":
        return "critical gate blocker exceeds deterministic threshold despite stable heldout uncertainty"
    if label == "uncertainty_fragile_noncritical":
        return "non-critical LCRI gap is large relative to heldout uncertainty; review before dismissing"
    if label == "uncertainty_watch_stable_gap":
        return "stable gate row is near heldout uncertainty watch threshold"
    return "gate severity and heldout fragility agree"


def _lcri_scope_stability_contradiction_label(row: dict[str, object]) -> str:
    decision = str(row.get("decision", ""))
    more_share = float(row.get("lcri_more_stable_share", 0.0))
    less_share = float(row.get("lcri_less_stable_share", 0.0))
    if decision == "block" and more_share >= 0.5:
        return "gate_blocks_despite_relative_stability"
    if decision == "pass" and less_share > 0.0:
        return "pass_scope_with_relative_regressions"
    if decision == "warn" and less_share >= 0.5 and more_share == 0.0:
        return "warning_scope_with_broad_relative_regression"
    return "aligned"


def _lcri_scope_stability_contradiction_note(row: dict[str, object]) -> str:
    label = str(row.get("contradiction_label", "aligned"))
    if label == "gate_blocks_despite_relative_stability":
        return "absolute LCRI gate blocks while LCRI is usually more stable than raw imbalance in this scope"
    if label == "pass_scope_with_relative_regressions":
        return "absolute LCRI gate passes but at least one context is less stable than raw imbalance"
    if label == "warning_scope_with_broad_relative_regression":
        return "warning scope also shows broad relative regression versus raw imbalance"
    if int(row.get("fragility_review_required_rows", 0)) > 0:
        return "scope is gate/delta aligned but has fragility rows requiring review"
    return "scope gate posture and relative-stability dashboard agree"


def _worst_scope_gate_row(lcri_severity: pd.DataFrame, scope: str) -> dict[str, float | str]:
    scope_rows = lcri_severity.loc[lcri_severity["scope"].astype(str) == scope]
    if scope_rows.empty:
        return {
            "context": "none",
            "severity": "none",
            "directional_accuracy_gap": 0.0,
        }
    values = scope_rows["directional_accuracy_gap"].astype(float)
    row = scope_rows.loc[values.idxmax()]
    return {
        "context": str(row["context"]),
        "severity": str(row["severity"]),
        "directional_accuracy_gap": float(row["directional_accuracy_gap"]),
    }


def _worst_scope_delta_row(gap_delta: pd.DataFrame, scope: str) -> dict[str, float | str]:
    scope_rows = gap_delta.loc[gap_delta["scope"].astype(str) == scope]
    if scope_rows.empty:
        return {
            "context": "none",
            "raw_minus_lcri_directional_accuracy_gap": 0.0,
        }
    values = scope_rows["raw_minus_lcri_directional_accuracy_gap"].astype(float)
    row = scope_rows.loc[values.idxmin()]
    return {
        "context": str(row["context"]),
        "raw_minus_lcri_directional_accuracy_gap": float(
            row["raw_minus_lcri_directional_accuracy_gap"]
        ),
    }


def _worst_scope_fragility_row(
    fragility_gate_alignment: pd.DataFrame,
    scope: str,
) -> dict[str, float | str]:
    scope_rows = fragility_gate_alignment.loc[
        fragility_gate_alignment["scope"].astype(str) == scope
    ]
    if scope_rows.empty:
        return {
            "context": "none",
            "alignment_label": "none",
            "abs_gap_to_se_ratio": 0.0,
        }
    review_rows = scope_rows.loc[scope_rows["alignment_label"].astype(str) != "aligned"]
    candidate = review_rows if not review_rows.empty else scope_rows
    values = candidate["abs_gap_to_se_ratio"].astype(float)
    row = candidate.loc[values.idxmax()]
    return {
        "context": str(row["context"]),
        "alignment_label": str(row["alignment_label"]),
        "abs_gap_to_se_ratio": float(row["abs_gap_to_se_ratio"]),
    }


def _contradiction_review_priority(contradiction_label: str, fragility_review_rows: int) -> int:
    if contradiction_label != "aligned" and fragility_review_rows > 0:
        return 3
    if contradiction_label != "aligned":
        return 2
    if fragility_review_rows > 0:
        return 1
    return 0


def _uncertainty_weighted_priority(row: dict[str, object]) -> float:
    base = float(row.get("base_review_priority", 0.0))
    fragility_ratio = max(float(row.get("worst_fragility_abs_gap_to_se_ratio", 0.0)), 0.0)
    max_ci_width = max(float(row.get("max_ci_width", 0.0)), 0.0)
    wide_ci_share = max(float(row.get("wide_ci_share", 0.0)), 0.0)
    ci_gate_rows = max(int(row.get("ci_gate_contradiction_rows", 0)), 0)
    high_priority_rows = max(int(row.get("high_priority_ci_gate_rows", 0)), 0)
    score = (
        base
        + min(fragility_ratio / 2.0, 2.0)
        + min(max_ci_width * 2.0, 1.0)
        + min(wide_ci_share, 1.0)
        + min(ci_gate_rows * 0.5, 1.0)
        + min(high_priority_rows, 2.0)
    )
    return float(score)


def _uncertainty_weighted_priority_label(score: float) -> str:
    if score >= 6.0:
        return "critical"
    if score >= 4.5:
        return "high"
    if score >= 2.5:
        return "medium"
    return "low"


def _uncertainty_weighted_priority_note(row: dict[str, object]) -> str:
    label = str(row.get("priority_label", "low"))
    coverage = str(row.get("coverage_label", "missing_ci_coverage"))
    if label == "critical":
        return f"review first: contradiction evidence is amplified by {coverage} and heldout uncertainty"
    if label == "high":
        return f"prioritize review: scope combines owner contradiction evidence with {coverage}"
    if label == "medium":
        return f"schedule review after critical/high scopes; {coverage} uncertainty evidence is non-trivial"
    return "low urgency unless release owners need complete audit coverage"


def _cross_artifact_evidence_score(row: dict[str, object]) -> float:
    gate_weight = {"block": 3.0, "warn": 1.5, "pass": 0.0}.get(
        str(row.get("gate_decision", "pass")),
        0.5,
    )
    priority_weight = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.25}.get(
        str(row.get("priority_label", "none")),
        0.0,
    )
    score = (
        gate_weight
        + min(int(row.get("critical_rows", 0)) * 2.0, 4.0)
        + min(int(row.get("warning_rows", 0)) * 1.0, 2.0)
        + min(float(row.get("lcri_less_stable_share", 0.0)) * 2.0, 2.0)
        + min(int(row.get("fragility_review_required_rows", 0)) * 0.75, 2.25)
        + min(int(row.get("ci_gate_contradiction_rows", 0)) * 0.75, 1.5)
        + min(int(row.get("high_priority_ci_gate_rows", 0)) * 1.25, 2.5)
        + min(float(row.get("uncertainty_weighted_priority", 0.0)) / 3.0, 2.0)
        + priority_weight
    )
    return float(score)


def _cross_artifact_evidence_label(score: float) -> str:
    if score >= 9.0:
        return "urgent"
    if score >= 5.0:
        return "review"
    if score >= 2.0:
        return "monitor"
    return "aligned"


def _cross_artifact_evidence_note(row: dict[str, object]) -> str:
    label = str(row.get("evidence_label", "aligned"))
    decision = str(row.get("gate_decision", "pass"))
    contradiction = str(row.get("contradiction_label", "aligned"))
    if label == "urgent":
        return f"owner review first: {decision} gate with {contradiction} cross-artifact evidence"
    if label == "review":
        return f"review before release sign-off: {decision} gate and {contradiction} evidence need reconciliation"
    if label == "monitor":
        return "monitor in release notes: evidence is non-zero but below blocking review thresholds"
    return "aligned across gate, stability, fragility, CI, and priority artifacts"


def _release_check_status(row: dict[str, object]) -> str:
    evidence_label = str(row.get("evidence_label", "aligned"))
    gate_decision = str(row.get("gate_decision", "pass"))
    priority_label = str(row.get("priority_label", "low"))
    if evidence_label == "urgent" or gate_decision == "block" or priority_label == "critical":
        return "blocked"
    if evidence_label == "review" or gate_decision == "warn" or priority_label in {"high", "medium"}:
        return "needs_review"
    if evidence_label == "monitor":
        return "monitor"
    return "ready"


def _release_check_required_action(row: dict[str, object]) -> str:
    status = str(row.get("check_status", _release_check_status(row)))
    scope = str(row.get("scope", "scope"))
    if status == "blocked":
        return f"resolve or explicitly waive {scope} evidence before release sign-off"
    if status == "needs_review":
        return f"owner review required for {scope} before final go/no-go"
    if status == "monitor":
        return f"include {scope} evidence caveat in release notes"
    return f"{scope} evidence is ready for release checklist sign-off"


def _release_check_status_rank(status: object) -> int:
    return {"blocked": 0, "needs_review": 1, "monitor": 2, "ready": 3}.get(str(status), 4)


def _evidence_lineage_status(row: dict[str, object]) -> str:
    if (
        str(row.get("evidence_label")) != "missing"
        and str(row.get("check_status")) != "missing"
        and str(row.get("handoff_status")) != "missing"
        and str(row.get("checklist_source_artifact")) == "lcri_cross_artifact_evidence_index.csv"
        and str(row.get("handoff_source_artifact")) == "lcri_cross_artifact_evidence_index.csv"
        and str(row.get("handoff_checklist_source_artifact"))
        == "lcri_evidence_release_checklist.csv"
    ):
        return "complete"
    if (
        str(row.get("checklist_source_artifact"))
        not in {"lcri_cross_artifact_evidence_index.csv", "missing_release_checklist"}
        or str(row.get("handoff_source_artifact"))
        not in {"lcri_cross_artifact_evidence_index.csv", "missing_owner_handoff"}
        or str(row.get("handoff_checklist_source_artifact"))
        not in {"lcri_evidence_release_checklist.csv", "missing_owner_handoff"}
    ):
        return "source_mismatch"
    return "incomplete_lineage"


def _evidence_lineage_note(row: dict[str, object]) -> str:
    scope = str(row.get("scope", "unknown"))
    status = _evidence_lineage_status(row)
    if status == "complete":
        return f"{scope} evidence chain is complete from index to release checklist to handoff"
    if status == "source_mismatch":
        return f"{scope} has a stale or unexpected source artifact reference"
    return f"{scope} is missing at least one owner-facing lineage artifact"


def _evidence_lineage_status_rank(status: object) -> int:
    return {"source_mismatch": 0, "incomplete_lineage": 1, "complete": 2}.get(str(status), 3)


def _owner_handoff_status(row: dict[str, object]) -> str:
    status = str(row.get("check_status", "ready"))
    if status == "blocked":
        return "immediate_owner_decision"
    if status == "needs_review":
        return "owner_review"
    if status == "monitor":
        return "release_note_monitor"
    return "signoff_ready"


def _owner_handoff_queue(row: dict[str, object]) -> str:
    handoff = str(row.get("handoff_status", _owner_handoff_status(row)))
    scope = str(row.get("scope", "scope"))
    if handoff == "immediate_owner_decision":
        return f"owner must decide waive/fix posture for {scope} before release"
    if handoff == "owner_review":
        return f"owner review queue for {scope} evidence reconciliation"
    if handoff == "release_note_monitor":
        return f"capture {scope} caveat in release note monitor section"
    return f"{scope} can proceed to sign-off after checklist confirmation"


def _owner_handoff_status_rank(status: object) -> int:
    return {
        "immediate_owner_decision": 0,
        "owner_review": 1,
        "release_note_monitor": 2,
        "signoff_ready": 3,
    }.get(str(status), 4)


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
