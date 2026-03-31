"""
Automated Root Cause Analysis using Causal Inference.

When an anomaly is detected, this module:
1. Identifies correlated metric changes (correlation analysis)
2. Applies Granger causality to find leading indicators
3. Segments the anomaly by region/device/category
4. Generates a structured root cause report

Based on Amazon's internal RCA playbook patterns.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


def correlate_metrics(df: pd.DataFrame, target: str, features: List[str]) -> Dict:
    """
    Compute Pearson correlation between target metric and candidate causes.
    Returns ranked list of correlated features.
    """
    correlations = {}
    target_series = df[target].fillna(0)

    for feat in features:
        if feat == target or feat not in df.columns:
            continue
        feat_series = df[feat].fillna(0)
        if feat_series.std() < 1e-6:
            continue
        r, p = stats.pearsonr(target_series, feat_series)
        correlations[feat] = {
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 6),
            "significant": p < 0.05,
            "direction": "positive" if r > 0 else "negative",
            "strength": "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak",
        }

    return dict(sorted(correlations.items(), key=lambda x: -abs(x[1]["pearson_r"])))


def granger_causality(
    df: pd.DataFrame,
    target: str,
    candidate: str,
    max_lag: int = 3,
) -> Dict:
    """
    Granger causality test: does `candidate` Granger-cause `target`?
    If yes, past values of `candidate` help predict `target` — useful
    for finding leading indicators (e.g., error_rate → revenue drop).
    """
    try:
        data = pd.DataFrame({
            "target": df[target].fillna(0),
            "candidate": df[candidate].fillna(0),
        }).dropna()

        if len(data) < 10:
            return {"error": "Insufficient data"}

        results = grangercausalitytests(data[["target", "candidate"]], maxlag=max_lag, verbose=False)

        min_p = 1.0
        best_lag = 1
        for lag, result in results.items():
            p = result[0]["ssr_ftest"][1]
            if p < min_p:
                min_p = p
                best_lag = lag

        return {
            "granger_causes": min_p < 0.05,
            "best_lag": best_lag,
            "p_value": round(min_p, 6),
            "interpretation": (
                f"{candidate} Granger-causes {target} with lag={best_lag}h (p={min_p:.4f})"
                if min_p < 0.05
                else f"No significant Granger causality from {candidate} to {target}"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def segment_anomaly(
    df: pd.DataFrame,
    metric: str,
    anomaly_window_start: str,
    anomaly_window_end: str,
    segment_cols: List[str] = ["region", "device"],
) -> Dict:
    """
    Segment the anomaly to find which dimension is driving it.
    Uses contribution analysis (% of metric from each segment before/after).
    """
    df = df.copy()
    df["timestamp_dt"] = pd.to_datetime(df.get("hour_bucket", df.index))

    baseline = df[df["timestamp_dt"] < anomaly_window_start]
    anomaly  = df[(df["timestamp_dt"] >= anomaly_window_start) &
                  (df["timestamp_dt"] <= anomaly_window_end)]

    segments_report = {}
    for col in segment_cols:
        if col not in df.columns:
            continue
        baseline_by_seg = baseline.groupby(col)[metric].mean()
        anomaly_by_seg  = anomaly.groupby(col)[metric].mean()

        changes = {}
        for seg in set(list(baseline_by_seg.index) + list(anomaly_by_seg.index)):
            b = float(baseline_by_seg.get(seg, 0))
            a = float(anomaly_by_seg.get(seg, 0))
            pct_change = (a - b) / (b + 1e-10)
            changes[str(seg)] = {
                "baseline": round(b, 4),
                "anomaly": round(a, 4),
                "pct_change": round(pct_change, 4),
                "abs_change": round(a - b, 4),
            }

        # Find biggest driver
        biggest = max(changes.items(), key=lambda x: abs(x[1]["pct_change"]))
        segments_report[col] = {
            "breakdown": changes,
            "primary_driver": biggest[0],
            "driver_change": biggest[1]["pct_change"],
        }

    return segments_report


def generate_rca_report(
    hourly_df: pd.DataFrame,
    anomaly: Dict,
    raw_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Full automated Root Cause Analysis report for a detected anomaly.
    """
    target_metric = anomaly.get("metric", "error_rate")
    timestamp = anomaly.get("timestamp", "")

    logger.info(f"Running RCA for {target_metric} anomaly at {timestamp}")

    # 1. Correlation analysis
    candidate_metrics = [
        "error_rate", "avg_response_ms", "total_revenue",
        "event_count", "unique_users", "purchases", "p95_response_ms"
    ]
    candidate_metrics = [m for m in candidate_metrics if m != target_metric and m in hourly_df.columns]
    correlations = correlate_metrics(hourly_df, target_metric, candidate_metrics)

    # 2. Granger causality for top correlated metrics
    granger_results = {}
    top_correlated = sorted(correlations.items(), key=lambda x: -abs(x[1]["pearson_r"]))[:3]
    for feat, _ in top_correlated:
        granger_results[feat] = granger_causality(hourly_df, target_metric, feat)

    # 3. Leading indicators (Granger causal features)
    leading_indicators = [
        feat for feat, result in granger_results.items()
        if result.get("granger_causes", False)
    ]

    # 4. Hypothesis generation
    hypotheses = []
    for feat, corr in list(correlations.items())[:3]:
        r = corr["pearson_r"]
        direction = corr["direction"]
        if feat == "avg_response_ms" and direction == "positive" and target_metric == "error_rate":
            hypotheses.append("Infrastructure degradation: high response times co-occur with errors → possible timeout cascade")
        elif feat == "event_count" and direction == "negative" and target_metric in ["total_revenue", "purchases"]:
            hypotheses.append("Traffic drop caused revenue decline — investigate upstream traffic source")
        elif feat == "error_rate" and direction == "negative" and target_metric == "total_revenue":
            hypotheses.append("Error spike blocking checkout flow — users unable to complete purchases")
        else:
            hypotheses.append(f"{feat} ({direction}ly correlated, r={r:.2f}) may be contributing cause")

    # 5. Recommended actions
    actions = []
    if target_metric == "error_rate" and anomaly.get("value", 0) > 0.05:
        actions.extend([
            "Check service health dashboard for upstream dependencies",
            "Review recent deployments in the anomaly time window",
            "Scale up error-prone services if load-related",
        ])
    elif target_metric == "avg_response_ms":
        actions.extend([
            "Check database query performance and connection pool saturation",
            "Review CDN cache hit rates",
            "Check for N+1 query patterns in recent code changes",
        ])
    elif target_metric == "total_revenue":
        actions.extend([
            "Validate checkout flow end-to-end",
            "Check payment gateway success rates",
            "Review pricing engine for anomalous discounts",
        ])

    return {
        "anomaly": anomaly,
        "target_metric": target_metric,
        "top_correlations": dict(list(correlations.items())[:5]),
        "granger_causality": granger_results,
        "leading_indicators": leading_indicators,
        "hypotheses": hypotheses,
        "recommended_actions": actions,
        "confidence": "high" if leading_indicators else "medium",
        "summary": (
            f"Anomaly detected in {target_metric} at {timestamp}. "
            f"Top correlated metrics: {list(correlations.keys())[:2]}. "
            f"Leading indicators: {leading_indicators if leading_indicators else 'none found'}. "
            f"Hypothesis: {hypotheses[0] if hypotheses else 'under investigation'}."
        ),
    }
