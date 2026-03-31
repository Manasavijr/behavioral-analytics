"""
Funnel Analysis + RFM Customer Segmentation.

Funnel: page_view → search → product_view → add_to_cart → checkout_start → purchase
Measures conversion rates at each stage, drop-off analysis, and cohort comparison.

RFM (Recency, Frequency, Monetary) segmentation for customer value.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

FUNNEL_STAGES = [
    "page_view",
    "search",
    "product_view",
    "add_to_cart",
    "checkout_start",
    "purchase",
]


def compute_funnel(df: pd.DataFrame, segment_by: Optional[str] = None) -> Dict:
    """
    Compute conversion funnel metrics.
    Returns stage counts, conversion rates, and drop-off rates.
    """
    logger.info("Computing conversion funnel...")

    if segment_by and segment_by in df.columns:
        segments = df[segment_by].unique()
        results = {}
        for seg in segments:
            seg_df = df[df[segment_by] == seg]
            results[str(seg)] = _compute_single_funnel(seg_df)
        return results
    return {"overall": _compute_single_funnel(df)}


def _compute_single_funnel(df: pd.DataFrame) -> Dict:
    stage_counts = []
    for stage in FUNNEL_STAGES:
        # Count unique users who performed this event
        users = df[df["event_type"] == stage]["user_id"].nunique()
        stage_counts.append({"stage": stage, "users": users})

    stages = []
    for i, s in enumerate(stage_counts):
        top_users = stage_counts[0]["users"] if stage_counts[0]["users"] > 0 else 1
        prev_users = stage_counts[i-1]["users"] if i > 0 else s["users"]

        conv_from_top  = s["users"] / top_users if top_users > 0 else 0
        conv_from_prev = s["users"] / prev_users if prev_users > 0 else 0
        dropoff        = 1 - conv_from_prev if i > 0 else 0

        stages.append({
            "stage": s["stage"],
            "users": s["users"],
            "conversion_from_top": round(conv_from_top, 4),
            "conversion_from_prev": round(conv_from_prev, 4),
            "dropoff_rate": round(dropoff, 4),
        })

    overall_conversion = (
        stage_counts[-1]["users"] / stage_counts[0]["users"]
        if stage_counts[0]["users"] > 0 else 0
    )

    return {
        "stages": stages,
        "overall_conversion": round(overall_conversion, 4),
        "biggest_dropoff": max(stages[1:], key=lambda x: x["dropoff_rate"])["stage"]
        if len(stages) > 1 else None,
    }


def rfm_segmentation(df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    RFM (Recency, Frequency, Monetary) customer segmentation.
    Segments customers into: Champions, Loyal, At Risk, Lost, New.
    """
    logger.info("Computing RFM segmentation...")

    if reference_date is None:
        reference_date = pd.Timestamp(df["timestamp_dt"].max()) if "timestamp_dt" in df.columns \
            else pd.Timestamp.now()

    purchases = df[df["event_type"] == "purchase"].copy()
    if "timestamp_dt" not in purchases.columns:
        purchases["timestamp_dt"] = pd.to_datetime(purchases["timestamp"], unit="s")

    rfm = purchases.groupby("user_id").agg(
        last_purchase=("timestamp_dt", "max"),
        frequency=("event_id", "count"),
        monetary=("revenue", "sum"),
    ).reset_index()

    rfm["recency_days"] = (reference_date - rfm["last_purchase"]).dt.days

    # RFM scores (1-5)
    rfm["r_score"] = pd.qcut(rfm["recency_days"], 5, labels=[5,4,3,2,1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4: return "Champions"
        if r >= 3 and f >= 3:            return "Loyal Customers"
        if r >= 4 and f <= 2:            return "New Customers"
        if r <= 2 and f >= 3:            return "At Risk"
        if r <= 2 and f <= 2:            return "Lost"
        return "Potential Loyalists"

    rfm["segment"] = rfm.apply(segment, axis=1)

    logger.info(f"RFM segments:\n{rfm['segment'].value_counts().to_string()}")
    return rfm


def cohort_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly cohort retention analysis.
    Tracks what % of users acquired in month M return in month M+1, M+2, etc.
    """
    logger.info("Running cohort retention analysis...")
    if "timestamp_dt" not in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s")

    df["cohort_month"] = df.groupby("user_id")["timestamp_dt"].transform("min").dt.to_period("M")
    df["event_month"]  = df["timestamp_dt"].dt.to_period("M")
    df["months_since"] = (df["event_month"] - df["cohort_month"]).apply(lambda x: x.n)

    cohort = df.groupby(["cohort_month", "months_since"])["user_id"].nunique().reset_index()
    cohort.columns = ["cohort_month", "months_since", "users"]

    cohort_size = cohort[cohort["months_since"] == 0].set_index("cohort_month")["users"]
    cohort["retention"] = cohort.apply(
        lambda row: row["users"] / cohort_size.get(row["cohort_month"], 1), axis=1
    ).round(4)

    return cohort


def ab_test_analysis(
    control: pd.Series,
    treatment: pd.Series,
    metric_name: str = "conversion_rate",
    alpha: float = 0.05,
) -> Dict:
    """
    Statistical A/B test analysis.
    Two-proportion z-test for conversion rates.
    Computes p-value, confidence interval, and lift.
    """
    n_ctrl = len(control)
    n_trt  = len(treatment)
    mean_ctrl = control.mean()
    mean_trt  = treatment.mean()

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(control, treatment)
    lift = (mean_trt - mean_ctrl) / (mean_ctrl + 1e-10)

    # 95% CI for difference
    se = np.sqrt(control.var()/n_ctrl + treatment.var()/n_trt)
    ci_lower = (mean_trt - mean_ctrl) - 1.96 * se
    ci_upper = (mean_trt - mean_ctrl) + 1.96 * se

    return {
        "metric": metric_name,
        "control_mean": round(float(mean_ctrl), 6),
        "treatment_mean": round(float(mean_trt), 6),
        "lift": round(float(lift), 4),
        "lift_pct": f"{lift*100:.2f}%",
        "p_value": round(float(p_value), 6),
        "significant": p_value < alpha,
        "confidence_interval": [round(float(ci_lower), 6), round(float(ci_upper), 6)],
        "sample_sizes": {"control": n_ctrl, "treatment": n_trt},
        "recommendation": "Ship treatment" if p_value < alpha and lift > 0 else
                          "Reject treatment" if p_value < alpha else "Inconclusive",
    }
