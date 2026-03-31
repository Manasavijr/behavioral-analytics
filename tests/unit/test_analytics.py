"""Unit tests for behavioral analytics pipeline."""
import numpy as np
import pandas as pd
import pytest


def make_hourly_df(n=100):
    """Create synthetic hourly aggregates for testing."""
    np.random.seed(42)
    hours = pd.date_range("2024-01-01", periods=n, freq="H")
    df = pd.DataFrame({
        "hour_bucket": hours,
        "region": np.random.choice(["us-east-1","us-west-2","eu-west-1"], n),
        "device": np.random.choice(["mobile","desktop"], n),
        "event_count": np.random.randint(1000, 50000, n),
        "unique_users": np.random.randint(100, 5000, n),
        "unique_sessions": np.random.randint(150, 6000, n),
        "purchases": np.random.randint(10, 500, n),
        "total_revenue": np.random.uniform(1000, 50000, n),
        "avg_response_ms": np.random.lognormal(5.5, 0.3, n),
        "p95_response_ms": np.random.lognormal(6.5, 0.4, n),
        "error_count": np.random.randint(0, 100, n),
        "error_rate": np.random.uniform(0.005, 0.04, n),
        "conversions": np.random.randint(5, 200, n),
    })
    # Inject spike
    df.loc[50, "error_rate"] = 0.35
    df.loc[50, "avg_response_ms"] = 8000
    return df


def make_events_df(n=5000):
    np.random.seed(42)
    events = np.random.choice(
        ["page_view","search","product_view","add_to_cart","checkout_start","purchase","error"],
        n, p=[0.35,0.18,0.22,0.09,0.05,0.04,0.07]
    )
    return pd.DataFrame({
        "event_id": [f"E{i}" for i in range(n)],
        "user_id": np.random.choice([f"U{i}" for i in range(500)], n),
        "session_id": np.random.choice([f"S{i}" for i in range(1000)], n),
        "event_type": events,
        "timestamp": np.random.uniform(1704067200, 1711929600, n),
        "timestamp_dt": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "region": np.random.choice(["us-east-1","eu-west-1"], n),
        "device": np.random.choice(["mobile","desktop"], n),
        "category": np.random.choice(["Electronics","Clothing","Books"], n),
        "product_id": np.random.choice([f"P{i}" for i in range(100)], n),
        "price": np.random.lognormal(3.5, 1.0, n).round(2),
        "revenue": np.where(events=="purchase", np.random.uniform(10, 500, n), 0),
        "response_time_ms": np.random.lognormal(5.5, 0.5, n).astype(int),
        "error_code": np.where(events=="error", "500", "200"),
        "search_query": "",
        "hour": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
    })


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def test_zscore_detects_spike():
    from analytics.anomaly.detector import zscore_detection
    df = make_hourly_df(100)
    df.index = pd.to_datetime(df["hour_bucket"])
    anomalies = zscore_detection(df, "error_rate", threshold=2.0)
    assert len(anomalies) > 0
    # The injected spike should be detected
    spike_anomalies = [a for a in anomalies if a.value > 0.1]
    assert len(spike_anomalies) > 0


def test_iqr_detection():
    from analytics.anomaly.detector import iqr_detection
    df = make_hourly_df(100)
    df.index = pd.to_datetime(df["hour_bucket"])
    anomalies = iqr_detection(df, "avg_response_ms", multiplier=1.5)
    assert isinstance(anomalies, list)


def test_isolation_forest():
    from analytics.anomaly.detector import isolation_forest_detection
    df = make_hourly_df(100)
    result = isolation_forest_detection(df, ["error_rate","avg_response_ms","total_revenue"])
    assert "is_anomaly" in result.columns
    assert "anomaly_score" in result.columns
    assert result["is_anomaly"].any()


def test_severity_classification():
    from analytics.anomaly.detector import classify_severity
    assert classify_severity(4.5) == "critical"
    assert classify_severity(3.2) == "high"
    assert classify_severity(2.7) == "medium"
    assert classify_severity(1.5) == "low"


def test_detect_all():
    from analytics.anomaly.detector import detect_all
    df = make_hourly_df(100)
    result = detect_all(df)
    assert "total_anomalies" in result
    assert "by_severity" in result
    assert "anomalies" in result


# ── Funnel Analysis ───────────────────────────────────────────────────────────

def test_funnel_conversion():
    from analytics.funnel.funnel_analysis import compute_funnel
    df = make_events_df(5000)
    result = compute_funnel(df)
    assert "overall" in result
    stages = result["overall"]["stages"]
    assert len(stages) > 0
    assert all("conversion_from_top" in s for s in stages)


def test_rfm_segmentation():
    from analytics.funnel.funnel_analysis import rfm_segmentation
    df = make_events_df(5000)
    rfm = rfm_segmentation(df)
    assert "segment" in rfm.columns
    assert "rfm_score" in rfm.columns
    assert set(rfm["segment"].unique()).issubset(
        {"Champions","Loyal Customers","New Customers","At Risk","Lost","Potential Loyalists"}
    )


def test_ab_test():
    from analytics.funnel.funnel_analysis import ab_test_analysis
    np.random.seed(42)
    control = np.random.binomial(1, 0.10, 1000).astype(float)
    treatment = np.random.binomial(1, 0.13, 1000).astype(float)
    result = ab_test_analysis(control, treatment, "conversion_rate")
    assert "p_value" in result
    assert "lift" in result
    assert "significant" in result
    assert "recommendation" in result


# ── Causal Inference ──────────────────────────────────────────────────────────

def test_correlate_metrics():
    from analytics.causal.root_cause import correlate_metrics
    df = make_hourly_df(100)
    result = correlate_metrics(df, "error_rate", ["avg_response_ms","total_revenue","event_count"])
    assert len(result) > 0
    for feat, corr in result.items():
        assert "pearson_r" in corr
        assert "p_value" in corr
        assert -1 <= corr["pearson_r"] <= 1


def test_rca_report():
    from analytics.causal.root_cause import generate_rca_report
    df = make_hourly_df(100)
    anomaly = {"metric": "error_rate", "timestamp": "2024-01-01 02:00", "value": 0.35, "severity": "critical"}
    report = generate_rca_report(df, anomaly)
    assert "hypotheses" in report
    assert "recommended_actions" in report
    assert "top_correlations" in report
    assert "summary" in report
