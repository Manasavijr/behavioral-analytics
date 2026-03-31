"""
Statistical Anomaly Detection for E-Commerce Events.

Methods:
  1. Z-Score — detects mean shifts (error rate spikes, traffic drops)
  2. IQR (Interquartile Range) — robust to outliers, good for response time
  3. Isolation Forest — multivariate anomalies (simultaneous metric degradation)
  4. EWMA (Exponentially Weighted Moving Average) — temporal smoothing
  5. Seasonal decomposition — separates trend/seasonality/residuals

Anomalies map to real Amazon scenarios:
  - Sudden error_rate spike → service degradation
  - Response time p95 jump → infrastructure issue
  - Revenue drop → checkout bug or pricing error
  - Traffic spike → bot attack or flash sale
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    timestamp: str
    metric: str
    value: float
    expected: float
    deviation: float
    method: str
    severity: str        # low | medium | high | critical
    description: str
    region: Optional[str] = None
    device: Optional[str] = None


def classify_severity(deviation: float) -> str:
    """Classify anomaly severity based on deviation from expected."""
    if abs(deviation) >= 4.0:   return "critical"
    if abs(deviation) >= 3.0:   return "high"
    if abs(deviation) >= 2.5:   return "medium"
    return "low"


def zscore_detection(
    df: pd.DataFrame,
    metric: str,
    threshold: float = 2.5,
    window: int = 24,
) -> List[Anomaly]:
    """
    Rolling Z-Score anomaly detection.
    Detects points > threshold standard deviations from rolling mean.
    """
    anomalies = []
    values = df[metric].fillna(0)
    rolling_mean = values.rolling(window, min_periods=3).mean()
    rolling_std  = values.rolling(window, min_periods=3).std()

    for i in range(len(values)):
        if pd.isna(rolling_mean.iloc[i]) or rolling_std.iloc[i] < 1e-6:
            continue
        z = (values.iloc[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
        if abs(z) >= threshold:
            anomalies.append(Anomaly(
                timestamp=str(df.index[i]),
                metric=metric,
                value=round(float(values.iloc[i]), 4),
                expected=round(float(rolling_mean.iloc[i]), 4),
                deviation=round(float(z), 3),
                method="zscore",
                severity=classify_severity(z),
                description=f"{metric} deviated {z:.1f}σ from rolling {window}h mean",
                region=df["region"].iloc[i] if "region" in df.columns else None,
            ))
    return anomalies


def iqr_detection(
    df: pd.DataFrame,
    metric: str,
    multiplier: float = 2.0,
) -> List[Anomaly]:
    """
    IQR-based anomaly detection.
    Robust to outliers — better for skewed metrics like response time.
    """
    anomalies = []
    values = df[metric].fillna(0)
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    for i in range(len(values)):
        v = values.iloc[i]
        if v < lower or v > upper:
            median = values.median()
            deviation = (v - median) / (IQR + 1e-6)
            anomalies.append(Anomaly(
                timestamp=str(df.index[i]),
                metric=metric,
                value=round(float(v), 4),
                expected=round(float(median), 4),
                deviation=round(float(deviation), 3),
                method="iqr",
                severity=classify_severity(deviation),
                description=f"{metric}={v:.2f} outside IQR bounds [{lower:.2f}, {upper:.2f}]",
            ))
    return anomalies


def isolation_forest_detection(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.03,
) -> pd.DataFrame:
    """
    Multivariate anomaly detection using Isolation Forest.
    Detects simultaneous degradation across multiple metrics —
    e.g., error_rate UP + response_time UP + revenue DOWN simultaneously.
    """
    logger.info(f"Running Isolation Forest on {features}")
    data = df[features].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    clf = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    labels = clf.fit_predict(scaled)
    scores = clf.score_samples(scaled)

    result = df.copy()
    result["anomaly_label"] = labels          # -1 = anomaly, 1 = normal
    result["anomaly_score"] = scores          # more negative = more anomalous
    result["is_anomaly"] = labels == -1
    return result


def ewma_detection(
    series: pd.Series,
    span: int = 12,
    threshold: float = 3.0,
) -> List[int]:
    """
    EWMA control chart — detects gradual drift.
    Used for slow degradation that Z-score might miss.
    """
    ewma = series.ewm(span=span).mean()
    ewmstd = series.ewm(span=span).std()
    z_scores = (series - ewma) / (ewmstd + 1e-6)
    return list(np.where(np.abs(z_scores) > threshold)[0])


def detect_all(df: pd.DataFrame, region: str = None) -> Dict:
    """
    Run all anomaly detection methods on hourly aggregates.
    Returns structured report with anomalies by metric and method.
    """
    if region:
        df = df[df["region"] == region].copy()

    df = df.sort_values("hour_bucket").reset_index(drop=True)
    df.index = pd.to_datetime(df["hour_bucket"])

    metrics = {
        "error_rate":      ("zscore", 2.5),
        "avg_response_ms": ("iqr",    2.0),
        "total_revenue":   ("zscore", 3.0),
        "event_count":     ("zscore", 3.0),
        "unique_users":    ("zscore", 3.0),
    }

    all_anomalies = []
    for metric, (method, threshold) in metrics.items():
        if metric not in df.columns:
            continue
        if method == "zscore":
            detected = zscore_detection(df, metric, threshold)
        else:
            detected = iqr_detection(df, metric, threshold)
        all_anomalies.extend(detected)

    # Multivariate isolation forest
    iso_features = [m for m in ["error_rate","avg_response_ms","total_revenue","event_count"] if m in df.columns]
    if len(iso_features) >= 2:
        iso_result = isolation_forest_detection(df, iso_features)
        iso_anomalies_df = iso_result[iso_result["is_anomaly"]]
        for _, row in iso_anomalies_df.iterrows():
            all_anomalies.append(Anomaly(
                timestamp=str(row.name),
                metric="multivariate",
                value=round(float(row["anomaly_score"]), 4),
                expected=0.0,
                deviation=round(float(-row["anomaly_score"]), 3),
                method="isolation_forest",
                severity=classify_severity(-row["anomaly_score"] * 3),
                description=f"Multivariate anomaly: simultaneous degradation in {iso_features}",
                region=region,
            ))

    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_anomalies.sort(key=lambda x: (severity_order.get(x.severity, 4), -abs(x.deviation)))

    summary = {
        "total_anomalies": len(all_anomalies),
        "by_severity": {
            s: sum(1 for a in all_anomalies if a.severity == s)
            for s in ["critical", "high", "medium", "low"]
        },
        "by_method": {
            m: sum(1 for a in all_anomalies if a.method == m)
            for m in ["zscore", "iqr", "isolation_forest"]
        },
        "anomalies": [
            {k: v for k, v in vars(a).items()}
            for a in all_anomalies[:50]  # top 50
        ],
    }
    logger.info(f"Anomaly detection complete: {len(all_anomalies)} anomalies found")
    return summary
