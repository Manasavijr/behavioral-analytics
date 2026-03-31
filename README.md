# Large-Scale Behavioral Analytics Pipeline

Production-grade analytics pipeline processing 10M+ e-commerce events with PySpark, statistical anomaly detection, causal inference root cause analysis, funnel analysis, and an AWS S3 + Athena querying layer — served via FastAPI business dashboard.

Built to Amazon's data engineering scale and patterns.

---

## Architecture

```
Raw Events (10M+)
       │
       ▼
┌─────────────────────────────────────┐
│   PySpark ETL Pipeline              │
│   • Schema validation + DQ checks   │
│   • Window-based sessionization      │
│   • RFM + engagement features        │
│   • Hourly KPI aggregation           │
│   • Partitioned Parquet output       │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
S3 Data Lake         Athena SQL Layer
(moto mock /         (DuckDB local /
 real AWS)           real Athena)
    │                     │
    └──────────┬──────────┘
               ▼
┌──────────────────────────────────────┐
│   Analytics Layer                    │
│   • Z-Score + IQR anomaly detection  │
│   • Isolation Forest (multivariate)  │
│   • Granger causality RCA            │
│   • Conversion funnel analysis       │
│   • RFM customer segmentation        │
│   • A/B test statistical analysis    │
└──────────────┬───────────────────────┘
               ▼
    FastAPI Business Dashboard
    (live charts, SQL console, anomaly feed)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Distributed Processing | PySpark 3.5 |
| Cloud Storage | AWS S3 (boto3 + moto mock) |
| Query Layer | DuckDB (Athena-compatible SQL) |
| Anomaly Detection | Z-Score, IQR, Isolation Forest |
| Causal Inference | Granger causality, Pearson correlation |
| Stats | SciPy, statsmodels |
| Dashboard | FastAPI + Chart.js |

---

## Setup

```bash
cd behavioral-analytics
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run Pipeline

```bash
# Step 1: Generate 10M events (~2 min)
python pipeline/ingestion/generate_events.py

# Step 2: PySpark ETL (~3-5 min, needs JAVA_HOME set)
export JAVA_HOME=$(brew --prefix openjdk@17)
python pipeline/processing/spark_pipeline.py

# Step 3: Start dashboard
uvicorn api.main:app --reload --port 8083
# Open http://localhost:8083
```

## Saved SQL Queries (Athena-style)

```sql
-- Daily revenue by region
SELECT DATE_TRUNC('day', timestamp_dt) AS date, region, SUM(revenue) AS revenue
FROM events GROUP BY 1,2 ORDER BY date;

-- Error rate by hour (for anomaly investigation)
SELECT DATE_TRUNC('hour', timestamp_dt) AS hour,
       SUM(CASE WHEN event_type='error' THEN 1.0 ELSE 0.0 END)/COUNT(*) AS error_rate
FROM events GROUP BY 1 ORDER BY 1;
```

## Run Tests

```bash
pytest tests/ -v
```

---

## Results

| Metric | Value |
|---|---|
| Events processed | 10M+ |
| PySpark processing time | ~3-5 min |
| Anomaly detection methods | Z-Score, IQR, Isolation Forest |
| Funnel stages | 6 (page_view → purchase) |
| Customer segments | 6 RFM segments |
| SQL queries | 5 saved Athena-style queries |
