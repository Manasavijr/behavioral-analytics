"""
Behavioral Analytics Pipeline — FastAPI Backend + Business Dashboard
"""
import json
import logging
from typing import Optional
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from analytics.anomaly.detector import detect_all
from analytics.funnel.funnel_analysis import compute_funnel, rfm_segmentation
from storage.athena_sim.query_engine import AthenaQueryEngine, SAVED_QUERIES, QUERY_DESCRIPTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH", "data/processed")
_engine: AthenaQueryEngine = None
_hourly_df: pd.DataFrame = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _hourly_df
    logger.info("Starting Behavioral Analytics API...")
    _engine = AthenaQueryEngine(DATA_PATH)
    hourly_path = Path(DATA_PATH) / "hourly_aggregates.csv"
    if hourly_path.exists():
        _hourly_df = pd.read_csv(hourly_path)
        logger.info(f"Loaded {len(_hourly_df):,} hourly aggregate rows")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Behavioral Analytics Pipeline",
    description="10M+ event analytics: PySpark ETL, anomaly detection, funnel analysis, Athena SQL",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Dashboard ──────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Behavioral Analytics Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e2e8f0;font-family:Inter,system-ui,sans-serif}
.header{background:linear-gradient(135deg,#1a1f2e,#0f1117);border-bottom:1px solid #2d3748;padding:24px 40px}
h1{font-size:22px;font-weight:700}
.subtitle{color:#718096;font-size:13px;margin-top:5px}
.badge{display:inline-block;background:#1a1f2e;color:#b794f4;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;margin-top:8px;margin-right:6px;border:1px solid #2d3748}
.container{max-width:1300px;margin:0 auto;padding:28px 40px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:22px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:22px}
.card{background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:20px}
.card-title{font-size:12px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.05em;margin-bottom:14px}
.kpi-num{font-size:32px;font-weight:700}
.kpi-label{font-size:11px;color:#718096;margin-top:4px}
.green{color:#68d391}.red{color:#fc8181}.yellow{color:#f6e05e}.purple{color:#b794f4}.blue{color:#63b3ed}.teal{color:#4fd1c5}
.anomaly-item{background:#141822;border-radius:7px;padding:10px 14px;margin-bottom:7px;border-left:3px solid #fc8181;font-size:13px}
.anomaly-item.high{border-color:#fc8181}
.anomaly-item.medium{border-color:#f6e05e}
.anomaly-item.low{border-color:#63b3ed}
.anomaly-item.critical{border-color:#fc2f2f;background:#3a1010}
.tag{padding:2px 7px;border-radius:8px;font-size:10px;font-weight:700}
.tag-red{background:#3a1010;color:#fc8181}
.tag-yellow{background:#3a3010;color:#f6e05e}
.tag-blue{background:#1a2a3a;color:#63b3ed}
.tab-bar{display:flex;gap:4px;border-bottom:1px solid #2d3748;margin-bottom:20px}
.tab{padding:9px 16px;background:none;border:none;cursor:pointer;color:#718096;font-size:13px;border-bottom:2px solid transparent}
.tab.active{color:#b794f4;border-bottom-color:#b794f4;font-weight:600}
.sql-area{width:100%;background:#0f1117;border:1px solid #2d3748;border-radius:7px;padding:12px;color:#e2e8f0;font-family:monospace;font-size:13px;min-height:80px;resize:vertical}
.btn{padding:9px 18px;border-radius:7px;border:none;background:#b794f4;color:#1a1f2e;font-weight:700;cursor:pointer;font-size:13px}
.result-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:12px}
.result-table th{background:#141822;padding:8px 12px;text-align:left;color:#718096;font-weight:600;border-bottom:1px solid #2d3748}
.result-table td{padding:7px 12px;border-bottom:1px solid #141822}
.result-table tr:hover td{background:#141822}
canvas{max-height:220px}
.funnel-bar{height:32px;border-radius:5px;background:linear-gradient(90deg,#b794f4,#63b3ed);margin-bottom:8px;display:flex;align-items:center;padding:0 10px;font-size:12px;font-weight:600;color:#1a1f2e;transition:width .8s ease}
.segment-pill{display:inline-block;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;margin:4px}
</style></head>
<body>
<div class="header">
  <h1>📊 Behavioral Analytics Pipeline</h1>
  <p class="subtitle">10M+ E-Commerce Events · PySpark ETL · Anomaly Detection · Funnel Analysis · Athena SQL</p>
  <div>
    <span class="badge">PySpark</span><span class="badge">AWS S3</span><span class="badge">Athena SQL</span>
    <span class="badge">Anomaly Detection</span><span class="badge">Causal Inference</span><span class="badge">RFM</span>
  </div>
</div>
<div class="container">
  <!-- KPI Row -->
  <div class="grid4" id="kpi_row">
    <div class="card"><div class="kpi-num purple" id="kpi_events">—</div><div class="kpi-label">Total Events</div></div>
    <div class="card"><div class="kpi-num green" id="kpi_revenue">—</div><div class="kpi-label">Total Revenue</div></div>
    <div class="card"><div class="kpi-num red" id="kpi_errors">—</div><div class="kpi-label">Error Rate</div></div>
    <div class="card"><div class="kpi-num blue" id="kpi_response">—</div><div class="kpi-label">Avg Response (ms)</div></div>
  </div>

  <!-- Tabs -->
  <div class="tab-bar">
    <button class="tab active" onclick="showTab('overview')">Overview</button>
    <button class="tab" onclick="showTab('anomalies')">Anomalies</button>
    <button class="tab" onclick="showTab('funnel')">Funnel</button>
    <button class="tab" onclick="showTab('sql')">SQL Query</button>
  </div>

  <!-- Overview Tab -->
  <div id="tab_overview">
    <div class="grid2">
      <div class="card"><div class="card-title">Hourly Event Volume</div><canvas id="events_chart"></canvas></div>
      <div class="card"><div class="card-title">Hourly Revenue</div><canvas id="revenue_chart"></canvas></div>
    </div>
    <div class="grid2">
      <div class="card"><div class="card-title">Error Rate Over Time</div><canvas id="error_chart"></canvas></div>
      <div class="card"><div class="card-title">Response Time P95</div><canvas id="rt_chart"></canvas></div>
    </div>
  </div>

  <!-- Anomalies Tab -->
  <div id="tab_anomalies" style="display:none">
    <div class="grid2">
      <div class="card">
        <div class="card-title">Detected Anomalies</div>
        <div id="anomaly_list">Loading...</div>
      </div>
      <div class="card">
        <div class="card-title">Anomaly Summary</div>
        <div id="anomaly_summary"></div>
      </div>
    </div>
  </div>

  <!-- Funnel Tab -->
  <div id="tab_funnel" style="display:none">
    <div class="grid2">
      <div class="card">
        <div class="card-title">Conversion Funnel</div>
        <div id="funnel_bars">Loading...</div>
      </div>
      <div class="card">
        <div class="card-title">RFM Customer Segments</div>
        <div id="rfm_segments"></div>
        <canvas id="rfm_chart" style="margin-top:16px"></canvas>
      </div>
    </div>
  </div>

  <!-- SQL Tab -->
  <div id="tab_sql" style="display:none">
    <div class="card">
      <div class="card-title">Athena-Style Saved Queries</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px">
        <button class="btn" onclick="runSaved('daily_revenue')">Daily Revenue by Region</button>
        <button class="btn" onclick="runSaved('error_rate_by_hour')">Error Rate by Hour</button>
        <button class="btn" onclick="runSaved('funnel_conversion')">Funnel Conversion</button>
        <button class="btn" onclick="runSaved('top_categories')">Top Categories</button>
        <button class="btn" onclick="runSaved('hourly_kpis')" style="grid-column:span 2">Hourly KPIs</button>
      </div>
      <div id="query_result"></div>
    </div>
  </div>
</div>

<script>
const chartDefaults = {
  responsive:true,
  plugins:{legend:{labels:{color:'#e2e8f0',font:{size:11}}}},
  scales:{x:{ticks:{color:'#718096',maxTicksLimit:8},grid:{color:'#2d3748'}},
          y:{ticks:{color:'#718096'},grid:{color:'#2d3748'}}}
};
let charts={};

function showTab(name){
  ['overview','anomalies','funnel','sql'].forEach(t=>{
    document.getElementById('tab_'+t).style.display = t===name?'block':'none';
    document.querySelectorAll('.tab').forEach((b,i)=>{
      b.classList.toggle('active', ['overview','anomalies','funnel','sql'][i]===name);
    });
  });
  if(name==='anomalies') loadAnomalies();
  if(name==='funnel') loadFunnel();
}

async function loadKPIs(){
  try{
    const r = await fetch('/api/kpis'); const d = await r.json();
    document.getElementById('kpi_events').textContent = (d.total_events/1e6).toFixed(1)+'M';
    document.getElementById('kpi_revenue').textContent = '$'+(d.total_revenue/1e6).toFixed(2)+'M';
    document.getElementById('kpi_errors').textContent = (d.error_rate*100).toFixed(2)+'%';
    document.getElementById('kpi_response').textContent = Math.round(d.avg_response_ms)+'ms';
  }catch(e){console.error(e)}
}

async function loadCharts(){
  try{
    const r = await fetch('/api/timeseries'); const d = await r.json();
    const labels = d.labels.slice(-48);
    const mkChart = (id,label,data,color,yLabel='')=>{
      if(charts[id]) charts[id].destroy();
      charts[id] = new Chart(document.getElementById(id),{
        type:'line',
        data:{labels,datasets:[{label,data:data.slice(-48),borderColor:color,
          backgroundColor:color+'22',tension:0.3,fill:true,pointRadius:1}]},
        options:{...chartDefaults,plugins:{...chartDefaults.plugins,
          legend:{labels:{color:'#e2e8f0',font:{size:11}}}}}
      });
    };
    mkChart('events_chart','Events/hr',d.event_count,'#b794f4');
    mkChart('revenue_chart','Revenue ($)',d.revenue,'#68d391');
    mkChart('error_chart','Error Rate',d.error_rate,'#fc8181');
    mkChart('rt_chart','P95 Response (ms)',d.p95_response,'#63b3ed');
  }catch(e){console.error(e)}
}

async function loadAnomalies(){
  try{
    const r = await fetch('/api/anomalies'); const d = await r.json();
    const list = document.getElementById('anomaly_list');
    const top = (d.anomalies||[]).slice(0,15);
    list.innerHTML = top.map(a=>`
      <div class="anomaly-item ${a.severity}">
        <div style="display:flex;justify-content:space-between;margin-bottom:3px">
          <strong>${a.metric}</strong>
          <span class="tag tag-${a.severity==='critical'||a.severity==='high'?'red':a.severity==='medium'?'yellow':'blue'}">${a.severity.toUpperCase()}</span>
        </div>
        <div style="color:#718096;font-size:11px">${a.description}</div>
        <div style="color:#718096;font-size:11px">val=${a.value} · expected=${a.expected} · method=${a.method}</div>
      </div>`).join('');
    const sum = document.getElementById('anomaly_summary');
    sum.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        ${Object.entries(d.by_severity||{}).map(([s,c])=>`
          <div style="text-align:center;padding:14px;background:#141822;border-radius:8px">
            <div style="font-size:28px;font-weight:700;color:${s==='critical'?'#fc2f2f':s==='high'?'#fc8181':s==='medium'?'#f6e05e':'#63b3ed'}">${c}</div>
            <div style="font-size:11px;color:#718096;margin-top:3px">${s.toUpperCase()}</div>
          </div>`).join('')}
      </div>
      <div style="margin-top:14px;font-size:13px;color:#718096">Total: ${d.total_anomalies} anomalies detected</div>`;
  }catch(e){document.getElementById('anomaly_list').textContent='Run pipeline first to see anomalies'}
}

async function loadFunnel(){
  try{
    const r = await fetch('/api/funnel'); const d = await r.json();
    const stages = d.overall?.stages || [];
    const maxUsers = stages[0]?.users || 1;
    document.getElementById('funnel_bars').innerHTML = stages.map(s=>`
      <div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
          <span>${s.stage}</span>
          <span style="color:#718096">${s.users?.toLocaleString()} users · ${(s.conversion_from_top*100).toFixed(1)}%</span>
        </div>
        <div style="background:#141822;border-radius:5px;overflow:hidden;height:28px">
          <div class="funnel-bar" style="width:${(s.users/maxUsers*100).toFixed(1)}%;height:100%">
            ${(s.dropoff_rate*100).toFixed(1)}% drop
          </div>
        </div>
      </div>`).join('');
  }catch(e){document.getElementById('funnel_bars').textContent='Run pipeline first'}
}

async function runSaved(name){
  document.getElementById('query_result').innerHTML = '<div style="color:#718096;margin-top:10px">Running...</div>';
  try{
    const r = await fetch(`/api/query/${name}`);
    const d = await r.json();
    if(d.error){document.getElementById('query_result').innerHTML=`<div style="color:#fc8181;margin-top:10px">${d.error}</div>`;return;}
    const rows = d.rows||[];const cols = d.columns||[];
    document.getElementById('query_result').innerHTML = `
      <div style="color:#718096;font-size:11px;margin-top:8px">${rows.length} rows · ${d.elapsed_ms}ms</div>
      <div style="overflow-x:auto"><table class="result-table">
        <tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>
        ${rows.slice(0,50).map(row=>`<tr>${cols.map(c=>`<td>${row[c]??''}</td>`).join('')}</tr>`).join('')}
      </table></div>`;
  }catch(e){document.getElementById('query_result').innerHTML=`<div style="color:#fc8181;margin-top:10px">${e.message}</div>`}
}

loadKPIs(); loadCharts();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/api/kpis")
async def get_kpis():
    if _hourly_df is None:
        raise HTTPException(503, "Run pipeline first: python pipeline/processing/spark_pipeline.py")
    return {
        "total_events": int(_hourly_df["event_count"].sum()),
        "total_revenue": float(_hourly_df["total_revenue"].sum()),
        "error_rate": float(_hourly_df["error_rate"].mean()),
        "avg_response_ms": float(_hourly_df["avg_response_ms"].mean()),
        "unique_regions": int(_hourly_df["region"].nunique()) if "region" in _hourly_df.columns else 0,
    }


@app.get("/api/timeseries")
async def get_timeseries():
    if _hourly_df is None:
        raise HTTPException(503, "Run pipeline first")
    df = _hourly_df.groupby("hour_bucket").agg({
        "event_count": "sum", "total_revenue": "sum",
        "error_rate": "mean", "p95_response_ms": "mean",
    }).reset_index().sort_values("hour_bucket").head(168)  # 1 week
    return {
        "labels": df["hour_bucket"].astype(str).tolist(),
        "event_count": df["event_count"].tolist(),
        "revenue": df["total_revenue"].tolist(),
        "error_rate": df["error_rate"].tolist(),
        "p95_response": df["p95_response_ms"].fillna(0).tolist(),
    }


@app.get("/api/anomalies")
async def get_anomalies(region: Optional[str] = Query(None)):
    if _hourly_df is None:
        raise HTTPException(503, "Run pipeline first")
    result = detect_all(_hourly_df, region=region)
    return result


@app.get("/api/funnel")
async def get_funnel():
    raw_path = Path("data/raw/events")
    if not raw_path.exists():
        raise HTTPException(503, "Run data generation first: python pipeline/ingestion/generate_events.py")
    try:
        df = pd.read_parquet(raw_path)
        return compute_funnel(df)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/query/{name}")
async def run_saved_query(name: str):
    if _engine is None:
        return {"error": "Query engine not initialized"}
    import time
    t0 = time.time()
    try:
        result = _engine.run_saved_query(name)
        return {
            "columns": result.columns.tolist(),
            "rows": result.to_dict(orient="records"),
            "elapsed_ms": round((time.time()-t0)*1000, 1),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8083, reload=True)
