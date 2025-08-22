# app/ui_streamlit/pages/1_Metrics.py
from __future__ import annotations
import sys, pathlib, sqlite3, time
import pandas as pd
import numpy as np
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))
from monitoring.logger import DB_PATH

st.set_page_config(page_title="📈 Metrics & Monitoring", layout="wide")
st.title("📈 Metrics & Monitoring")

@st.cache_data(show_spinner=False, ttl=10)
def load_interactions(limit: int = 10000) -> pd.DataFrame:
    db = pathlib.Path(DB_PATH)
    if not db.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(db)) as conn:
        # Carga todo y deja que abajo añadamos columnas faltantes
        df = pd.read_sql_query("SELECT * FROM interactions ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    # Normaliza columnas que podrían faltar
    for col in ["latency_ms_retrieval", "latency_ms_llm", "ctx_len", "feedback"]:
        if col not in df.columns:
            df[col] = pd.NA
    # Tipos seguros
    df["feedback"] = df["feedback"].apply(lambda v: int(v) if pd.notna(v) else 0)
    # Timestamps legibles
    if "ts_utc" in df.columns:
        df["ts"] = df["ts_utc"].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(x)) if pd.notna(x) else "")
        df["date"] = df["ts_utc"].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)) if pd.notna(x) else "")
    else:
        df["ts"] = ""
        df["date"] = ""
    return df

df = load_interactions()

if df.empty:
    st.info("No interactions recorded yet.")
    st.caption(f"DB path: {DB_PATH.resolve()}")
    st.stop()

# ======= Filtros globales =======
with st.sidebar:
    st.header("Filters")
    providers = ["All"] + sorted([x for x in df["provider"].dropna().unique()])
    models = ["All"] + sorted([x for x in df["model"].dropna().unique()])
    provider_sel = st.selectbox("Provider", providers, index=0)
    model_sel = st.selectbox("Model", models, index=0)
    date_min, date_max = st.date_input(
        "Date range",
        value=(pd.to_datetime(df["date"]).min(), pd.to_datetime(df["date"]).max())
    )

def within_date(s: str) -> bool:
    if not s: return True
    d = pd.to_datetime(s).date()
    return (d >= date_min) and (d <= date_max)

mask = df["date"].apply(within_date)
if provider_sel != "All":
    mask &= df["provider"] == provider_sel
if model_sel != "All":
    mask &= df["model"] == model_sel
df_f = df[mask].copy()

# ======= KPIs =======
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Interactions", len(df_f))
c2.metric("👍 Likes", int((df_f["feedback"] == 1).sum()))
c3.metric("👎 Dislikes", int((df_f["feedback"] == -1).sum()))
fb_rate = (df_f["feedback"] != 0).mean() * 100 if len(df_f) else 0
c4.metric("Feedback rate", f"{fb_rate:.1f}%")
lat_med = df_f["latency_ms"].median() if "latency_ms" in df_f.columns and len(df_f) else np.nan
c5.metric("Median latency (ms)", f"{lat_med:.0f}" if pd.notna(lat_med) else "—")

st.divider()

# ==========================================================
# 1) Time series: likes vs dislikes por día
# ==========================================================
st.subheader("1) Likes vs Dislikes over time")
ts = (
    df_f.groupby(["date","feedback"])["id"]
    .count()
    .reset_index()
    .pivot(index="date", columns="feedback", values="id")
    .fillna(0)
    .rename(columns={-1:"Dislikes", 0:"No feedback", 1:"Likes"})
    .sort_index()
)
if not ts.empty:
    st.line_chart(ts[["Likes","Dislikes"]])
else:
    st.caption("No data in current filter.")

# ==========================================================
# 2) Top queries by feedback (likes/dislikes)
# ==========================================================
st.subheader("2) Top queries by feedback")
left, right = st.columns(2)

# Likes
top_likes = (
    df_f[df_f["feedback"] == 1]
    .groupby("query")["id"].count().sort_values(ascending=False).head(10)
    .rename("Likes")
)
with left:
    st.markdown("**Most liked queries**")
    if not top_likes.empty:
        st.bar_chart(top_likes)
    else:
        st.caption("No likes in current filter.")

# Dislikes
top_dislikes = (
    df_f[df_f["feedback"] == -1]
    .groupby("query")["id"].count().sort_values(ascending=False).head(10)
    .rename("Dislikes")
)
with right:
    st.markdown("**Most disliked queries**")
    if not top_dislikes.empty:
        st.bar_chart(top_dislikes)
    else:
        st.caption("No dislikes in current filter.")

# ==========================================================
# 3) Feedback rate by model/provider
# ==========================================================
st.subheader("3) Feedback rate by model / provider")
grp = df_f.copy()
grp["has_feedback"] = (grp["feedback"] != 0).astype(int)
by_model = grp.groupby(["provider","model"]).agg(
    interactions=("id","count"),
    fb_rate=("has_feedback","mean"),
    likes=("feedback", lambda s: (s==1).sum()),
    dislikes=("feedback", lambda s: (s==-1).sum()),
).reset_index()
if not by_model.empty:
    by_model_display = by_model.copy()
    by_model_display["fb_rate"] = (by_model_display["fb_rate"]*100).round(1).astype(str) + "%"
    st.dataframe(by_model_display, use_container_width=True, hide_index=True)
else:
    st.caption("No data in current filter.")

# ==========================================================
# 4) Latency distribution
# ==========================================================
st.subheader("4) Latency distribution (ms)")
lat_series = df_f["latency_ms"].dropna().astype(float)
if not lat_series.empty:
    st.bar_chart(lat_series, height=180)
    st.caption(f"Median: {lat_series.median():.0f} ms • P90: {lat_series.quantile(0.9):.0f} ms")
else:
    st.caption("No latency data in current filter.")

# ==========================================================
# 5) Context length vs latency (scatter-ish via binning)
# ==========================================================
st.subheader("5) Context length vs. latency (binned)")
if "ctx_len" in df_f.columns and "latency_ms" in df_f.columns:
    df_bin = df_f.dropna(subset=["ctx_len","latency_ms"]).copy()
    if not df_bin.empty:
        df_bin["ctx_len"] = df_bin["ctx_len"].astype(int)
        agg = df_bin.groupby("ctx_len")["latency_ms"].median().reset_index()
        agg = agg.sort_values("ctx_len")
        st.line_chart(agg.set_index("ctx_len"))
        st.caption("Median latency by number of context passages.")
    else:
        st.caption("No ctx_len/latency data in current filter.")
else:
    st.caption("ctx_len / latency columns not found.")
