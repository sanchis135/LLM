import os
import json
import sqlite3
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st

# --- Config ---
DEFAULT_DB = "monitoring/feedback.sqlite"
DB_PATH = Path(os.getenv("FEEDBACK_DB", DEFAULT_DB))

st.set_page_config(page_title="ModelOpsRAG – Monitoring Dashboard", layout="wide")
st.title("ModelOpsRAG – Monitoring & Feedback Dashboard")

# --- Helpers ---
def load_events(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        st.warning(f"No feedback DB found at: {db_path}. Interact with the app to generate feedback.")
        return pd.DataFrame()
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM events ORDER BY ts ASC", con)
    finally:
        con.close()
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    # unroll payload JSON into columns
    def parse_payload(s):
        try:
            return json.loads(s or "{}")
        except Exception:
            return {}
    payload = df["payload"].apply(parse_payload)
    df["useful"] = payload.apply(lambda x: x.get("useful"))
    df["note"] = payload.apply(lambda x: x.get("note"))
    df["query"] = payload.apply(lambda x: x.get("query"))
    return df

STOPWORDS = {
    "the","a","an","and","or","of","to","for","in","on","at","with","how","what","is","are",
    "que","como","de","la","el","los","las","y","o","en","por","para","con","un","una","del",
    "rollout","deployment","deploy","model","models"  # quita términos demasiado frecuentes en este dominio
}

def top_keywords(queries: pd.Series, n=10):
    words = []
    for q in queries.dropna():
        for w in str(q).lower().replace("/", " ").replace("-", " ").split():
            w = "".join(ch for ch in w if ch.isalnum())
            if w and w not in STOPWORDS and len(w) > 2:
                words.append(w)
    return Counter(words).most_common(n)

# --- Load data ---
df = load_events(DB_PATH)
if df.empty:
    st.info("No events recorded yet.")
    st.stop()

# Focus on explicit feedback events
feedback = df[df["event"] == "eval_feedback"].copy()
if feedback.empty:
    st.info("No user feedback events yet. Submit feedback from the main app to populate this dashboard.")
    st.dataframe(df.tail(20))
    st.stop()

# Normalize values
feedback["useful_norm"] = feedback["useful"].str.lower().map({"yes": 1, "no": 0})

# --- KPIs row ---
colA, colB, colC, colD = st.columns(4)
total_fb = len(feedback)
helpful = int(feedback["useful_norm"].sum()) if "useful_norm" in feedback else 0
rate = (helpful / total_fb * 100.0) if total_fb else 0.0
unique_queries = feedback["query"].dropna().nunique()

colA.metric("Total feedback", f"{total_fb}")
colB.metric("Helpful rate", f"{rate:.1f}%")
colC.metric("Unique queries", f"{unique_queries}")
colD.metric("DB path", DB_PATH.as_posix())

st.divider()

# --- Row 1: time series + helpful vs not ---
c1, c2 = st.columns([2,1])

with c1:
    st.subheader("Activity over time (feedback per day)")
    per_day = feedback.groupby(feedback["dt"].dt.date).size().rename("count").to_frame()
    st.line_chart(per_day)

with c2:
    st.subheader("Helpful vs Not")
    hvn = feedback["useful"].fillna("Unknown").value_counts().rename_axis("useful").to_frame("count")
    st.bar_chart(hvn)

st.divider()

# --- Row 2: top queries + rolling helpful rate ---
c3, c4 = st.columns([2,1])

with c3:
    st.subheader("Top queries")
    top_q = feedback["query"].dropna().value_counts().head(10).rename_axis("query").to_frame("count")
    st.dataframe(top_q)

with c4:
    st.subheader("7-day rolling helpful rate")
    # resample daily and take mean of useful_norm
    daily = feedback.set_index("dt").resample("D")["useful_norm"].mean().fillna(0.0)
    rolling = daily.rolling(window=7, min_periods=1).mean().to_frame("helpful_rate")
    st.line_chart(rolling)

st.divider()

# --- Row 3: top keywords + recent feedback table ---
c5, c6 = st.columns([1,2])

with c5:
    st.subheader("Top keywords (from queries)")
    kw = top_keywords(feedback["query"], n=12)
    if kw:
        kw_df = pd.DataFrame(kw, columns=["keyword","count"]).set_index("keyword")
        st.bar_chart(kw_df)
    else:
        st.info("Not enough query text to extract keywords yet.")

with c6:
    st.subheader("Recent feedback")
    recent = feedback[["dt","query","useful","note"]].sort_values("dt", ascending=False).head(20)
    st.dataframe(recent, use_container_width=True)

st.markdown("---")
st.caption("Tip: leave feedback from the main app (Yes/No + optional note) to see this dashboard update in real time.")
