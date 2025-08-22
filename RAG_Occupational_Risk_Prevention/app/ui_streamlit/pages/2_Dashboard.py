from __future__ import annotations
import sys, pathlib, time, io, csv
import streamlit as st

# Ensure repo package import
ROOT = pathlib.Path(__file__).resolve().parents[3]  # .../rag-prl
sys.path.append(str(ROOT))

from monitoring.logger import recent, DB_PATH  # DB_PATH is useful to show the DB location

st.set_page_config(page_title="RAG-PRL — Dashboard", page_icon="📊", layout="wide")
st.title("📊 Interaction Dashboard")

# Top bar: refresh + filters
col_refresh, col_q, col_fb = st.columns([1, 3, 2])
with col_refresh:
    if st.button("🔄 Refresh"):
        st.rerun()


with col_q:
    q_filter = st.text_input("Filter by query text", value="")

with col_fb:
    fb_filter = st.selectbox("Filter by feedback", options=["All", "👍 Like", "👎 Dislike", "No feedback"])

# Load recent data
rows = recent(300)  # fetch more to allow filtering
if not rows:
    st.info("No interactions recorded yet.")
    st.caption(f"DB path: {DB_PATH.resolve()}")
else:
    # Apply filters
    def norm_fb(v):
        # Normalize feedback to int: 1 / -1 / 0
        if v is None:
            return 0
        try:
            return int(v)
        except Exception:
            # handle strings like "1", "-1", "0", or anything odd
            s = str(v).strip()
            if s in {"1", "+1"}:
                return 1
            if s in {"-1"}:
                return -1
            return 0

    def match_row(r):
        ok_q = (q_filter.lower() in r.get("query", "").lower()) if q_filter else True
        fbi = norm_fb(r.get("feedback"))
        if fb_filter == "👍 Like":
            ok_fb = (fbi == 1)
        elif fb_filter == "👎 Dislike":
            ok_fb = (fbi == -1)
        elif fb_filter == "No feedback":
            ok_fb = (fbi == 0)
        else:
            ok_fb = True
        return ok_q and ok_fb

    rows_f = [r for r in rows if match_row(r)]

    # KPIs
    total = len(rows_f)
    likes = sum(1 for r in rows_f if norm_fb(r.get("feedback")) == 1)
    dislikes = sum(1 for r in rows_f if norm_fb(r.get("feedback")) == -1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Interactions (filtered)", total)
    col2.metric("👍 Likes", likes)
    col3.metric("👎 Dislikes", dislikes)
    col4.caption(f"DB: {DB_PATH.resolve()}")

    # Nice table + CSV download
    def fmt_ts(ts): return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))
    table = [
        {
            "id": r["id"],
            "timestamp_utc": fmt_ts(r["ts_utc"]),
            "query": r["query"],
            "retriever": r["retriever"],
            "topk": r["topk"],
            "fanout": r["fanout"],
            "latency_ms": r["latency_ms"],
            "provider": r["provider"],
            "model": r["model"],
            "feedback": norm_fb(r.get("feedback")),
        }
        for r in rows_f
    ]

    st.dataframe(table, use_container_width=True, hide_index=True)

    # Download CSV
    if st.button("⬇️ Download CSV (filtered)"):
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
        st.download_button("Download", output.getvalue(), file_name="dashboard_filtered.csv", mime="text/csv")

    # Expandable view per interaction
    st.markdown("### Interaction details")
    for r in rows_f:
        with st.expander(f"#{r['id']} — {fmt_ts(r['ts_utc'])} — feedback={norm_fb(r.get('feedback'))}"):
            st.write(f"**Query:** {r['query']}")
            st.caption(
                f"retriever={r['retriever']} • topk={r['topk']} • fanout={r['fanout']} • "
                f"latency={r['latency_ms']}ms • provider={r['provider']} • model={r['model']}"
            )
