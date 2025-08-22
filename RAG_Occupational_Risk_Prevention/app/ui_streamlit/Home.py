from __future__ import annotations

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import time
import streamlit as st

from retrieval.retrieval import BM25Client, VectorClient, HybridRetriever
from app.llm.generate import generate_answer

from monitoring.logger import log_interaction, update_feedback
import os

DEFAULT_KB = str(ROOT / "data" / "kb" / "bm25.jsonl")
DEFAULT_CHROMA_DIR = str(ROOT / "data" / "chroma")
DEFAULT_COLLECTION = "osha"

if "last_interaction_id" not in st.session_state:
    st.session_state["last_interaction_id"] = None

st.set_page_config(page_title="RAG‑OSHA", page_icon="🦺", layout="wide")

# ---------- Client cache ----------
@st.cache_resource(show_spinner=False)
def get_bm25(kb_path: str = DEFAULT_KB):
    return BM25Client(kb_path)

@st.cache_resource(show_spinner=False)
def get_vec(persist_dir: str = DEFAULT_CHROMA_DIR, collection: str = DEFAULT_COLLECTION):
    return VectorClient(persist_dir=persist_dir, collection=collection)

@st.cache_resource(show_spinner=False)
def get_hybrid(kb_path=DEFAULT_KB, persist_dir=DEFAULT_CHROMA_DIR, collection=DEFAULT_COLLECTION):
    return HybridRetriever(bm25_kb_path=kb_path, chroma_dir=persist_dir, chroma_collection=collection)

# ---------- Sidebar ----------
st.sidebar.header("Settings")
topk = st.sidebar.slider("Top‑k results", min_value=3, max_value=20, value=5, step=1)
fanout = st.sidebar.slider("Fanout (pre-candidates per engine)", min_value=10, max_value=100, value=30, step=5)
kb_path = st.sidebar.text_input("KB JSONL (BM25)", DEFAULT_KB)
chroma_dir = st.sidebar.text_input("Chroma dir", DEFAULT_CHROMA_DIR)
collection = st.sidebar.text_input("Chroma collection", DEFAULT_COLLECTION)

st.sidebar.caption("Tip: build the indexes before using the app:")
st.sidebar.code(
    "python -m ingestion.ingest --raw_dir docs --kb_out data/kb/bm25.jsonl\n"
    "python -m ingestion.index_vectors --kb_jsonl data/kb/bm25.jsonl --persist_dir data/chroma --collection osha"
)

# ---------- Tabs ----------
tab_ret, tab_ans = st.tabs(["🔎 Retrieval", "🧠 Answer (LLM)"])

def highlight_terms(text: str, query: str) -> str:
    t = text
    for tok in set(query.lower().split()):
        if len(tok) >= 3:
            t = t.replace(tok, f"**{tok}**")
            t = t.replace(tok.capitalize(), f"**{tok.capitalize()}**")
    return t

def render_hits(title: str, hits: list[dict], query: str):
    st.subheader(title)
    if not hits:
        st.info("No results found.")
        return
    for i, h in enumerate(hits, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. score:** `{h.get('score', 0):.4f}`  •  **source:** `{h.get('source','?')}`")
            st.write(highlight_terms(h.get("text",""), query))

# =========================
# TAB: RETRIEVAL
# =========================
with tab_ret:
    st.title("🔎 Retrieval (BM25 / Vector / Hybrid)")
    q = st.text_input(
        "Query",
        value="Purpose of the OSH Act",
        help="Examples: 'noise exposure limit', 'imminent danger'"
    )

    col_run1, col_run2 = st.columns([1, 3])
    with col_run1:
        run_bm25 = st.button("BM25")
    with col_run2:
        run_all = st.button("BM25 + Vector + Hybrid (RRF)")

    if (run_bm25 or run_all) and q.strip():
        try:
            bm25 = get_bm25(kb_path)
            vec = get_vec(chroma_dir, collection)
            hyb = get_hybrid(kb_path, chroma_dir, collection)
        except Exception as e:
            st.error(f"Error loading indexes: {e}")
        else:
            t0 = time.time()
            bm25_hits = bm25.search(q, k=topk if run_bm25 else max(topk, fanout))
            t1 = time.time()
            vec_hits = vec.search(q, k=max(topk, fanout)) if run_all else []
            hyb_hits = hyb.search(q, k=topk, fanout=fanout) if run_all else []
            t2 = time.time()

            if run_bm25 and not run_all:
                st.caption(f"BM25 time: {(t1 - t0)*1000:.0f} ms")
                render_hits("BM25", bm25_hits, q)
                st.session_state["last_hits"] = bm25_hits
                st.session_state["last_query"] = q
            else:
                colA, colB, colC = st.columns(3)
                with colA:
                    st.caption(f"BM25 time: {(t1 - t0)*1000:.0f} ms")
                    render_hits("BM25", bm25_hits[:topk], q)
                with colB:
                    render_hits("Vector", vec_hits[:topk], q)
                with colC:
                    st.caption(f"Hybrid time: {(t2 - t1)*1000:.0f} ms")
                    render_hits("Hybrid (RRF)", hyb_hits, q)

                # Guardar para la pestaña Answer: preferimos híbrido
                st.session_state["last_hits"] = hyb_hits if run_all else bm25_hits
                st.session_state["last_query"] = q

# =========================
# TAB: ANSWER (LLM)
# =========================
with tab_ans:
    st.title("🧠 Answer (LLM) with citations")
    st.write("Uses passages from the last retrieval (preferably Hybrid).")

    q2 = st.text_input(
        "Query (for generation)", 
        value=st.session_state.get("last_query", "Purpose of the OSH Act")
    )
    max_ctx = st.slider("Passages to use", 1, 8, 4, 1)

    if st.button("Generate answer"):
        hits = st.session_state.get("last_hits", [])
        if not hits:
            st.warning("Please run a retrieval first in the 'Retrieval' tab.")
        else:
            ctx = [{"text": h["text"], "source": h["source"]} for h in hits[:max_ctx]]

            t0 = time.time()
            with st.spinner("Generating with LLM..."):
                try:
                    ans = generate_answer(q2, ctx)
                except Exception as e:
                    st.error(f"Could not generate the answer: {e}")
                    ans = None
            latency_ms = int((time.time() - t0) * 1000)

            if ans:
                st.markdown("### Answer")
                st.write(ans)

            st.markdown("### Sources")
            for i, c in enumerate(ctx, start=1):
                st.markdown(f"[{i}] `{c['source']}`")

            st.markdown("### Passages used for the answer")
            for i, c in enumerate(ctx, start=1):
                with st.expander(f"[{i}] {c['source']}"):
                    st.write(c["text"][:1200])

            # =========================
            # LOG INTERACTION
            # =========================
            provider = "ollama"
            model = os.getenv("OLLAMA_MODEL", "llama3.1")

            sources_list = [c["source"] for c in ctx]
            interaction_id = log_interaction(
                query=q2,
                retriever="hybrid",       # or "bm25" if only BM25 is used
                topk=topk,
                fanout=fanout,
                latency_ms=latency_ms,
                provider=provider,
                model=model,
                answer=ans,
                sources=sources_list,
                ctx_len=len(ctx),
            )
            st.session_state["last_interaction_id"] = interaction_id

    # =========================
    # FEEDBACK
    # =========================
    st.markdown("### Feedback")
    iid = st.session_state.get("last_interaction_id")
    col_like, col_dislike = st.columns([1,1])
    with col_like:
        if st.button("👍 Helpful"):
            if iid:
                update_feedback(iid, +1, "")
                st.success("Thanks for your feedback!")
            else:
                st.warning("No active interaction to associate feedback. Please generate an answer first.")
    with col_dislike:
        fb_txt = st.text_input("Tell us what went wrong (optional)")
        if st.button("Submit feedback"):
            if iid:
                update_feedback(iid, -1, fb_txt)
                st.success("Thank you! Your comments will help us improve.")
            else:
                st.warning("No active interaction to associate feedback. Please generate an answer first.")            