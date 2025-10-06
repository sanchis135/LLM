# ==== Import path fix (Windows-friendly) ====
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, json, textwrap
from pathlib import Path
import streamlit as st
import pandas as pd

from retriever.bm25 import BM25Index
from retriever.vectors import VectorIndex
from retriever.rewrite import rewrite_query
from retriever.rrf import rrf_fuse
from monitoring.feedback_db import FeedbackStore

# Optional: Ollama LLM
try:
    import ollama
except Exception:
    ollama = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed' / 'kb_chunks.jsonl'
DB_PATH = os.getenv('FEEDBACK_DB', str(ROOT / 'monitoring' / 'feedback.sqlite'))

st.set_page_config(page_title='ModelOpsRAG', layout='wide')
st.title('ModelOpsRAG — MLOps & Deployment Assistant (Ollama-ready)')

if not DATA.exists():
    st.error('Missing processed dataset. Run `python ingest/ingest.py` first.')
    st.stop()

# Load docs for BM25 and a text map for details
docs = []
with open(DATA, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        o = json.loads(line)
        o['_id'] = f"doc-{i}"
        docs.append(o)

bm25 = BM25Index(docs)
vec = VectorIndex(persist_dir=os.getenv('CHROMA_DIR', None))
vec.upsert_jsonl(str(DATA))  # ensure vector embeddings exist

fb = FeedbackStore(DB_PATH)

# ------------- UI -------------
col1, col2 = st.columns([2,1])
with col1:
    query = st.text_input(
        'Ask about MLOps/Deployment (e.g., "canary rollout on OpenShift")',
        'canary rollout on OpenShift'
    )
    if st.button('Search'):
        st.session_state['do_search'] = True
with col2:
    topk = st.slider('Top-k context', 3, 12, 6)
    facet = st.selectbox('Facet', ['deployment','cicd','monitoring','governance','general'])
    gen_with_ollama = st.checkbox(
        'Generate answer with Ollama',
        value=True if ollama else False,
        help='Requires Ollama running and local models.'
    )

if st.session_state.get('do_search'):
    # Query rewrite + facet hint
    q = rewrite_query(query)
    if facet and facet != 'general':
        q += ' ' + facet

    # Retrieve
    b = bm25.search(q, k=topk)
    v = vec.search(q, k=topk)

    id2text = { d['_id']: d['text'] for d in docs }
    id2meta = { d['_id']: d for d in docs }

    # Optional: collapsible raw rankings (nice for debugging, hide in demo)
    with st.expander('Top BM25 hits (debug)', expanded=False):
        for doc_id, score in b:
            meta = id2meta.get(doc_id, {})
            st.write(f"- `{doc_id}` · {score:.3f} — **{meta.get('section','')}** — {id2text.get(doc_id, '')[:180]}…")

    st.subheader('Top Vector hits')
    vector_rows = []
    for vec_id, score in v:
        md, tx = vec.get_by_id(vec_id)
        if not md and not tx:
            st.write(f"- `{vec_id}` · {score:.3f}")
            continue
        sec = (md or {}).get('section','')
        st.write(f"- `{vec_id}` · {score:.3f} — **{sec}** — { (tx or '')[:180] }…")
        if md and 'row_index' in md:
            vector_rows.append((md['row_index'], score))

    # ==== RRF FUSION (BM25 + Vector) ====
    bm25_rows = []
    for doc_id, score in b:
        try:
            idx = int(doc_id.split('-')[-1])  # 'doc-<i>'
            bm25_rows.append((f"row-{idx}", score))
        except:
            pass

    vector_rows_norm = [(f"row-{ri}", sc) for (ri, sc) in vector_rows]
    fused = rrf_fuse({'bm25': bm25_rows, 'vector': vector_rows_norm})

    # Build preliminary ranking list
    ranked_rows = []
    for rid, rrf_score in fused[:max(topk*2, 10)]:
        try:
            idx = int(rid.split('-')[-1])
        except:
            continue
        d = docs[idx]
        ranked_rows.append((idx, rrf_score, d))

    # ==== Heuristic reranker for OpenShift canary ====
    ql = q.lower()

    QUERY_KEYWORDS = {
        "canary": 3.0,
        "route": 2.5,
        "servicemesh": 2.2,
        "istio": 2.2,
        "virtualservice": 2.0,
        "destinationrule": 1.8,
        "deployment": 1.2,
        "rollout": 1.2,
    }
    def kw_score(text: str) -> float:
        t = (text or "").lower()
        return sum(w for k, w in QUERY_KEYWORDS.items() if k in ql and k in t)

    def penalty(text: str) -> float:
        t = (text or "").lower()
        if any(k in ql for k in ["canary","route","istio","servicemesh","virtualservice"]) and "registry" in t:
            return -3.0
        return 0.0

    ranked_rows = sorted(
        ranked_rows,
        key=lambda x: kw_score(x[2].get('text','')) + kw_score(x[2].get('section','')) + penalty(x[2].get('section','')) + x[1],
        reverse=True
    )

    st.markdown('---')
    st.subheader('Fused (RRF) top results')
    for idx, rrf_score, d in ranked_rows[:topk]:
        st.write(f"- row-{idx} · rrf={rrf_score:.4f} — **{d.get('section','')}** — { (d.get('text','')[:180]) }…")

    # ==== Build context from final ranking ====
    context_chunks = []
    sources = []

    def mk_tag(d: dict) -> str:
        doc = (d.get('doc_id','') or '').strip()
        sec_raw = (d.get('section','') or '')
        sec = sec_raw.splitlines()[0].strip()[:120]  # sanitize one-liner
        return f"[{doc} :: {sec}]"

    for idx, _, d in ranked_rows[:min(3, len(ranked_rows))]:
        tag = mk_tag(d)
        context_chunks.append(f"{tag}\n{ d.get('text','') }")
        sources.append(tag)

    context = "\n\n---\n\n".join(context_chunks)

    def render_sources():
        st.markdown("**Sources**")
        for s in sorted(set(sources)):
            st.write(f"- {s}")

    # ==== Answer ====
    st.markdown('---')
    st.subheader('Answer')

    USE_OLLAMA = bool(gen_with_ollama and ollama)
    if USE_OLLAMA:
        # ensure host if running dockerized app talking to host ollama
        if os.getenv('OLLAMA_HOST'):
            os.environ['OLLAMA_HOST'] = os.getenv('OLLAMA_HOST')

        model = os.getenv('OLLAMA_CHAT_MODEL', 'llama3.1:8b')
        prompt = f"""
        You are a senior MLOps assistant.
        - Answer ONLY with the context below. If missing info, say: "Insufficient context."
        - Output MUST be valid Markdown.
        - Use fenced code blocks for YAML: ```yaml ... ```
        - Cite inline like [doc_id :: section] EXACTLY. Do not invent sources or placeholders.
        - Do NOT print the word "undefined". If you don't have a value, omit it.
        - If the question mentions OpenShift, PREFER Route (route.openshift.io/v1) or Istio (VirtualService/DestinationRule).
        - AVOID proposing Kubernetes Ingress or Service type LoadBalancer unless such resources are explicitly present in the provided context.
        - Do NOT output deprecated API versions (e.g., networking.k8s.io/v1beta1).

        Question: {query}

        Context:
        {context}
        """.strip()

        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "Be concise, accurate, and cite sections."},
                    {"role": "user", "content": prompt}
                ]
            )
            ans = (resp.get('message', {}).get('content', '') or '').strip()  # <-- one dot is a typo; fix next line
        except Exception as e:
            ans = None
            st.warning(f"Ollama call failed: {e}. Showing heuristic draft instead.")

        if not ans:
            st.write("Based on retrieved docs: consider OpenShift weighted Route or Service Mesh (Istio) for traffic-splitting (e.g., 90/10), monitor p95 latency & error rate, ramp if SLOs hold, and keep rollback ready.")
            render_sources()
        else:
            # Post-process: fence YAML if flat, remove literal 'undefined'
            if "apiVersion:" in ans and "```" not in ans:
                ans = ans.replace("apiVersion:", "```yaml\napiVersion:").rstrip() + "\n```"
            ans = ans.replace("undefined", "").strip()
            st.write(ans)
            render_sources()
    else:
        st.write("Based on retrieved docs: consider OpenShift weighted Route or Service Mesh (Istio) for traffic-splitting (e.g., 90/10), monitor p95 latency & error rate, ramp if SLOs hold, and keep rollback ready.")
        render_sources()

    st.divider()
    st.subheader('Engineer feedback')
    useful = st.radio('Was this helpful?', ['Yes','No'], horizontal=True)
    note = st.text_input('Optional comment')
    if st.button('Submit feedback'):
        fb.record(event='eval_feedback', payload={'useful': useful, 'note': note, 'query': query})
        st.toast('Thanks for your feedback!')
