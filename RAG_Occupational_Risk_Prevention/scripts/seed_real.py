from __future__ import annotations
import pathlib, sys, time, random
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from retrieval.hybrid import HybridRetriever
from app.llm.generate import generate_answer
from monitoring.logger import log_interaction

QUESTIONS = [
    "What is the purpose of the OSH Act?",
    "Employer obligations regarding hazard communication",
    "Noise exposure limits",
    "What rights do workers have under the Act?",
    "Who is responsible for workplace safety?",
    "PPE requirements summary",
    "Machine guarding basics",
    "Emergency action plans",
    "Recordkeeping requirements (OSHA 300)",
    "Permit-required confined spaces overview",
]

def main():
    retr = HybridRetriever(
        bm25_kb_path=str(ROOT / "data" / "kb" / "bm25.jsonl"),
        chroma_dir=str(ROOT / "data" / "chroma"),
        chroma_collection="osha",
    )

    for i, q in enumerate(QUESTIONS, start=1):
        t0 = time.time()
        hits = retr.search(q, k=6, fanout=60)
        t1 = time.time()
        retrieval_ms = int((t1 - t0) * 1000)

        ctx = [{"text": h["text"], "source": h["source"]} for h in hits[:6]]
        t2 = time.time()
        try:
            ans = generate_answer(q, ctx) if ctx else "Not enough context found."
        except Exception as e:
            ans = f"[LLM ERROR] {e}"
        t3 = time.time()
        llm_ms = int((t3 - t2) * 1000)

        iid = log_interaction(
            query=q,
            retriever="hybrid",
            topk=6,
            fanout=60,
            latency_ms=retrieval_ms + llm_ms,
            provider="ollama",
            model="llama3.1",
            answer=ans,
            sources=[c["source"] for c in ctx],
            ctx_len=len(ctx),
            latency_ms_retrieval=retrieval_ms,
            latency_ms_llm=llm_ms,
        )

        # feedback aleatorio para alimentar gráficas
        # (puedes quitarlos si prefieres solo likes manuales)
        from monitoring.logger import update_feedback
        fb = random.choice([1, 0, 0, -1, 0, 1])  # un poco de todo
        if fb != 0:
            update_feedback(iid, fb, "")

        print(f"✓ [{i}/{len(QUESTIONS)}] {q} — id#{iid} — ret:{retrieval_ms}ms llm:{llm_ms}ms fb:{fb}")

if __name__ == "__main__":
    main()