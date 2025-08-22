from __future__ import annotations
import argparse, csv, json, time, pathlib, sys

# --- local imports (from repo root) ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from retrieval.retrieval import HybridRetriever  # make sure it is implemented
from app.llm.generate import generate_answer     # uses Ollama by default if active

def load_questions(path: pathlib.Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    qs: list[str] = []
    if path.suffix.lower() in {".txt"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                qs.append(line)
    elif path.suffix.lower() in {".jsonl"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("query") or obj.get("question")
            if q:
                qs.append(str(q))
    else:
        raise ValueError("Unsupported format; use .txt (one question per line) or .jsonl ({'query': ...}).")
    return qs

def main():
    ap = argparse.ArgumentParser(description="Batch QA: run questions, perform RAG, and save results.")
    ap.add_argument("--questions", default="evaluation/datasets/ad_hoc_questions.txt", help="TXT (1/line) or JSONL with field 'query'")
    ap.add_argument("--kb_bm25", default=str(ROOT / "data" / "kb" / "bm25.jsonl"))
    ap.add_argument("--chroma_dir", default=str(ROOT / "data" / "chroma"))
    ap.add_argument("--collection", default="osha")
    ap.add_argument("--topk", type=int, default=6, help="final k (after RRF) to show/use in generation")
    ap.add_argument("--fanout", type=int, default=60, help="candidates per engine before fusion")
    ap.add_argument("--max_ctx", type=int, default=6, help="number of passages used for generation")
    ap.add_argument("--out_csv", default=str(ROOT / "reports" / "batch_qa.csv"))
    ap.add_argument("--out_jsonl", default=str(ROOT / "reports" / "batch_qa.jsonl"))
    args = ap.parse_args()

    questions_path = pathlib.Path(args.questions)
    out_csv = pathlib.Path(args.out_csv)
    out_jsonl = pathlib.Path(args.out_jsonl)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions = load_questions(questions_path)
    if not questions:
        print("⚠️ No questions to run.")
        return

    # Load hybrid retriever
    retr = HybridRetriever(bm25_kb_path=args.kb_bm25, chroma_dir=args.chroma_dir, chroma_collection=args.collection)

    # CSV headers
    headers = [
        "id","query","retriever","topk","fanout","ctx_len",
        "latency_ms_retrieval","latency_ms_llm",
        "answer",
        "sources",          # separated by ' | '
        "sources_count",
        "used_text_preview" # first ~200 chars of the first passage
    ]

    rows = []
    with out_jsonl.open("w", encoding="utf-8") as fj:
        with out_csv.open("w", encoding="utf-8", newline="") as fc:
            writer = csv.DictWriter(fc, fieldnames=headers)
            writer.writeheader()

            for i, q in enumerate(questions, start=1):
                # --- Retrieval ---
                t0 = time.time()
                hits = retr.search(q, k=args.topk, fanout=args.fanout)  # assumes internal RRF
                t1 = time.time()
                retrieval_ms = int((t1 - t0) * 1000)

                # Context for LLM
                ctx = [{"text": h["text"], "source": h["source"]} for h in (hits[:args.max_ctx] if hits else [])]

                # --- LLM ---
                t2 = time.time()
                try:
                    ans = generate_answer(q, ctx) if ctx else "Not enough context to generate an answer."
                except Exception as e:
                    ans = f"[ERROR LLM] {e}"
                t3 = time.time()
                llm_ms = int((t3 - t2) * 1000)

                sources = [c["source"] for c in ctx]
                used_preview = (ctx[0]["text"][:200] + "…") if ctx else ""

                rec = {
                    "id": i,
                    "query": q,
                    "retriever": "hybrid",
                    "topk": args.topk,
                    "fanout": args.fanout,
                    "ctx_len": len(ctx),
                    "latency_ms_retrieval": retrieval_ms,
                    "latency_ms_llm": llm_ms,
                    "answer": ans,
                    "sources": " | ".join(sources),
                    "sources_count": len(sources),
                    "used_text_preview": used_preview,
                }
                writer.writerow(rec)
                rows.append(rec)

                # Detailed JSONL (useful for auditing/post-mortem)
                fj.write(json.dumps({
                    "id": i,
                    "query": q,
                    "hits": hits,      # includes text and metadata
                    "ctx": ctx,
                    "answer": ans,
                    "latency_ms": {"retrieval": retrieval_ms, "llm": llm_ms}
                }, ensure_ascii=False) + "\n")

                print(f"✓ [{i}/{len(questions)}] '{q}'  (retrieval {retrieval_ms}ms, llm {llm_ms}ms)")

    print(f"\n✅ Saved: {out_csv}")
    print(f"✅ Saved: {out_jsonl}")
    print("Done.")
if __name__ == "__main__":
    main()
