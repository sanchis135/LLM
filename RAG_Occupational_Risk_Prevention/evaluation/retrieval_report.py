#Running:
# python -m evaluation.retrieval_report --queries evaluation/datasets/osha_gold.jsonl --kb_bm25 data/kb/bm25.jsonl --chroma_dir data/chroma --collection osha --k_list 5,10 --out_csv reports/retrieval_report.csv


from __future__ import annotations
import argparse, json, pathlib, sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from retrieval.bm25_client import BM25Client
from retrieval.vector_client import VectorClient
from retrieval.hybrid import HybridRetriever

def load_queries(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            out.append({"query": obj["query"], "gold": set(obj.get("gold_sources", []))})
    return out

def recall_at_k(results: list[str], gold: set[str]) -> float:
    if not gold: return 0.0
    return float(len(gold.intersection(results))) / float(len(gold))

def main():
    ap = argparse.ArgumentParser(description="Compute recall@k for BM25, Vector, Hybrid.")
    ap.add_argument("--queries", required=True, help="JSONL with {query, gold_sources}")
    ap.add_argument("--kb_bm25", default=str(ROOT / "data" / "kb" / "bm25.jsonl"))
    ap.add_argument("--chroma_dir", default=str(ROOT / "data" / "chroma"))
    ap.add_argument("--collection", default="osha")
    ap.add_argument("--k_list", default="5,10", help="comma-separated, e.g., 5,10")
    ap.add_argument("--out_csv", default=str(ROOT / "reports" / "retrieval_report.csv"))
    args = ap.parse_args()

    ks = [int(x) for x in args.k_list.split(",")]
    queries = load_queries(args.queries)

    bm25 = BM25Client(args.kb_bm25)
    vec  = VectorClient(persist_dir=args.chroma_dir, collection=args.collection)
    hyb  = HybridRetriever(bm25_kb_path=args.kb_bm25, chroma_dir=args.chroma_dir, chroma_collection=args.collection)

    rows = []
    for q in queries:
        query, gold = q["query"], q["gold"]
        bm25_hits = bm25.search(query, k=max(ks))
        vec_hits  = vec.search(query,  k=max(ks))
        hyb_hits  = hyb.search(query,  k=max(ks), fanout=max(30, max(ks)*5))

        for k in ks:
            rows.append({
                "query": query,
                "k": k,
                "recall_bm25":  recall_at_k([h["source"] for h in bm25_hits[:k]], gold),
                "recall_vec":   recall_at_k([h["source"] for h in vec_hits[:k]],   gold),
                "recall_hyb":   recall_at_k([h["source"] for h in hyb_hits[:k]],   gold),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("\n=== Mean Recall ===")
    print(df.groupby("k")[["recall_bm25","recall_vec","recall_hyb"]].mean().round(3))
    print(f"\n✅ Saved report → {args.out_csv}")

if __name__ == "__main__":
    main()
