from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from typing import Dict, List

from evaluation.metrics import recall_at_k, ndcg_at_k
from retrieval.bm25_client import BM25Client
from retrieval.vector_client import VectorClient
from retrieval.hybrid import HybridRetriever

def load_queries(path: str | Path) -> List[Dict]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def filename_of(source: str) -> str:
    return source.split("#", 1)[0] if "#" in source else source

def make_retrieved_lists(sources: List[str], gold: List[str]) -> List[str]:
    """
    Devuelve una lista 'retrieved_for_eval' del mismo tamaño que 'sources'.
    Si el gold no lleva '#', evaluamos por archivo (prefix-match del filename).
    Si lleva '#', evaluamos por el source completo (chunk exacto).
    """
    # Clasifica golds en exactos y por archivo
    gold_exact = [g for g in gold if "#" in g]
    gold_files = [g for g in gold if "#" not in g]
    # Construye lista equivalente de labels para ndcg/recall
    retrieved_for_eval = []
    for s in sources:
        fn = filename_of(s)
        # match exacto si está en gold_exact
        if s in gold_exact:
            retrieved_for_eval.append(s)  # exact match
            continue
        # match por archivo si el filename está en gold_files
        if fn in gold_files:
            # marcamos como si fuese ese filename gold (para contar como 1)
            retrieved_for_eval.append(fn)
        else:
            retrieved_for_eval.append(s)  # sin match (no estará en gold)
    return retrieved_for_eval

def eval_system(
    name: str,
    queries: List[Dict],
    search_fn,
    k_list=(5, 10),
) -> Dict[str, float]:
    metrics = {f"{name}_Recall@{k}": [] for k in k_list}
    metrics |= {f"{name}_nDCG@{k}": [] for k in k_list}

    for q in queries:
        gold = q.get("gold_sources", [])
        hits = search_fn(q["query"], max(k_list))
        sources = [h["source"] for h in hits]

        # normaliza retrieved para evaluación por archivo o exacta
        retrieved_eval = make_retrieved_lists(sources, gold)

        for k in k_list:
            metrics[f"{name}_Recall@{k}"].append(recall_at_k(gold, retrieved_eval, k))
            metrics[f"{name}_nDCG@{k}"].append(ndcg_at_k(gold, retrieved_eval, k))

    return {m: (sum(vals) / len(vals) if vals else 0.0) for m, vals in metrics.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="evaluation/datasets/prl_queries.jsonl")
    ap.add_argument("--kb_bm25", default="data/kb/bm25.jsonl")
    ap.add_argument("--chroma_dir", default="data/chroma")
    ap.add_argument("--collection", default="prl")
    ap.add_argument("--k_list", default="5,10")
    ap.add_argument("--out_csv", default="reports/retrieval_eval.csv")
    ap.add_argument("--fanout", type=int, default=25)
    args = ap.parse_args()

    k_list = tuple(int(x) for x in args.k_list.split(","))
    queries = load_queries(args.queries)
    if not queries:
        raise SystemExit(f"No queries found in {args.queries}")

    bm25 = BM25Client(args.kb_bm25)
    vec = VectorClient(persist_dir=args.chroma_dir, collection=args.collection)
    hyb = HybridRetriever(bm25_kb_path=args.kb_bm25, chroma_dir=args.chroma_dir, chroma_collection=args.collection)

    def bm25_search(q: str, k: int): return bm25.search(q, k=k)
    def vec_search(q: str, k: int): return vec.search(q, k=k)
    def hyb_search(q: str, k: int): return hyb.search(q, k=k, fanout=args.fanout)

    res_bm25 = eval_system("BM25", queries, bm25_search, k_list=k_list)
    res_vec  = eval_system("VEC", queries, vec_search, k_list=k_list)
    res_hyb  = eval_system("HYB", queries, hyb_search, k_list=k_list)

    print("=== Retrieval Evaluation (means) ===")
    for name, res in [("BM25", res_bm25), ("VEC", res_vec), ("HYB", res_hyb)]:
        for k in k_list:
            print(f"{name}: Recall@{k}={res.get(f'{name}_Recall@{k}', 0):.3f}  "
                  f"nDCG@{k}={res.get(f'{name}_nDCG@{k}', 0):.3f}")
        print("-")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    k_max = max(k_list)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "system", "k", "Recall", "nDCG"])
        for q in queries:
            gold = q.get("gold_sources", [])
            for sys_name, fn in [("BM25", bm25_search), ("VEC", vec_search), ("HYB", hyb_search)]:
                sources = [h["source"] for h in fn(q["query"], k_max)]
                retrieved_eval = make_retrieved_lists(sources, gold)
                writer.writerow([
                    q["query"], sys_name, k_max,
                    f"{recall_at_k(gold, retrieved_eval, k_max):.4f}",
                    f"{ndcg_at_k(gold, retrieved_eval, k_max):.4f}",
                ])
    print(f"\n✅ Per-query results saved to {out.resolve()}")

if __name__ == "__main__":
    main()

