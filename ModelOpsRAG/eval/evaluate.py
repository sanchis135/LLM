import os, json, math, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Make local packages importable when run from repo root
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from retriever.bm25 import BM25Index
from retriever.vectors import VectorIndex
from retriever.rrf import rrf_fuse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed' / 'kb_chunks.jsonl'
QS_FILE = ROOT / 'eval' / 'questions.jsonl'
GL_FILE = ROOT / 'eval' / 'gold_labels.jsonl'

# ----------------- Metrics -----------------

def dcg(rels: List[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

def ndcg(rels: List[float]) -> float:
    ideal = sorted(rels, reverse=True)
    denom = dcg(ideal)
    return (dcg(rels) / denom) if denom > 0 else 0.0

def precision_at_k(relevances: List[int], k: int) -> float:
    if k == 0: return 0.0
    return sum(relevances[:k]) / float(k)

def recall_at_k(retrieved_ids: List[int], relevant_set: Set[int]) -> float:
    if not relevant_set: 
        # Edge case: no relevant docs in corpus per keyword heuristic → define as 1.0
        return 1.0
    hits = sum(1 for i in retrieved_ids if i in relevant_set)
    return hits / float(len(relevant_set))

# ----------------- Loading -----------------

def load_docs() -> List[Dict]:
    docs = []
    with open(DATA, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            o['_id'] = f"doc-{i}"
            docs.append(o)
    return docs

def load_eval() -> Tuple[List[Dict], Dict[str, List[str]]]:
    questions = []
    with open(QS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    gold = {}
    with open(GL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                gold[obj['id']] = [w.lower() for w in obj.get('contains_keywords', [])]
    return questions, gold

# ----------------- Heuristic relevance -----------------

def make_relevant_set(docs: List[Dict], keywords: List[str]) -> Set[int]:
    """All doc indices whose section/text contain ANY of the keywords (lowercased)."""
    rel = set()
    if not keywords:
        return rel
    kws = [k.lower() for k in keywords]
    for i, d in enumerate(docs):
        sec = (d.get('section') or '').lower()
        txt = (d.get('text') or '').lower()
        if any(k in sec or k in txt for k in kws):
            rel.add(i)
    return rel

# ----------------- Retrieval wrappers -----------------

def bm25_search(bm25: BM25Index, k: int, q: str) -> List[int]:
    res = bm25.search(q, k=k)
    idxs = []
    for doc_id, _ in res:
        try:
            idxs.append(int(doc_id.split('-')[-1]))
        except:
            continue
    return idxs

def vector_search(vec: VectorIndex, k: int, q: str) -> List[int]:
    # fetch more to give fusion a chance; we'll trim later
    res = vec.search(q, k=max(k, 10))
    idxs = []
    for vid, _ in res:
        md, _txt = vec.get_by_id(vid)
        if md and 'row_index' in md:
            idxs.append(int(md['row_index']))
    return idxs[:k]

def rrf_search(bm25: BM25Index, vec: VectorIndex, k: int, q: str) -> List[int]:
    b = bm25.search(q, k=max(k, 20))
    v = vec.search(q, k=max(k, 20))
    # normalize ids to 'row-<idx>'
    bm = []
    for doc_id, score in b:
        try:
            idx = int(doc_id.split('-')[-1])
            bm.append((f"row-{idx}", float(score)))
        except:
            pass
    vv = []
    for vid, score in v:
        md, _ = vec.get_by_id(vid)
        if md and 'row_index' in md:
            vv.append((f"row-{int(md['row_index'])}", float(score)))
    fused = rrf_fuse({'bm25': bm, 'vector': vv})
    out = []
    for rid, _score in fused[:k]:
        try:
            out.append(int(rid.split('-')[-1]))
        except:
            continue
    return out

# ----------------- Main eval -----------------

def evaluate(k: int = 5):
    if not DATA.exists():
        raise SystemExit(f"Missing dataset: {DATA}. Run: python ingest/ingest.py")

    docs = load_docs()
    questions, gold = load_eval()

    bm25 = BM25Index(docs)
    vec = VectorIndex(persist_dir=os.getenv('CHROMA_DIR', None))
    # ensure vectors exist for current dataset
    vec.upsert_jsonl(str(DATA))

    # aggregates
    agg = {
        'bm25': {'ndcg': [], 'prec': [], 'recall': []},
        'vector': {'ndcg': [], 'prec': [], 'recall': []},
        'rrf': {'ndcg': [], 'prec': [], 'recall': []},
    }

    for q in questions:
        qid = q['id']
        query = q['question']
        keywords = gold.get(qid, [])

        # relevant set from corpus by keyword heuristic
        relevant_set = make_relevant_set(docs, keywords)

        # run methods
        bm_ids = bm25_search(bm25, k, query)
        ve_ids = vector_search(vec, k, query)
        rf_ids = rrf_search(bm25, vec, k, query)

        # build graded relevance vectors for nDCG (binary 0/1 here)
        def rel_list(pred_ids: List[int]) -> List[int]:
            return [1 if i in relevant_set else 0 for i in pred_ids]

        for name, ids in [('bm25', bm_ids), ('vector', ve_ids), ('rrf', rf_ids)]:
            rels = rel_list(ids)
            agg[name]['ndcg'].append(ndcg(rels))
            agg[name]['prec'].append(precision_at_k(rels, k))
            agg[name]['recall'].append(recall_at_k(ids, relevant_set))

    def avg(xs: List[float]) -> float:
        return sum(xs)/len(xs) if xs else 0.0

    summary = {
        'k': k,
        'bm25':   {'ndcg@k': round(avg(agg['bm25']['ndcg']), 4),
                   'precision@k': round(avg(agg['bm25']['prec']), 4),
                   'recall@k': round(avg(agg['bm25']['recall']), 4)},
        'vector': {'ndcg@k': round(avg(agg['vector']['ndcg']), 4),
                   'precision@k': round(avg(agg['vector']['prec']), 4),
                   'recall@k': round(avg(agg['vector']['recall']), 4)},
        'rrf':    {'ndcg@k': round(avg(agg['rrf']['ndcg']), 4),
                   'precision@k': round(avg(agg['rrf']['prec']), 4),
                   'recall@k': round(avg(agg['rrf']['recall']), 4)},
    }

    # pick best by ndcg, then recall
    best = max(['bm25','vector','rrf'],
               key=lambda m: (summary[m]['ndcg@k'], summary[m]['recall@k']))
    summary['best'] = best

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5, help='cutoff for @k metrics')
    args = parser.parse_args()
    evaluate(k=args.k)
