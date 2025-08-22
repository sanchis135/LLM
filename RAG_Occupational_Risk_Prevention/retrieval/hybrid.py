from __future__ import annotations
from typing import List, Dict, Tuple
from retrieval.bm25_client import BM25Client
from retrieval.vector_client import VectorClient

def reciprocal_rank_fusion(
    bm25_hits: List[Dict],
    dense_hits: List[Dict],
    k: int = 10,
    rrf_k: float = 60.0
) -> List[Dict]:
    """
    RRF: score(doc) = sum( 1 / (rrf_k + rank_i) ) over lists (bm25, dense).
    - bm25_hits/dense_hits: lists of dicts with at least 'text'
    - k: final size of the fused list
    """
    # Build ranking (by position) for each list
    ranks_maps = []
    for hits in (bm25_hits, dense_hits):
        rank_map = {}
        for rank, h in enumerate(hits):
            rank_map[h["text"]] = rank  # 0-based
        ranks_maps.append(rank_map)

    # Universe of docs
    all_docs = {}
    for hits in (bm25_hits, dense_hits):
        for h in hits:
            all_docs[h["text"]] = h  # keep the first dict as base

    # RRF scoring
    scored: List[Tuple[float, str]] = []
    for text, h in all_docs.items():
        score = 0.0
        for rank_map in ranks_maps:
            if text in rank_map:
                score += 1.0 / (rrf_k + rank_map[text] + 1.0)
        scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    out: List[Dict] = []
    for s, t in top:
        base = all_docs[t].copy()
        base["score"] = float(s)
        out.append(base)
    return out


class HybridRetriever:
    """
    Hybrid retriever: BM25 + vector (Chroma) + RRF fusion.
    - Uses the KB JSONL for BM25 and the persisted Chroma collection for dense search.
    """
    def __init__(
        self,
        bm25_kb_path: str = "data/kb/bm25.jsonl",
        chroma_dir: str = "data/chroma",
        chroma_collection: str = "osha",
    ) -> None:
        self.bm25 = BM25Client(bm25_kb_path)
        self.vec = VectorClient(persist_dir=chroma_dir, collection=chroma_collection)

    def search(self, query: str, k: int = 6, fanout: int = 20) -> List[Dict]:
        """
        fanout: number of initial candidates per engine before fusion.
        """
        bm25_hits = self.bm25.search(query, k=fanout)
        dense_hits = self.vec.search(query, k=fanout)
        fused = reciprocal_rank_fusion(bm25_hits, dense_hits, k=k, rrf_k=60.0)
        return fused
