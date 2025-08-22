from typing import List, Tuple
from rank_bm25 import BM25Okapi
import re

def _simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenizer:
    - lowercase
    - optionally remove accents (uncomment if you want stronger normalization)
    - split by alphanumeric characters
    """
    # If you want to strip accents:
    # import unicodedata
    # text = "".join(
    #     c for c in unicodedata.normalize("NFD", text.lower())
    #     if unicodedata.category(c) != "Mn"
    # )
    text = text.lower()
    return re.findall(r"\w+", text, flags=re.UNICODE)

class BM25Retriever:
    """
    True BM25 with rank-bm25.
    - docs: list of documents (strings)
    - you can pass long lists (chunks) for better granularity
    """
    def __init__(self, docs: List[str]):
        if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
            raise TypeError("docs must be a list of strings")
        self.docs = docs
        self.corpus_tokens = [_simple_tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int = 5) -> List[str]:
        q_tokens = _simple_tokenize(query)
        if not q_tokens:
            return self.docs[:k]
        scores = self.bm25.get_scores(q_tokens)
        # top-k indices by descending score
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in idx]

    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        q_tokens = _simple_tokenize(query)
        if not q_tokens:
            return [(d, 0.0) for d in self.docs[:k]]
        scores = self.bm25.get_scores(q_tokens)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.docs[i], float(scores[i])) for i in idx]


from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import re

# For vector search (Chroma)
import chromadb
from chromadb.utils import embedding_functions

def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

# -------- BM25Client over KB JSONL --------
class BM25Client:
    """
    Reads data/kb/bm25.jsonl and exposes search() returning dicts:
    {text, score, source, meta}
    """
    def __init__(self, kb_path: str = "data/kb/bm25.jsonl") -> None:
        from rank_bm25 import BM25Okapi
        self.kb_path = Path(kb_path)
        if not self.kb_path.exists():
            raise FileNotFoundError(f"KB not found: {self.kb_path}")

        self.docs: List[str] = []
        self.tokens: List[List[str]] = []
        self.sources: List[str] = []
        self.metas: List[Dict] = []
        with self.kb_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                toks = obj.get("tokens") or _simple_tokenize(text)
                self.docs.append(text)
                self.tokens.append(toks)
                self.sources.append(obj.get("source", "?"))
                self.metas.append(obj.get("meta", {}))
        if not self.docs:
            raise ValueError(f"KB is empty: {self.kb_path}")

        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        q = _simple_tokenize(query)
        if not q:
            idx = list(range(min(k, len(self.docs))))
            return [self._hit(i, 0.0) for i in idx]
        scores = self.bm25.get_scores(q)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._hit(i, float(scores[i])) for i in idx]

    def _hit(self, i: int, score: float) -> Dict:
        return {"text": self.docs[i], "score": score, "source": self.sources[i], "meta": self.metas[i]}

# -------- VectorClient (Chroma + Sentence-Transformers) --------
def _to_similarity(distance: float) -> float:
    try:
        return 1.0 - float(distance)
    except Exception:
        return 1.0 / (1.0 + float(distance))

class VectorClient:
    def __init__(self, persist_dir: str = "data/chroma", collection: str = "osha", model_name: Optional[str] = None):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.model_name = model_name or "intfloat/multilingual-e5-small"
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        self.col = self.client.get_or_create_collection(name=collection, embedding_function=self.embedder)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        # ⚠️ E5 requires "query: " prefix in the query
        res = self.col.query(
            query_texts=[f"query: {query}"],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: List[Dict] = []
        for t, m, d in zip(docs, metas, dists):
            # t may come with "passage: " prefix if indexed like that (recommended)
            out.append({"text": t, "score": _to_similarity(d), "source": m.get("source", "?"), "meta": m})
        return out

# -------- RRF Fusion + HybridRetriever --------
def reciprocal_rank_fusion(bm25_hits: List[Dict], dense_hits: List[Dict], k: int = 10, rrf_k: float = 60.0) -> List[Dict]:
    rank_maps = []
    for hits in (bm25_hits, dense_hits):
        rank_map = {h["text"]: r for r, h in enumerate(hits)}
        rank_maps.append(rank_map)

    all_docs = {}
    for hits in (bm25_hits, dense_hits):
        for h in hits:
            all_docs[h["text"]] = h

    scored: List[Tuple[float, str]] = []
    for text in all_docs:
        s = 0.0
        for rm in rank_maps:
            if text in rm:
                s += 1.0 / (rrf_k + rm[text] + 1.0)
        scored.append((s, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    out: List[Dict] = []
    for s, t in top:
        base = all_docs[t].copy()
        base["score"] = float(s)
        out.append(base)
    return out

class HybridRetriever:
    def __init__(self, bm25_kb_path="data/kb/bm25.jsonl", chroma_dir="data/chroma", chroma_collection="osha"):
        self.bm25 = BM25Client(bm25_kb_path)
        self.vec = VectorClient(persist_dir=chroma_dir, collection=chroma_collection)

    def search(self, query: str, k: int = 6, fanout: int = 20) -> List[Dict]:
        bm25_hits = self.bm25.search(query, k=fanout)
        vec_hits = self.vec.search(query, k=fanout)
        return reciprocal_rank_fusion(bm25_hits, vec_hits, k=k, rrf_k=60.0)