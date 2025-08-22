from __future__ import annotations
from typing import List, Dict, Optional
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


def _to_similarity(distance: float) -> float:
    # Chroma returns distance (lower = better). Convert to approximate similarity 0..1
    try:
        return 1.0 / (1.0 + float(distance))
    except Exception:
        return 0.0


def _sanitize_meta(meta: Optional[dict]) -> dict:
    """Chroma does not accept None in metadata. Convert and remove None values."""
    out: Dict[str, object] = {}
    if not meta:
        return out
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


class VectorClient:
    def __init__(self, persist_dir: str = "data/chroma", collection: str = "osha", model_name: Optional[str] = None):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.model_name = model_name or "intfloat/multilingual-e5-small"
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        self.col = self.client.get_or_create_collection(name=collection, embedding_function=self.embedder)

    # ------- Search (E5 requires prefixes 'query:' / 'passage:') -------
    def search(self, query: str, k: int = 5) -> List[Dict]:
        res = self.col.query(
            query_texts=[f"query: {query}"],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs  = (res.get("documents")  or [[]])[0]
        metas = (res.get("metadatas")  or [[]])[0]
        dists = (res.get("distances")  or [[]])[0]
        out: List[Dict] = []
        for t, m, d in zip(docs, metas, dists):
            out.append({
                "text": t,  # may come prefixed with 'passage:' (that’s fine)
                "score": _to_similarity(d),
                "source": (m or {}).get("source", "?"),
                "meta": m or {}
            })
        return out

    # ------- Indexing from JSONL (BM25 KB) -------
    def index_from_jsonl(self, kb_jsonl: str | Path, batch_size: int = 512) -> int:
        """
        Reads JSONL lines with fields: text, source, meta{file,family,year}, and uploads them to Chroma.
        - Adds 'passage:' prefix to the text (recommended for E5).
        - Copies metadata and 'source' into Chroma metadata (no None values).
        """
        kb_jsonl = Path(kb_jsonl)
        if not kb_jsonl.exists():
            raise FileNotFoundError(f"KB JSONL does not exist: {kb_jsonl}")

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict] = []
        n = 0

        def flush():
            if ids:
                self.col.add(ids=ids, documents=docs, metadatas=metas)
                ids.clear(); docs.clear(); metas.clear()

        with kb_jsonl.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text   = obj.get("text", "") or ""
                source = obj.get("source", f"{kb_jsonl.name}#{i}")
                meta   = obj.get("meta", {}) or {}

                if not text.strip():
                    # skip empty chunks
                    continue

                # prepare sanitized metadata
                # ensure 'year' is int if possible, and always add 'source'
                year = meta.get("year")
                if year is not None:
                    try:
                        meta["year"] = int(year)
                    except Exception:
                        meta["year"] = str(year)
                sanitized = _sanitize_meta(meta)
                sanitized["source"] = str(source)

                ids.append(str(source))                 # unique id per chunk
                docs.append(f"passage: {text}")         # E5: 'passage:' prefix
                metas.append(sanitized)
                n += 1

                # batch flush
                if len(ids) >= batch_size:
                    flush()

        # final flush
        flush()
        return n

    def reset_collection(self):
        """Clear the current collection (⚠️ deletes everything)."""
        name = self.col.name
        self.client.delete_collection(name)
        # recreate
        self.col = self.client.get_or_create_collection(name=name, embedding_function=self.embedder)
