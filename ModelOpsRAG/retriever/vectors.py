# retriever/vectors.py
import os, json
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Tuple

# Optional Ollama embeddings
try:
    from retriever.ollama_embed import OllamaEmbeddingFunction
except Exception:
    OllamaEmbeddingFunction = None


class VectorIndex:
    def __init__(self, collection_name: str = 'modelops', persist_dir: str | None = None):
        # Elegir embedding function
        backend = os.getenv('EMB_BACKEND', 'sbert').lower()
        if backend == 'ollama' and OllamaEmbeddingFunction is not None:
            emb_fn = OllamaEmbeddingFunction(
                host=os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'),
                model=os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
            )
        else:
            model_name = os.getenv('EMB_MODEL', 'all-MiniLM-L6-v2')
            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        # Cliente + colección con embedding_function fijado
        self.client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=emb_fn
        )

    def upsert_jsonl(self, path: str):
        ids, texts, metas = [], [], []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                ids.append(f"vec-{i}")
                texts.append(obj['text'])
                meta = {k: obj[k] for k in obj if k != 'text'}
                meta['row_index'] = i              # 👈 clave para mapear a docs originales
                metas.append(meta)
        if ids:
            self.collection.upsert(ids=ids, documents=texts, metadatas=metas)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        res = self.collection.query(query_texts=[query], n_results=k)
        out = []
        if res and res['ids']:
            for i in range(len(res['ids'][0])):
                out.append((res['ids'][0][i],
                            float(res['distances'][0][i]) if 'distances' in res else 0.0))
        return out

    def get_by_id(self, _id: str):
        """Devuelve (metadata, document) para un id 'vec-*'."""
        got = self.collection.get(ids=[_id], include=['metadatas', 'documents'])
        # Estructura esperada en Chroma 0.5:
        # got = {'ids': ['vec-1'], 'metadatas': [ {...} ], 'documents': [ 'text...' ]}
        if not got or not got.get('ids'):
            return None, None
        mds = got.get('metadatas') or []
        docs = got.get('documents') or []
        md = mds[0] if len(mds) > 0 else {}
        doc = docs[0] if len(docs) > 0 else ''
        return md, doc

