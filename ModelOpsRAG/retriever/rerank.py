try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class SimpleReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name) if CrossEncoder else None


def rerank(self, query: str, candidates, top_k: int = 10):
    if self.model is None:
        return candidates[:top_k]
    scores = self.model.predict([[query, text] for _, text in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for (c, s) in ranked[:top_k]]