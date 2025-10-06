from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import json


class BM25Index:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        corpus = [ (d['text']).lower().split() for d in docs ]
        self.bm25 = BM25Okapi(corpus)


    def search(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [ (self.docs[i]['_id'], float(s)) for i, s in ranked ]