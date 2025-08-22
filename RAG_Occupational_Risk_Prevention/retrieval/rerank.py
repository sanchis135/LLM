
from typing import List, Dict

class CrossEncoderReranker:
    def __init__(self) -> None:
        self.enabled = True

    def rerank(self, query: str, passages: List[Dict]) -> List[Dict]:
        return sorted(passages, key=lambda x: x.get("score", 0.0), reverse=True)
