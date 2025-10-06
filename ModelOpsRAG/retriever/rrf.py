from collections import defaultdict
from typing import Dict, List, Tuple


def rrf_fuse(results: Dict[str, List[Tuple[str, float]]], k: int = 60):
    ranks = defaultdict(float)
    for _, arr in results.items():
        for rank, (doc_id, _) in enumerate(arr, start=1):
            ranks[doc_id] += 1.0 / (k + rank)
    return sorted(ranks.items(), key=lambda x: x[1], reverse=True)