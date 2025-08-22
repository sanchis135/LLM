# evaluation/metrics.py
from __future__ import annotations
from typing import List, Sequence
import math

def recall_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """
    gold: lista de sources correctas (IDs tipo file#chunkX)
    retrieved: lista de sources devueltas por el sistema ordenadas por ranking
    """
    if not gold:
        return 0.0
    topk = set(retrieved[:k])
    gold_set = set(gold)
    return len(gold_set & topk) / len(gold_set)

def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """
    relevances: lista de 0/1 (o ganancia >=0) en el orden del ranking.
    """
    rel = relevances[:k]
    dcg = 0.0
    for i, r in enumerate(rel, start=1):
        dcg += (2**r - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """
    gold binario: 1 si source está en gold, 0 si no.
    """
    if not retrieved:
        return 0.0
    rel = [1 if s in set(gold) else 0 for s in retrieved]
    dcg = dcg_at_k(rel, k)
    ideal = sorted(rel, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return (dcg / idcg) if idcg > 0 else 0.0
