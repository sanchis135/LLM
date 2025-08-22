from typing import List, Dict

TEMPLATE = """
Answer the question using ONLY the passages cited. 
If the answer is not present, reply: "Not found in the indexed sources".
Include references with [source] at the end.

Question: {query}

Passages:
{contexts}

Answer in English, technical and concise.
"""

def build_prompt(query: str, passages: List[Dict]) -> str:
    ctx = "\n\n".join([f"[{i+1}] {p['text']}\n(Source: {p.get('source','?')})" for i, p in enumerate(passages)])
    return TEMPLATE.format(query=query, contexts=ctx)
