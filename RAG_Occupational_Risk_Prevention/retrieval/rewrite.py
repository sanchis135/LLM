
def rewrite_query(query: str) -> str:
    q = query.strip()
    if len(q) < 12 and not q.endswith(" PRL"):
        q += " PRL"
    return q
