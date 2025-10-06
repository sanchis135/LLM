_SYNS = {
    'canary': ['progressive delivery','weighted routing','percentage traffic'],
    'rollout': ['deployment','blue-green','upgrade'],
    'monitoring': ['observability','metrics','logging','tracing'],
    'drift': ['concept drift','data drift','shift']
}


def rewrite_query(q: str) -> str:
    ql = q.lower()
    expansions = []
    for k, syns in _SYNS.items():
        if k in ql:
            expansions.extend(syns)
    return q if not expansions else q + ' OR ' + ' OR '.join(expansions)