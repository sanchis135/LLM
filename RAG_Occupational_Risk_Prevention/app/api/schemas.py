
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_hybrid: bool = True
    use_rerank: bool = True
    rewrite: bool = True

class Passage(BaseModel):
    text: str
    score: float
    source: str
    meta: Dict[str, Any] = {}

class AnswerResponse(BaseModel):
    answer: str
    passages: List[Passage]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
