
from fastapi import FastAPI
from app.api.schemas import QueryRequest, AnswerResponse, Passage
from retrieval.hybrid import HybridRetriever
from retrieval.prompt_builder import build_prompt

app = FastAPI(title="RAG-PRL API")

retriever = HybridRetriever()  # lazy init inside

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=AnswerResponse)
def query_api(req: QueryRequest):
    passages = retriever.search(
        req.query,
        top_k=req.top_k,
        use_hybrid=req.use_hybrid,
        use_rerank=req.use_rerank,
        rewrite=req.rewrite
    )

    prompt = build_prompt(req.query, passages)

    # Placeholder de respuesta (completa con tu proveedor LLM)
    answer = (
        "🔧 LLM no configurado. Este es un placeholder.\n\n"
        "Contexto recuperado:\n- "
        + "\n- ".join([p["text"][:160].replace("\n", " ") + "..." for p in passages])
        + "\n\nConfigura tu proveedor LLM en .env y completa la llamada."
    )

    return AnswerResponse(
        answer=answer,
        passages=[Passage(**p) for p in passages],
        prompt_tokens=None,
        completion_tokens=None,
    )
