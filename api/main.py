"""FastAPI app: /recommend, /feedback, /health.

Flow per /recommend: rewrite query (Groq) → hybrid+rerank search with filters
→ Groq answer → inline LLM-as-judge relevance → log conversation to Postgres.

Run:  uv run uvicorn api.main:app --reload
"""

import time

from fastapi import FastAPI, HTTPException

from api import db, rag
from api.schemas import (FeedbackRequest, FeedbackResponse, RecipeOut,
                         RecommendRequest, RecommendResponse)
from api.search import get_connection

app = FastAPI(title="AI Chef", version="0.1.0")


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t0 = time.time()
    rewritten = rag.rewrite_query(req.question)
    out = rag.generate_answer(req.question, k=req.k, filters=rewritten["filters"])
    if not out["recipes"] and rewritten["filters"]:
        # filters were too strict (zero matches) — retry unfiltered
        out = rag.generate_answer(req.question, k=req.k)
    relevance = rag.judge_relevance(req.question, out["answer"])
    elapsed = time.time() - t0

    conv_id = db.log_conversation(
        question=req.question, answer=out["answer"], model_used=out["model"],
        response_time_s=elapsed, relevance=relevance,
        prompt_tokens=out["prompt_tokens"],
        completion_tokens=out["completion_tokens"],
        total_tokens=out["tokens"], filters=rewritten["filters"],
    )
    return RecommendResponse(
        conversation_id=conv_id,
        answer=out["answer"],
        recipes=[RecipeOut(**r.__dict__) for r in out["recipes"]],
        rewritten_query=rewritten["query"],
        filters=rewritten["filters"],
        relevance=relevance,
        total_tokens=out["tokens"],
        response_time_ms=round(1000 * elapsed, 1),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    if req.feedback not in (1, -1):
        raise HTTPException(status_code=422, detail="feedback must be +1 or -1")
    try:
        db.log_feedback(req.conversation_id, req.feedback)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"unknown conversation_id: {e}")
    return FeedbackResponse()


@app.get("/health")
def health():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM recipes")
        n = cur.fetchone()[0]
    return {"status": "ok", "recipes": n}
