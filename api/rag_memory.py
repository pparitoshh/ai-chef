"""RAG flow for the in-memory backend: same Groq prompts as api/rag.py,
retrieval via api/memory_search.hybrid_search instead of Postgres.
"""

import time

from api.memory_search import hybrid_search
from api.rag import (GROQ_MODEL, PROMPT_VARIANTS, build_context, get_client,
                     judge_relevance, rewrite_query)

__all__ = ["rewrite_query", "judge_relevance", "generate_answer"]


def generate_answer(question: str, model: str = GROQ_MODEL,
                    variant: str = "detailed", k: int = 5,
                    filters: dict | None = None) -> dict:
    recipes = hybrid_search(question, k=k, filters=filters)
    prompt = PROMPT_VARIANTS[variant].format(
        question=question, context=build_context(recipes))
    t0 = time.time()
    resp = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    usage = resp.usage
    return {
        "answer": resp.choices[0].message.content.strip(),
        "recipes": recipes,
        "model": model,
        "variant": variant,
        "latency_ms": round(1000 * (time.time() - t0), 1),
        "tokens": usage.total_tokens if usage else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }
