"""RAG flow: query rewrite → hybrid retrieval → Groq answer → relevance judge.

Shared by the RAG evaluation (phase 4) and the FastAPI app (phase 5).
"""

import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

from api.search import hybrid_rerank_search

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()  # reads GROQ_API_KEY from env
    return _client


# --- 1. query rewriting ------------------------------------------------------

REWRITE_PROMPT = """You rewrite a user's conversational food request into a recipe
search query plus structured filters.

User request: "{user_text}"

Return a JSON object with:
- "search_query": a short keyword-style query capturing what the user wants
  (ingredients, dish, style) — no filler words.
- "cuisine": one of {cuisines}, or null.
- "dish_type": one of {dish_types}, or null.
- "diet": "vegan" | "vegetarian" | "non-veg", or null. If the user names a meat
  (chicken, pork, beef, fish, ...), use "non-veg".
- "skill_level": "beginner" | "intermediate" | "master_chef", or null. Map
  "can cook"/"decent cook" → intermediate, "pro"/"chef" → master_chef.
- "protein": "chicken" | "pork" | "beef", or null (only when explicitly named).

Only set filters the user actually stated or clearly implied; else null.
Reply with JSON only."""

CUISINES = ("american, cajun, caribbean, chinese, french, german, greek, indian, "
            "italian, japanese, korean, mexican, middle-eastern, moroccan, "
            "spanish, tex-mex, thai, vietnamese")
DISH_TYPES = ("appetizer, bbq & grilling, beverage, bread & baking, breakfast, "
              "casserole, curry, dessert, one-pot, other, pasta, pizza & flatbread, "
              "rice & grains, salad, sandwich, snack, soups & stews, stir-fry")
FILTER_KEYS = ("cuisine", "dish_type", "diet", "skill_level", "protein")


def rewrite_query(user_text: str) -> dict:
    """Conversational input → {"query": str, "filters": {...}} via Groq JSON mode."""
    resp = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(
            user_text=user_text, cuisines=CUISINES, dish_types=DISH_TYPES)}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=200,
    )
    data = json.loads(resp.choices[0].message.content)
    filters = {k: data[k] for k in FILTER_KEYS if data.get(k)}
    return {"query": data.get("search_query") or user_text, "filters": filters}


# --- 2. answer generation ----------------------------------------------------

PROMPT_VARIANTS = {
    "concise": """You are AI Chef, a friendly cooking assistant. Based ONLY on the
retrieved recipes below, answer the user's request: recommend the best
match(es), say briefly why each fits, and mention cuisine, time and skill
level. Keep it short and warm. Do not invent recipes.

User request: {question}

Retrieved recipes:
{context}""",
    "detailed": """You are AI Chef, an expert cooking assistant. Based ONLY on the
retrieved recipes below:

1. Start with the single best recommendation for the user's request, and
   explain why it matches their cuisine, diet, time and skill constraints.
2. Give a brief overview of its key ingredients and cooking approach.
3. Suggest 1-2 similar alternatives from the list ("you might also like").
4. If nothing fits well, say so honestly instead of forcing a match.

Do not invent recipes or details not present below.

User request: {question}

Retrieved recipes:
{context}""",
}


def build_context(recipes) -> str:
    return "\n".join(
        f"{i}. {r.name} — cuisine: {r.cuisine}, dish: {r.dish_type}, "
        f"diet: {r.diet}, skill: {r.skill_level}, time: {r.minutes} min"
        for i, r in enumerate(recipes, 1)
    )


def generate_answer(question: str, model: str = GROQ_MODEL,
                    variant: str = "concise", k: int = 5,
                    filters: dict | None = None) -> dict:
    """Full RAG: retrieve with the winning hybrid+rerank approach, then answer."""
    recipes = hybrid_rerank_search(question, k=k, filters=filters)
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


# --- 3. LLM-as-a-judge --------------------------------------------------------

JUDGE_PROMPT = """You are evaluating a recipe-recommendation assistant.

User request: {question}

Assistant answer:
{answer}

Is the answer relevant and useful for the user's request (right cuisine / dish
type / diet / skill fit, grounded in real recommendations, not evasive)?
Reply with exactly one word: RELEVANT, PARTLY_RELEVANT or NON_RELEVANT."""


def judge_relevance(question: str, answer: str) -> str:
    resp = get_client().chat.completions.create(
        model=GROQ_JUDGE_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, answer=answer)}],
        temperature=0,
        max_tokens=10,
    )
    verdict = resp.choices[0].message.content.strip().upper()
    for label in ("NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"):
        if label in verdict:
            return label
    return "UNKNOWN"
