"""Retrieval approaches for AI Chef: text, vector, hybrid, hybrid + re-rank.

Shared by the evaluation script (phase 3) and the FastAPI app (phase 5).
"""

import os
from dataclasses import dataclass, field

import psycopg2
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20  # per-method candidate pool before fusion / re-ranking
RRF_K = 60  # reciprocal-rank-fusion constant

_embedding_model = None
_rerank_model = None


@dataclass
class Recipe:
    id: str
    name: str
    cuisine: str
    dish_type: str
    diet: str
    skill_level: str
    minutes: int
    score: float = 0.0
    extra: dict = field(default_factory=dict)


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "ai_chef"),
        user=os.getenv("POSTGRES_USER", "chef"),
        password=os.getenv("POSTGRES_PASSWORD", "chef"),
    )


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder

        _rerank_model = CrossEncoder(RERANK_MODEL)
    return _rerank_model


def _filter_clause(filters: dict | None):
    """Build WHERE clause + params from optional structured filters."""
    clause, params = "", []
    if not filters:
        return clause, params
    for col in ("cuisine", "dish_type", "diet", "skill_level"):
        if filters.get(col):
            clause += f" AND {col} = %s"
            params.append(filters[col])
    if filters.get("protein"):
        clause += " AND proteins ? %s"
        params.append(filters["protein"])
    return clause, params


def _rows_to_recipes(rows) -> list[Recipe]:
    return [
        Recipe(
            id=r[0], name=r[1], cuisine=r[2], dish_type=r[3],
            diet=r[4], skill_level=r[5], minutes=r[6], score=r[7],
        )
        for r in rows
    ]


def vector_search(query: str, k: int = 5, filters: dict | None = None) -> list[Recipe]:
    """Semantic search: cosine similarity over pgvector embeddings."""
    q = get_embedding_model().encode([query], normalize_embeddings=True)[0].tolist()
    clause, params = _filter_clause(filters)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, cuisine, dish_type, diet, skill_level, minutes,
                   1 - (embedding <=> %s::vector) AS score
            FROM recipes WHERE TRUE {clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            [q, *params, q, k],
        )
        return _rows_to_recipes(cur.fetchall())


def text_search(query: str, k: int = 5, filters: dict | None = None) -> list[Recipe]:
    """Keyword search: Postgres FTS over name/cuisine/dish/ingredients.

    Natural-language questions ANDed term-by-term would match nothing, so the
    parsed tsquery is rewritten to OR semantics and ranked by ts_rank.
    """
    clause, params = _filter_clause(filters)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH q AS (
                SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery AS tq
            )
            SELECT id, name, cuisine, dish_type, diet, skill_level, minutes,
                   ts_rank(fts, q.tq) AS score
            FROM recipes, q
            WHERE fts @@ q.tq {clause}
            ORDER BY score DESC
            LIMIT %s
            """,
            [query, *params, k],
        )
        return _rows_to_recipes(cur.fetchall())


def _rrf_merge(result_lists: list[list[Recipe]], k: int) -> list[Recipe]:
    """Reciprocal rank fusion over several ranked lists."""
    scores: dict[str, float] = {}
    recipes: dict[str, Recipe] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (RRF_K + rank + 1)
            recipes[r.id] = r
    merged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for rid, score in merged:
        r = recipes[rid]
        r.score = score
        out.append(r)
    return out


def hybrid_search(query: str, k: int = 5, filters: dict | None = None) -> list[Recipe]:
    """Hybrid: RRF fusion of vector and full-text candidate lists."""
    vec = vector_search(query, k=CANDIDATES, filters=filters)
    txt = text_search(query, k=CANDIDATES, filters=filters)
    return _rrf_merge([vec, txt], k)


def hybrid_rerank_search(query: str, k: int = 5, filters: dict | None = None) -> list[Recipe]:
    """Hybrid + re-rank: RRF top-20 candidates re-scored by a cross-encoder."""
    candidates = hybrid_search(query, k=CANDIDATES, filters=filters)
    if not candidates:
        return []
    model = get_rerank_model()
    pairs = [(query, f"{r.name} | {r.cuisine} | {r.dish_type} | {r.diet}") for r in candidates]
    scores = model.predict(pairs)
    for r, s in zip(candidates, scores):
        r.score = float(s)
    return sorted(candidates, key=lambda r: r.score, reverse=True)[:k]


APPROACHES = {
    "text": text_search,
    "vector": vector_search,
    "hybrid": hybrid_search,
    "hybrid+rerank": hybrid_rerank_search,
}
