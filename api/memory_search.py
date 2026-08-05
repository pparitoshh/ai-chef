"""In-memory hybrid search: no Postgres/pgvector required.

Loads data/recipes_10k.csv + precomputed embeddings (pipeline/precompute_
embeddings.py) into memory once, then serves vector (cosine) + keyword
(TF-IDF) search fused with reciprocal rank fusion — same shape/approach as
api/search.py's hybrid_search, just against in-memory arrays instead of
Postgres. Used by the Streamlit Cloud deployment, which has no reachable
database. The pgvector path in api/search.py is untouched and still used
by the local Docker stack.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", "data/recipes_10k.csv")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "data/recipes_10k_embeddings.npy")
CANDIDATES = 20
RRF_K = 60

_embedder = None
_df = None
_embeddings = None
_tfidf = None
_tfidf_matrix = None


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


def _build_embedding_text(row) -> str:
    ingredients = " ".join(json.loads(row["ingredients"]))
    return f"{row['name']} | {row['cuisine']} | {row['dish_type']} | {ingredients}"


def _load():
    """Lazily load the CSV + embeddings + TF-IDF index once per process."""
    global _df, _embeddings, _tfidf, _tfidf_matrix
    if _df is not None:
        return
    _df = pd.read_csv(DATA_PATH)
    _embeddings = np.load(EMBEDDINGS_PATH)
    if len(_df) != _embeddings.shape[0]:
        raise ValueError(
            f"recipes ({len(_df)}) and embeddings ({_embeddings.shape[0]}) "
            "row-count mismatch — regenerate with pipeline/precompute_embeddings.py"
        )
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = _df.apply(_build_embedding_text, axis=1).tolist()
    _tfidf = TfidfVectorizer(stop_words="english")
    _tfidf_matrix = _tfidf.fit_transform(texts)


def get_embedder():
    """ONNX embedder (no torch) — downloads the model once, then caches on disk."""
    global _embedder
    if _embedder is None:
        from api.onnx_models import MODELS_DIR, OnnxEmbedder

        embedder_dir = MODELS_DIR / "embedder"
        if not (embedder_dir / "model.onnx").exists():
            _download_embedder(embedder_dir)
        _embedder = OnnxEmbedder(embedder_dir)
    return _embedder


def _download_embedder(embedder_dir: Path):
    import urllib.request

    embedder_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "model.onnx":
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
        "tokenizer.json":
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
    }
    for name, url in files.items():
        urllib.request.urlretrieve(url, embedder_dir / name)


def _matches_filters(row, filters: dict | None) -> bool:
    if not filters:
        return True
    for col in ("cuisine", "dish_type", "diet", "skill_level"):
        if filters.get(col) and row[col] != filters[col]:
            return False
    if filters.get("protein"):
        proteins = json.loads(row["proteins"]) if isinstance(row["proteins"], str) else row["proteins"]
        if filters["protein"] not in proteins:
            return False
    return True


def _to_recipe(row, score: float) -> Recipe:
    return Recipe(
        id=row["id"], name=row["name"], cuisine=row["cuisine"],
        dish_type=row["dish_type"], diet=row["diet"],
        skill_level=row["skill_level"], minutes=int(row["minutes"]), score=score,
    )


def get_recipe_detail(recipe_id: str) -> dict | None:
    _load()
    matches = _df[_df["id"] == recipe_id]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {
        "id": row["id"], "name": row["name"], "cuisine": row["cuisine"],
        "dish_type": row["dish_type"], "diet": row["diet"],
        "skill_level": row["skill_level"], "minutes": int(row["minutes"]),
        "calories": float(row["calories"]) if pd.notna(row["calories"]) else None,
        "ingredients": json.loads(row["ingredients"]),
        "steps": json.loads(row["steps"]),
    }


def vector_search(query: str, k: int = 5, filters: dict | None = None) -> list[Recipe]:
    _load()
    q_emb = get_embedder().encode([query], normalize_embeddings=True)[0]
    sims = _embeddings @ q_emb
    mask = _df.apply(lambda r: _matches_filters(r, filters), axis=1).to_numpy()
    sims = np.where(mask, sims, -np.inf)
    top_idx = np.argsort(-sims)[:k]
    return [_to_recipe(_df.iloc[i], float(sims[i])) for i in top_idx if sims[i] > -np.inf]


def text_search(query: str, k: int = 5, filters: dict | None = None) -> list[Recipe]:
    _load()
    q_vec = _tfidf.transform([query])
    sims = (_tfidf_matrix @ q_vec.T).toarray().ravel()
    mask = _df.apply(lambda r: _matches_filters(r, filters), axis=1).to_numpy()
    sims = np.where(mask, sims, -np.inf)
    top_idx = np.argsort(-sims)[:k]
    return [_to_recipe(_df.iloc[i], float(sims[i])) for i in top_idx if sims[i] > 0]


def _rrf_merge(result_lists: list[list[Recipe]], k: int) -> list[Recipe]:
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
    vec = vector_search(query, k=CANDIDATES, filters=filters)
    txt = text_search(query, k=CANDIDATES, filters=filters)
    return _rrf_merge([vec, txt], k)
