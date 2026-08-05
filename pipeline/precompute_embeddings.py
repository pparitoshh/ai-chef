"""Precompute recipe embeddings for the in-memory search backend.

Same embedding text as pipeline/ingest.py (name | cuisine | dish_type |
ingredients), so results are consistent with the pgvector path. Output is
committed to the repo so the deployed app never has to embed 10k recipes
at cold start — only the user's query gets embedded at request time.

Usage:  uv run --group dev python -m pipeline.precompute_embeddings
"""

import json
import os
from time import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/recipes_10k.csv")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OUT_PATH = "data/recipes_10k_embeddings.npy"
BATCH_SIZE = 64


def build_embedding_text(row):
    ingredients = " ".join(json.loads(row["ingredients"]))
    return f"{row['name']} | {row['cuisine']} | {row['dish_type']} | {ingredients}"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} recipes from {DATA_PATH}")

    texts = df.apply(build_embedding_text, axis=1).tolist()

    print(f"Embedding with {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    t0 = time()
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    print(f"  embedded {len(embeddings):,} texts in {time() - t0:.1f}s "
          f"(dim={embeddings.shape[1]})")

    np.save(OUT_PATH, embeddings)
    print(f"saved {embeddings.shape} to {OUT_PATH}")


if __name__ == "__main__":
    main()
