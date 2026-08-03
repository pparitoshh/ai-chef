"""Ingest the 10K recipe dataset into PostgreSQL + pgvector.

- Reads data/recipes_10k.csv
- Embeds `name + cuisine + dish_type + ingredients` with sentence-transformers
- Upserts into the `recipes` table (vector + full-text indexes in init_db.sql)

Prereq: docker compose up -d postgres
Usage:  uv run python -m pipeline.ingest
"""

import json
import os
from time import time

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/recipes_10k.csv")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = 64

UPSERT_SQL = """
INSERT INTO recipes
    (id, name, cuisine, dish_type, diet, proteins, skill_level,
     minutes, calories, n_steps, ingredients, steps, tags,
     ingredients_text, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    cuisine = EXCLUDED.cuisine,
    dish_type = EXCLUDED.dish_type,
    diet = EXCLUDED.diet,
    proteins = EXCLUDED.proteins,
    skill_level = EXCLUDED.skill_level,
    minutes = EXCLUDED.minutes,
    calories = EXCLUDED.calories,
    n_steps = EXCLUDED.n_steps,
    ingredients = EXCLUDED.ingredients,
    steps = EXCLUDED.steps,
    tags = EXCLUDED.tags,
    ingredients_text = EXCLUDED.ingredients_text,
    embedding = EXCLUDED.embedding;
"""


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "ai_chef"),
        user=os.getenv("POSTGRES_USER", "chef"),
        password=os.getenv("POSTGRES_PASSWORD", "chef"),
    )


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
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  embedded {len(embeddings):,} texts in {time() - t0:.1f}s "
          f"(dim={embeddings.shape[1]})")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for i, (_, row) in enumerate(df.iterrows()):
                ingredients_text = " ".join(json.loads(row["ingredients"]))
                cur.execute(
                    UPSERT_SQL,
                    (
                        row["id"], row["name"], row["cuisine"], row["dish_type"],
                        row["diet"], row["proteins"], row["skill_level"],
                        int(row["minutes"]), float(row["calories"]), int(row["n_steps"]),
                        row["ingredients"], row["steps"], row["tags"],
                        ingredients_text, embeddings[i].tolist(),
                    ),
                )
                if (i + 1) % 1000 == 0:
                    conn.commit()
                    print(f"  inserted {i + 1:,} / {len(df):,}")
            conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM recipes")
            print(f"recipes table now has {cur.fetchone()[0]:,} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
