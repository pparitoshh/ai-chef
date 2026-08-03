"""Smoke tests: vector search, full-text search, and filters on the recipes table.

Prereq: postgres running + ingestion done.
Usage:  uv run python -m tests.test_search
"""

import psycopg2
from sentence_transformers import SentenceTransformer

conn = psycopg2.connect(host="localhost", dbname="ai_chef", user="chef", password="chef")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Vector search ----------------------------------------------------------
q = model.encode(["spicy indian chicken curry"], normalize_embeddings=True)[0].tolist()
with conn.cursor() as cur:
    cur.execute(
        """
        SELECT name, cuisine, diet, 1 - (embedding <=> %s::vector) AS sim
        FROM recipes ORDER BY embedding <=> %s::vector LIMIT 5
        """,
        (q, q),
    )
    print("== VECTOR: 'spicy indian chicken curry' ==")
    for name, cuisine, diet, sim in cur.fetchall():
        print(f"  {sim:.3f}  {name}  [{cuisine}|{diet}]")

# 2. Full-text search --------------------------------------------------------
with conn.cursor() as cur:
    cur.execute(
        """
        SELECT name, cuisine, ts_rank(fts, plainto_tsquery('english', 'mexican tacos')) AS rank
        FROM recipes
        WHERE fts @@ plainto_tsquery('english', 'mexican tacos')
        ORDER BY rank DESC LIMIT 5
        """
    )
    print("\n== FTS: 'mexican tacos' ==")
    for name, cuisine, rank in cur.fetchall():
        print(f"  {rank:.3f}  {name}  [{cuisine}]")

# 3. Filters ------------------------------------------------------------------
with conn.cursor() as cur:
    cur.execute("SELECT count(*) FROM recipes WHERE cuisine='italian' AND diet='vegetarian'")
    print("\n== FILTER: italian + vegetarian ==", cur.fetchone()[0], "recipes")

    cur.execute("SELECT count(*) FROM recipes WHERE proteins ? 'chicken' AND cuisine='mexican'")
    print("== FILTER: chicken + mexican ==", cur.fetchone()[0], "recipes")

    cur.execute("SELECT count(*) FROM recipes WHERE skill_level='beginner' AND dish_type='one-pot'")
    print("== FILTER: beginner + one-pot ==", cur.fetchone()[0], "recipes")

conn.close()
print("\nAll smoke tests passed.")
