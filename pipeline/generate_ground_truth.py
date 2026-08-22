"""Generate ground-truth user questions for retrieval evaluation.

For a stratified sample of recipes, asks Groq to write one realistic user
question that the recipe would perfectly answer (mentions cuisine / dish type /
diet / skill in natural language, the way a user of AI Chef would talk).

Output: data/ground_truth.csv with columns [recipe_id, question].
The script is resumable: recipe_ids already present in the output are skipped.

Usage:  uv run python -m pipeline.generate_ground_truth [--n 1000]
"""

import argparse
import csv
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/recipes_10k.csv")
OUT_PATH = Path("data/ground_truth.csv")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
SEED = 42
# free tier ~30 RPM for openai/gpt-oss-20b — pace ourselves under that
MIN_INTERVAL_S = 2.2

PROMPT = """You are simulating a hungry user of a recipe recommendation app.

Given this recipe:
- Name: {name}
- Cuisine: {cuisine}
- Dish type: {dish_type}
- Diet: {diet}
- Skill level: {skill_level}
- Time: {minutes} minutes
- Key ingredients: {ingredients}

Write ONE short, natural question or request this user might ask for which the
recipe above is the perfect answer. Mention some of: the cuisine, the kind of
dish, the diet/protein, the time available, or the skill level — the way a real
person would phrase it. Do NOT use the exact recipe name.

Reply with the question only, no quotes, no preamble."""


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sqrt-proportional stratified sample across cuisines (same idea as phase 1)."""
    counts = df["cuisine"].value_counts()
    weights = counts**0.5
    alloc = (weights / weights.sum() * n).round().astype(int)
    parts = []
    for cuisine, k in alloc.items():
        pool = df[df["cuisine"] == cuisine]
        parts.append(pool.sample(n=min(k, len(pool)), random_state=SEED))
    return pd.concat(parts).sample(frac=1.0, random_state=SEED)  # shuffle


def build_prompt(row) -> str:
    import json

    ingredients = ", ".join(json.loads(row["ingredients"])[:8])
    return PROMPT.format(
        name=row["name"],
        cuisine=row["cuisine"],
        dish_type=row["dish_type"],
        diet=row["diet"],
        skill_level=row["skill_level"],
        minutes=row["minutes"],
        ingredients=ingredients,
    )


def generate_question(client: Groq, prompt: str, max_retries: int = 6) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=60,
            )
            q = resp.choices[0].message.content.strip().strip('"').strip()
            return " ".join(q.splitlines())  # single line
        except RateLimitError:
            wait = min(2**attempt * 5, 60)
            print(f"  rate limited, waiting {wait}s ...")
            time.sleep(wait)
    raise RuntimeError("too many rate-limit retries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="sample size")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    done_ids = set()
    if args.out.exists():
        with args.out.open() as f:
            done_ids = {r["recipe_id"] for r in csv.DictReader(f)}
        print(f"Resuming: {len(done_ids)} questions already generated")

    sample = stratified_sample(df, args.n)
    sample = sample[~sample["id"].astype(str).isin(done_ids)]
    print(f"Generating questions for {len(sample)} recipes with {MODEL} ...")

    client = Groq()  # reads GROQ_API_KEY from env
    new_file = not args.out.exists()
    with args.out.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["recipe_id", "question"])
        last = 0.0
        for i, (_, row) in enumerate(sample.iterrows()):
            dt = time.time() - last
            if dt < MIN_INTERVAL_S:
                time.sleep(MIN_INTERVAL_S - dt)
            q = generate_question(client, build_prompt(row))
            last = time.time()
            writer.writerow([row["id"], q])
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(sample)} done")

    total = sum(1 for _ in args.out.open()) - 1
    print(f"Done. {args.out} now holds {total} question pairs.")


if __name__ == "__main__":
    main()
