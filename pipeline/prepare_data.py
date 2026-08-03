"""Prepare the AI Chef 10K recipe dataset from Food.com RAW_recipes.

Reads:   data/RAW_recipes.csv  (231k recipes: name, minutes, tags, calories, steps, ingredients)
Writes:  data/recipes_10k.csv  (10k stratified sample with derived fields)

Derived fields:
  cuisine     - from cuisine tags (priority order, specific first)
  dish_type   - from course/preparation tags
  diet        - vegan / vegetarian (from tags) / non-veg
  proteins    - multi-label meat detection from ingredients (chicken, beef, pork, fish_seafood, other_meat)
  skill_level - beginner / intermediate / master_chef (tags + steps + time)

Usage: uv run python -m pipeline.prepare_data
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/RAW_recipes.csv")
OUT_PATH = Path("data/recipes_10k.csv")
SAMPLE_SIZE = 10_000
SEED = 42

# Priority order: distinctive/specific cuisines first, broad ones last.
CUISINES = [
    "indian", "mexican", "italian", "chinese", "thai", "japanese",
    "french", "greek", "spanish", "middle-eastern", "moroccan",
    "caribbean", "german", "tex-mex", "cajun", "korean", "vietnamese",
    "american",
]

DISH_TYPE_TAGS = [
    ("one-dish-meal", "one-pot"),
    ("curries", "curry"),
    ("stir-fry", "stir-fry"),
    ("soups-stews", "soups & stews"),
    ("pasta", "pasta"),
    ("casseroles", "casserole"),
    ("salads", "salad"),
    ("sandwiches", "sandwich"),
    ("pizza", "pizza & flatbread"),
    ("rice", "rice & grains"),
    ("barbecue", "bbq & grilling"),
    ("grilling", "bbq & grilling"),
    ("breakfast", "breakfast"),
    ("appetizers", "appetizer"),
    ("snacks", "snack"),
    ("breads", "bread & baking"),
    ("desserts", "dessert"),
    ("beverages", "beverage"),
]

PROTEIN_PATTERNS = {
    "chicken": r"\bchicken\b",
    "beef": r"\bbeef\b|\bsteak\b|\bbrisket\b|\bveal\b",
    "pork": r"\bpork\b|\bham\b|\bbacon\b|\bprosciutto\b|\bpancetta\b|\bchorizo\b|\bsalami\b|\bpepperoni\b|\bsausage\b",
    "fish_seafood": (
        r"\bfish\b|\bsalmon\b|\btuna\b|\bcod\b|\bshrimp\b|\bprawns?\b|\bcrab\b|"
        r"\blobster\b|\btilapia\b|\banchovy|\bsardines?\b|\bmussels?\b|\bclams?\b|"
        r"\bscallops?\b|\bsquid\b|\boctopus\b"
    ),
    "other_meat": (
        r"\blamb\b|\bmutton\b|\bturkey\b|\bduck\b|\bvenison\b|\bgoat\b|"
        r"\blard\b|\bgelatin\b|\boyster|\bfish sauce\b"
    ),
}

# Dairy / eggs / honey: present → not vegan, but still vegetarian.
DAIRY_EGG_PATTERN = (
    r"\bmilk\b|\bcheese\b|\bbutter\b|\bcream\b|\byogurt\b|\byoghurt\b|"
    r"\bghee\b|\beggs?\b|\bmayonnaise\b|\bhoney\b|\bcondensed milk\b"
)


def parse_list(value):
    """Parse a stringified Python list; return [] on failure."""
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def derive_cuisine(tags):
    tagset = set(tags)
    for cuisine in CUISINES:
        if cuisine in tagset:
            return cuisine
    return None


def derive_dish_type(tags):
    tagset = set(tags)
    for tag, label in DISH_TYPE_TAGS:
        if tag in tagset:
            return label
    return "other"


def derive_diet(tags, proteins, ingredients_text):
    """Tags win; otherwise infer from ingredient content.

    Many plant-based recipes are not tagged vegetarian/vegan, so:
    no meat protein keywords -> vegetarian (if dairy/egg) or vegan (if neither).
    """
    tagset = set(tags)
    if "vegan" in tagset:
        return "vegan"
    if "vegetarian" in tagset:
        return "vegetarian"
    if not proteins:  # no meat detected in ingredients
        if re.search(DAIRY_EGG_PATTERN, ingredients_text):
            return "vegetarian"
        return "vegan"
    return "non-veg"


def derive_proteins(ingredients_text):
    found = [
        protein
        for protein, pattern in PROTEIN_PATTERNS.items()
        if re.search(pattern, ingredients_text)
    ]
    return found


def derive_skill(tags, n_steps, minutes):
    tagset = set(tags)
    beginner_tags = {"beginner-cook", "easy", "3-steps-or-less", "5-ingredients-or-less"}
    if tagset & beginner_tags or (n_steps <= 5 and minutes <= 30):
        return "beginner"
    if n_steps >= 12 or minutes >= 240:
        return "master_chef"
    return "intermediate"


def main():
    print(f"Loading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"  {len(df):,} raw recipes")

    # --- Parse list columns -------------------------------------------------
    df["tags"] = df["tags"].apply(parse_list)
    df["steps"] = df["steps"].apply(parse_list)
    df["ingredients"] = df["ingredients"].apply(parse_list)

    # --- Clean ---------------------------------------------------------------
    df = df.dropna(subset=["name"])
    df = df[df["steps"].str.len() > 0]
    df = df[df["ingredients"].str.len() > 0]
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).clip(0, 24 * 60)
    df = df.drop_duplicates(subset=["name"])
    print(f"  {len(df):,} after cleaning")

    # --- Derived fields -------------------------------------------------------
    df["cuisine"] = df["tags"].apply(derive_cuisine)
    df["dish_type"] = df["tags"].apply(derive_dish_type)

    ingredients_text = df["ingredients"].apply(lambda xs: " ".join(xs).lower())
    df["proteins"] = ingredients_text.apply(derive_proteins)
    df["diet"] = [
        derive_diet(tags, proteins, text)
        for tags, proteins, text in zip(df["tags"], df["proteins"], ingredients_text)
    ]

    df["n_steps"] = df["steps"].str.len()
    df["skill_level"] = df.apply(
        lambda r: derive_skill(r["tags"], r["n_steps"], r["minutes"]), axis=1
    )

    # --- Keep only recipes with a supported cuisine ---------------------------
    pool = df.dropna(subset=["cuisine"])
    print(f"  {len(pool):,} with a supported cuisine tag")

    # --- Stratified sample across cuisines -------------------------------------
    # sqrt-proportional allocation: compresses the dominance of huge cuisines
    # (american has 31k recipes) while keeping every cuisine well represented.
    counts = pool["cuisine"].value_counts()
    weights = counts**0.5
    alloc = (weights / weights.sum() * SAMPLE_SIZE).round().astype(int)
    alloc = alloc.clip(lower=50)
    diff = SAMPLE_SIZE - alloc.sum()
    alloc[alloc.idxmax()] += diff

    parts = []
    for cuisine, n in alloc.items():
        subset = pool[pool["cuisine"] == cuisine]
        take = min(n, len(subset))
        parts.append(subset.sample(n=take, random_state=SEED))
    sample = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"  sampled {len(sample):,} recipes")

    # --- Output ------------------------------------------------------------------
    sample.insert(0, "id", [f"recipe_{i:05d}" for i in range(len(sample))])
    out = sample[
        [
            "id", "name", "cuisine", "dish_type", "diet", "proteins",
            "skill_level", "minutes", "calories", "n_steps",
            "ingredients", "steps", "tags",
        ]
    ].copy()
    for col in ["proteins", "ingredients", "steps", "tags"]:
        out[col] = out[col].apply(json.dumps)

    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")

    # --- Summary -----------------------------------------------------------------
    print("\n== cuisine ==");     print(out["cuisine"].value_counts().to_string())
    print("\n== dish_type ==");   print(out["dish_type"].value_counts().to_string())
    print("\n== skill_level =="); print(out["skill_level"].value_counts().to_string())
    print("\n== diet ==");        print(out["diet"].value_counts().to_string())
    proteins_exploded = out["proteins"].apply(json.loads).explode()
    print("\n== proteins (multi-label) ==")
    print(proteins_exploded.value_counts().to_string())


if __name__ == "__main__":
    main()
