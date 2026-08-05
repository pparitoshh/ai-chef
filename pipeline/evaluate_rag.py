"""RAG evaluation: 2 answer models × 2 prompt variants, judged by LLM-as-judge.

Samples questions from data/ground_truth.csv, runs the full RAG flow
(hybrid+rerank retrieval → Groq answer) for every model × variant combo, and
asks the judge model whether each answer is RELEVANT / PARTLY_RELEVANT /
NON_RELEVANT. Rows stream to data/rag_eval_rows.csv (resumable); the summary
lands in data/rag_eval_results.json.

Prereq: postgres running + ingestion done + ground truth generated.
Usage:  uv run python -m pipeline.evaluate_rag [--n 50]
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import RateLimitError

from api import rag

load_dotenv()

GT_PATH = Path("data/ground_truth.csv")
ROWS_PATH = Path("data/rag_eval_rows.csv")
OUT_PATH = Path("data/rag_eval_results.json")
MODELS = [rag.GROQ_MODEL, rag.GROQ_JUDGE_MODEL]  # 8b vs 70b
VARIANTS = list(rag.PROMPT_VARIANTS)
SEED = 42
# free tier ~30 RPM per model — pace all calls under that
MIN_INTERVAL_S = 2.2

ROW_FIELDS = ["question", "model", "variant", "answer", "relevance",
              "tokens", "latency_ms"]


def call_with_retry(fn, max_retries: int = 6):
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            wait = min(2**attempt * 5, 60)
            print(f"  rate limited, waiting {wait}s ...")
            time.sleep(wait)
    raise RuntimeError("too many rate-limit retries")


def done_combos() -> set[tuple[str, str, str]]:
    if not ROWS_PATH.exists():
        return set()
    with ROWS_PATH.open() as f:
        return {(r["question"], r["model"], r["variant"])
                for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of questions")
    args = parser.parse_args()

    pairs = pd.read_csv(GT_PATH).sample(n=min(args.n, sum(1 for _ in GT_PATH.open()) - 1),
                                        random_state=SEED)
    done = done_combos()
    if done:
        print(f"Resuming: {len(done)} rows already evaluated")

    new_file = not ROWS_PATH.exists()
    last = 0.0

    def pace():
        nonlocal last
        dt = time.time() - last
        if dt < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - dt)
        last = time.time()

    with ROWS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if new_file:
            writer.writeheader()
        total = len(pairs) * len(MODELS) * len(VARIANTS)
        i = 0
        for question in pairs["question"]:
            for model in MODELS:
                for variant in VARIANTS:
                    i += 1
                    if (question, model, variant) in done:
                        continue
                    pace()
                    out = call_with_retry(
                        lambda: rag.generate_answer(question, model=model,
                                                    variant=variant))
                    pace()
                    verdict = call_with_retry(
                        lambda: rag.judge_relevance(question, out["answer"]))
                    writer.writerow({
                        "question": question, "model": model, "variant": variant,
                        "answer": out["answer"], "relevance": verdict,
                        "tokens": out["tokens"], "latency_ms": out["latency_ms"],
                    })
                    f.flush()
                    print(f"[{i}/{total}] {model.split('-')[1]}·{variant}: {verdict}")

    # --- summary --------------------------------------------------------------
    rows = pd.read_csv(ROWS_PATH)
    summary = []
    for (model, variant), grp in rows.groupby(["model", "variant"]):
        counts = Counter(grp["relevance"])
        n = len(grp)
        summary.append({
            "model": model,
            "variant": variant,
            "n": n,
            "relevant": round(counts.get("RELEVANT", 0) / n, 4),
            "partly": round(counts.get("PARTLY_RELEVANT", 0) / n, 4),
            "non_relevant": round(counts.get("NON_RELEVANT", 0) / n, 4),
            "tokens_mean": round(grp["tokens"].mean(), 1),
            "latency_ms_mean": round(grp["latency_ms"].mean(), 1),
        })
    summary.sort(key=lambda s: s["relevant"], reverse=True)

    print("\n== Summary (sorted by RELEVANT share) ==")
    print(f"{'model':<26} {'variant':<10} {'rel':>6} {'part':>6} {'non':>6} "
          f"{'tok':>7} {'ms':>8}")
    for s in summary:
        print(f"{s['model']:<26} {s['variant']:<10} {s['relevant']:>6.3f} "
              f"{s['partly']:>6.3f} {s['non_relevant']:>6.3f} "
              f"{s['tokens_mean']:>7.1f} {s['latency_ms_mean']:>8.1f}")

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
