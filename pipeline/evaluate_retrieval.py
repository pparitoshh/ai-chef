"""Retrieval evaluation: Hit Rate and MRR across search approaches.

Compares text / vector / hybrid / hybrid+rerank on the Groq-generated
ground-truth questions (data/ground_truth.csv) and saves the metrics to
data/retrieval_eval_results.json.

Prereq: postgres running + ingestion done + ground truth generated.
Usage:  uv run python -m pipeline.evaluate_retrieval [--k 5] [--limit N]
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from api.search import APPROACHES

GT_PATH = Path("data/ground_truth.csv")
OUT_PATH = Path("data/retrieval_eval_results.json")


def evaluate(approach: str, pairs: pd.DataFrame, k: int) -> dict:
    search = APPROACHES[approach]
    hits, rr_sum, latencies = 0, 0.0, []
    for _, row in pairs.iterrows():
        t0 = time.time()
        results = search(row["question"], k=k)
        latencies.append(time.time() - t0)
        ids = [r.id for r in results]
        if row["recipe_id"] in ids:
            hits += 1
            rr_sum += 1.0 / (ids.index(row["recipe_id"]) + 1)
    n = len(pairs)
    return {
        "approach": approach,
        "n_queries": n,
        "k": k,
        "hit_rate": round(hits / n, 4),
        "mrr": round(rr_sum / n, 4),
        "latency_ms_mean": round(1000 * sum(latencies) / n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="debug: use only N pairs")
    parser.add_argument("--approaches", nargs="+", default=list(APPROACHES))
    parser.add_argument("--gt", type=Path, default=GT_PATH, help="ground-truth CSV")
    args = parser.parse_args()

    pairs = pd.read_csv(args.gt, dtype={"recipe_id": str})
    if args.limit:
        pairs = pairs.head(args.limit)
    print(f"Evaluating {len(pairs)} questions, k={args.k}\n")

    results = []
    for approach in args.approaches:
        print(f"--- {approach} ---")
        metrics = evaluate(approach, pairs, args.k)
        results.append(metrics)
        print(f"    hit_rate={metrics['hit_rate']}  mrr={metrics['mrr']}  "
              f"latency={metrics['latency_ms_mean']}ms")

    print("\n== Summary (sorted by MRR) ==")
    results.sort(key=lambda m: m["mrr"], reverse=True)
    print(f"{'approach':<16} {'hit_rate':>8} {'mrr':>8} {'ms/query':>9}")
    for m in results:
        print(f"{m['approach']:<16} {m['hit_rate']:>8.4f} {m['mrr']:>8.4f} "
              f"{m['latency_ms_mean']:>9.1f}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
