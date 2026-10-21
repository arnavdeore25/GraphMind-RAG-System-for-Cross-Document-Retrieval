"""Compare vector-only vs graph-only vs hybrid.

eval_set.json format:
[{"question": "...", "answer": "reference answer", "gold_sources": ["a.pdf","b.txt"], "hops": 1|2|3}, ...]

Usage: python evaluate.py eval_set.json
"""
import json
import sys
import time

import pandas as pd

from generation import answer
from llm import generate_json
from retrieval import Retriever

JUDGE = """You are grading a RAG answer.
Question: {q}
Reference answer: {ref}
Model answer: {ans}
Is the model answer factually consistent with the reference and complete enough? 
Return JSON: {{"correct": 1 or 0}}"""


def run(eval_path):
    data = json.load(open(eval_path, encoding="utf-8"))
    retriever = Retriever()
    rows = []
    for item in data:
        for mode in ("vector", "graph", "hybrid"):
            r = retriever.retrieve(item["question"], mode=mode)
            t0 = time.perf_counter()
            ans = answer(item["question"], r)
            total = r.latency + (time.perf_counter() - t0)
            got = {c["source"] for c in r.chunks}
            gold = set(item.get("gold_sources", []))
            rows.append({
                "mode": mode,
                "hops": item.get("hops", 1),
                "correct": generate_json(JUDGE.format(q=item["question"], ref=item["answer"], ans=ans)).get("correct", 0),
                "evidence_recall": len(gold & got) / len(gold) if gold else None,
                "full_coverage": int(gold <= got) if gold else None,
                "retrieval_s": r.latency,
                "total_s": total,
                "context_chars": sum(len(c["text"]) for c in r.chunks),  # cost proxy
            })
        print(f"done: {item['question'][:60]}")
    df = pd.DataFrame(rows)
    df.to_csv("eval_results.csv", index=False)
    print("\n=== Overall ===")
    print(df.groupby("mode").mean(numeric_only=True).round(3))
    print("\n=== By hop count (accuracy / evidence recall) ===")
    print(df.groupby(["hops", "mode"])[["correct", "evidence_recall"]].mean().round(3))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "eval_set.json")
