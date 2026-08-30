import csv
import json
import ssl
import sys
import time
from pathlib import Path

import requests


# Local workaround for the Windows certificate-store issue.
ssl.SSLContext.load_default_certs = (
    lambda self, purpose=ssl.Purpose.SERVER_AUTH: None
)

sys.path.insert(0, r".\reproduction\experiments")

from cpu_dense_reranker import CPUDenseReranker


DATASET = Path(
    r".\original\Agent-UniRAG\inference\processed_data\musique\dev_500_subsampled.jsonl"
)

FAILURES = Path(
    r".\reproduction\results\four_hop_candidate_100_diagnosis.csv"
)

OUTPUT = Path(
    r".\reproduction\results\four_hop_decomposition_e5.csv"
)

URL = "http://127.0.0.1:8000/retrieve/"
CORPUS = "musique_dev"

PER_SUBQUERY_K = 25
TOP_K = 10


def norm(text):
    return " ".join(str(text).lower().split())


def passage_key(item):
    return (
        norm(item.get("title", "")),
        norm(item.get("paragraph_text", "")),
    )


def load_failure_ids():
    ids = set()

    with FAILURES.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.add(row["question_id"])

    return ids


def load_records(failure_ids):
    records = {}

    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            qid = record.get("question_id")

            if qid in failure_ids:
                records[qid] = record

    return records


def extract_subqueries(record):
    steps = record.get("reasoning_steps", [])

    subqueries = []

    for step in steps:
        if ">>>>" not in step:
            continue

        question = step.split(">>>>", 1)[0].strip()

        if question:
            subqueries.append(question)

    return subqueries


def retrieve(query):
    body = {
        "retrieval_method": "retrieve_from_elasticsearch",
        "query_text": query,
        "max_hits_count": PER_SUBQUERY_K,
        "document_type": "paragraph_text",
        "corpus_name": CORPUS,
    }

    response = requests.post(
        URL,
        json=body,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    retrieved = data.get("retrieval", [])

    if isinstance(retrieved, dict):
        retrieved = [retrieved]

    return [
        item
        for item in retrieved
        if isinstance(item, dict)
    ]


def main():

    failure_ids = load_failure_ids()
    records = load_records(failure_ids)

    print("4-hop failure cases:", len(records))
    print("Loading E5 reranker...")

    reranker = CPUDenseReranker()

    print("E5 reranker ready.")

    results = []

    for index, qid in enumerate(sorted(records), 1):

        record = records[qid]
        question = record.get("question_text", "")

        gold = {
            passage_key(ctx)
            for ctx in record.get("contexts", [])
            if ctx.get("is_supporting", False)
        }

        subqueries = extract_subqueries(record)

        print()
        print(
            f"[{index}/{len(records)}] "
            f"{qid}"
        )
        print("Subqueries:")

        for sq in subqueries:
            print("  ", sq)

        start = time.perf_counter()

        candidate_map = {}

        for subquery in subqueries:

            retrieved = retrieve(subquery)

            for item in retrieved:
                candidate_map[passage_key(item)] = item

        candidates = list(candidate_map.values())

        if not candidates:
            print("  No candidates retrieved.")

            results.append({
                "question_id": qid,
                "question": question,
                "subquery_count": len(subqueries),
                "candidate_count": 0,
                "gold_in_candidates": 0,
                "recall@1": 0,
                "recall@5": 0,
                "recall@10": 0,
                "mrr": 0.0,
                "first_gold_rank": "",
                "latency_seconds": time.perf_counter() - start,
            })

            continue

        docs = [
            item.get("title", "") + "\n" +
            item.get("paragraph_text", "")
            for item in candidates
        ]

        reranked = reranker.rerank(
            question,
            docs,
            top_k=min(TOP_K, len(docs)),
        )

        final_docs = [
            candidates[item["doc_id"]]
            for item in reranked
        ]

        hit_ranks = [
            rank
            for rank, item in enumerate(final_docs, start=1)
            if passage_key(item) in gold
        ]

        first_hit = min(hit_ranks) if hit_ranks else ""

        elapsed = time.perf_counter() - start

        gold_in_candidates = int(
            any(passage_key(item) in gold for item in candidates)
        )

        results.append({
            "question_id": qid,
            "question": question,
            "subquery_count": len(subqueries),
            "candidate_count": len(candidates),
            "gold_in_candidates": gold_in_candidates,
            "recall@1": int(any(r <= 1 for r in hit_ranks)),
            "recall@5": int(any(r <= 5 for r in hit_ranks)),
            "recall@10": int(any(r <= 10 for r in hit_ranks)),
            "mrr": 1.0 / first_hit if first_hit else 0.0,
            "first_gold_rank": first_hit,
            "latency_seconds": elapsed,
        })

        print(
            f"Candidates: {len(candidates)} | "
            f"gold_in_candidates={gold_in_candidates} | "
            f"first_gold={first_hit if first_hit else 'NOT IN TOP10'} | "
            f"latency={elapsed:.2f}s"
        )

    if not results:
        raise RuntimeError("No decomposition results produced.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    def mean(key):
        return sum(
            float(row[key])
            for row in results
        ) / len(results)

    candidate_recall = sum(
        row["gold_in_candidates"]
        for row in results
    ) / len(results)

    print()
    print("========== DECOMPOSITION + E5 RESULTS ==========")
    print("Evaluated:", len(results))
    print(f"Gold candidate recall: {candidate_recall:.4f}")
    print(f"Recall@1            : {mean('recall@1'):.4f}")
    print(f"Recall@5            : {mean('recall@5'):.4f}")
    print(f"Recall@10           : {mean('recall@10'):.4f}")
    print(f"MRR                 : {mean('mrr'):.4f}")
    print(f"Latency             : {mean('latency_seconds'):.4f}s")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()