import csv
import json
import time
import requests
import sys

sys.path.insert(0, r".\reproduction\experiments")

from cpu_dense_reranker import CPUDenseReranker

RESULTS = r".\reproduction\results\four_hop_candidate_100_diagnosis.csv"
DATASET = r".\original\Agent-UniRAG\inference\processed_data\musique\dev_500_subsampled.jsonl"

URL = "http://127.0.0.1:8000/retrieve/"
CORPUS = "musique_dev"

CANDIDATES = 100
TOP_K = 10

OUTPUT = r".\reproduction\results\four_hop_e5_reranking.csv"


def norm(text):
    return " ".join(str(text).lower().split())


def passage_key(item):
    return (
        norm(item.get("title", "")),
        norm(item.get("paragraph_text", ""))
    )


# Load the 19 baseline failures.
failed_ids = set()

with open(
    RESULTS,
    "r",
    encoding="utf-8"
) as f:
    for row in csv.DictReader(f):
        failed_ids.add(row["question_id"])


# Load their full MuSiQue records.
records = {}

with open(
    DATASET,
    "r",
    encoding="utf-8"
) as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        if record.get("question_id") in failed_ids:
            records[record["question_id"]] = record


print("Failure cases:", len(records))
print("Loading E5 reranker...")

reranker = CPUDenseReranker()

print("E5 reranker ready.")


results = []


for i, qid in enumerate(sorted(records), 1):

    record = records[qid]
    question = record["question_text"]

    gold = {
        passage_key(ctx)
        for ctx in record.get("contexts", [])
        if ctx.get("is_supporting", False)
    }

    body = {
        "retrieval_method": "retrieve_from_elasticsearch",
        "query_text": question,
        "max_hits_count": CANDIDATES,
        "document_type": "paragraph_text",
        "corpus_name": CORPUS,
    }

    start = time.perf_counter()

    response = requests.post(
        URL,
        json=body,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    candidates = data.get("retrieval", [])

    if isinstance(candidates, dict):
        candidates = [candidates]

    candidates = [
        c for c in candidates
        if isinstance(c, dict)
    ]

    # E5 expects raw text for each candidate.
    docs = [
        c.get("title", "") + "\n" +
        c.get("paragraph_text", "")
        for c in candidates
    ]

    reranked = reranker.rerank(
        question,
        docs,
        top_k=min(TOP_K, len(docs)),
    )

    final_docs = [
        candidates[x["doc_id"]]
        for x in reranked
    ]

    latency = time.perf_counter() - start

    hit_ranks = []

    for rank, item in enumerate(final_docs, 1):
        if passage_key(item) in gold:
            hit_ranks.append(rank)

    first_hit = min(hit_ranks) if hit_ranks else ""

    results.append({
        "question_id": qid,
        "question": question,
        "recall@1": int(any(r <= 1 for r in hit_ranks)),
        "recall@5": int(any(r <= 5 for r in hit_ranks)),
        "recall@10": int(any(r <= 10 for r in hit_ranks)),
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "first_gold_rank": first_hit,
        "latency_seconds": latency,
    })

    print(
        f"[{i}/{len(records)}] "
        f"{qid} | "
        f"first_gold={first_hit if first_hit else 'NOT IN TOP10'}"
    )


def mean(key):
    return sum(
        float(row[key])
        for row in results
    ) / len(results)


print()
print("========== E5 RERANKING RESULTS ==========")
print("Evaluated:", len(results))
print(f"Recall@1 : {mean('recall@1'):.4f}")
print(f"Recall@5 : {mean('recall@5'):.4f}")
print(f"Recall@10: {mean('recall@10'):.4f}")
print(f"MRR      : {mean('mrr'):.4f}")
print(f"Latency  : {mean('latency_seconds'):.4f}s")

with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print("Saved:", OUTPUT)
