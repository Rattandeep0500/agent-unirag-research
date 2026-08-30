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

OUTPUT = Path(
    r".\reproduction\results\musique_decomposition_e5_results.csv"
)

URL = "http://127.0.0.1:8000/retrieve/"
CORPUS = "musique_dev"

PER_SUBQUERY_K = 25
TOP_K = 10

FIELDS = [
    "question_id",
    "question",
    "subquery_count",
    "candidate_count",
    "gold_in_candidates",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "first_gold_rank",
    "latency_seconds",
]


def norm(text):
    return " ".join(str(text).lower().split())


def passage_key(item):
    return (
        norm(item.get("title", "")),
        norm(item.get("paragraph_text", "")),
    )


def load_records():
    records = []

    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_completed_ids():
    completed = set()

    if not OUTPUT.exists():
        return completed

    with OUTPUT.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = row.get("question_id", "")
            if qid:
                completed.add(qid)

    return completed


def ensure_output():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        return

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()


def append_result(row):
    with OUTPUT.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)


def extract_subqueries(record):
    steps = record.get("reasoning_steps", [])

    subqueries = []

    for step in steps:
        if ">>>>" not in step:
            continue

        query = step.split(">>>>", 1)[0].strip()

        if query:
            subqueries.append(query)

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


def summarize():
    if not OUTPUT.exists():
        return

    with OUTPUT.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    def mean(key):
        return sum(float(row[key]) for row in rows) / len(rows)

    candidate_recall = mean("gold_in_candidates")

    print()
    print("========== RUNNING DECOMPOSITION + E5 RESULTS ==========")
    print("Evaluated:", len(rows))
    print(f"Gold candidate recall: {candidate_recall:.4f}")
    print(f"Recall@1            : {mean('recall@1'):.4f}")
    print(f"Recall@5            : {mean('recall@5'):.4f}")
    print(f"Recall@10           : {mean('recall@10'):.4f}")
    print(f"MRR                 : {mean('mrr'):.4f}")
    print(f"Latency             : {mean('latency_seconds'):.4f}s")


records = load_records()
completed = load_completed_ids()
ensure_output()

print("Total questions:", len(records))
print("Already completed:", len(completed))
print("Remaining:", len(records) - len(completed))

if len(completed) == len(records):
    print("Nothing to evaluate.")
    summarize()
    raise SystemExit(0)

print("Loading E5 reranker...")
reranker = CPUDenseReranker()
print("E5 reranker ready.")

for i, record in enumerate(records, 1):

    qid = record.get("question_id", "")

    if qid in completed:
        continue

    question = record.get("question_text", "")

    gold = {
        passage_key(ctx)
        for ctx in record.get("contexts", [])
        if ctx.get("is_supporting", False)
    }

    subqueries = extract_subqueries(record)

    if not subqueries:
        print(
            f"[{i}/{len(records)}] SKIP {qid}: "
            "no reasoning_steps"
        )
        continue

    start = time.perf_counter()

    try:
        candidate_map = {}

        for subquery in subqueries:
            retrieved = retrieve(subquery)

            for item in retrieved:
                candidate_map[passage_key(item)] = item

        candidates = list(candidate_map.values())

        gold_in_candidates = int(
            any(
                passage_key(item) in gold
                for item in candidates
            )
        )

        if not candidates:
            latency = time.perf_counter() - start

            append_result({
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
                "latency_seconds": latency,
            })

            completed.add(qid)
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
        latency = time.perf_counter() - start

        row = {
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
            "latency_seconds": latency,
        }

        append_result(row)
        completed.add(qid)

        print(
            f"[{i}/{len(records)}] "
            f"{qid} | "
            f"subqueries={len(subqueries)} | "
            f"candidates={len(candidates)} | "
            f"gold={gold_in_candidates} | "
            f"first_gold="
            f"{first_hit if first_hit else 'NOT IN TOP10'} | "
            f"latency={latency:.2f}s"
        )

        if len(completed) % 10 == 0:
            summarize()

    except Exception as exc:
        print(
            f"[{i}/{len(records)}] ERROR {qid}: {exc}"
        )
        print("Checkpoint preserved; continuing.")
        continue


print()
print("========== DECOMPOSITION + E5 FULL EVALUATION COMPLETE ==========")
summarize()
print("Saved:", OUTPUT)
