import json
import time
import csv
import requests

DATASET = r".\original\Agent-UniRAG\inference\processed_data\musique\dev_500_subsampled.jsonl"
URL = "http://127.0.0.1:8000/retrieve/"
CORPUS = "musique_dev"
TOP_K = 10


def norm(text):
    return " ".join(str(text).lower().split())


def gold_passages(record):
    gold = set()

    for ctx in record.get("contexts", []):
        if ctx.get("is_supporting", False):
            key = (
                norm(ctx.get("title", "")),
                norm(ctx.get("paragraph_text", ""))
            )
            gold.add(key)

    return gold


def retrieved_passage(item):
    return (
        norm(item.get("title", "")),
        norm(item.get("paragraph_text", ""))
    )


records = []

with open(DATASET, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("Loaded questions:", len(records))
print("Evaluating top-k:", TOP_K)

results = []

for i, record in enumerate(records, 1):

    question = record.get("question_text", "")
    gold = gold_passages(record)

    if not gold:
        print(f"[{i}/{len(records)}] SKIP - no supporting passages")
        continue

    body = {
        "retrieval_method": "retrieve_from_elasticsearch",
        "query_text": question,
        "max_hits_count": TOP_K,
        "document_type": "paragraph_text",
        "corpus_name": CORPUS,
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            URL,
            json=body,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[{i}/{len(records)}] ERROR: {e}")
        continue

    latency = time.perf_counter() - start

    retrieved = data.get("retrieval", [])

    # Convert possible dict/list response formats.
    if isinstance(retrieved, dict):
        retrieved = [retrieved]

    ranked = []
    for item in retrieved:
        if isinstance(item, dict):
            ranked.append(item)

    hits = [retrieved_passage(x) for x in ranked]

    hit_ranks = [
        rank
        for rank, passage in enumerate(hits, start=1)
        if passage in gold
    ]

    recall1 = int(any(rank <= 1 for rank in hit_ranks))
    recall5 = int(any(rank <= 5 for rank in hit_ranks))
    recall10 = int(any(rank <= 10 for rank in hit_ranks))
    reciprocal_rank = 1.0 / min(hit_ranks) if hit_ranks else 0.0

    results.append({
        "question_id": record.get("question_id", ""),
        "question": question,
        "recall@1": recall1,
        "recall@5": recall5,
        "recall@10": recall10,
        "mrr": reciprocal_rank,
        "latency_seconds": latency,
        "first_hit_rank": min(hit_ranks) if hit_ranks else "",
    })

    if i % 25 == 0 or i == 1:
        print(
            f"[{i}/{len(records)}] "
            f"R@1={recall1} R@5={recall5} R@10={recall10} "
            f"RR={reciprocal_rank:.3f} "
            f"latency={latency:.3f}s"
        )


if not results:
    raise RuntimeError("No successful evaluations were produced.")


def mean(key):
    return sum(float(x[key]) for x in results) / len(results)


print("\n========== BASELINE RESULTS ==========")
print("Evaluated:", len(results))
print(f"Recall@1 : {mean('recall@1'):.4f}")
print(f"Recall@5 : {mean('recall@5'):.4f}")
print(f"Recall@10: {mean('recall@10'):.4f}")
print(f"MRR      : {mean('mrr'):.4f}")
print(f"Latency  : {mean('latency_seconds'):.4f}s")

with open(
    r".\reproduction\musique_baseline_results.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nSaved:")
print(r".\reproduction\musique_baseline_results.csv")
