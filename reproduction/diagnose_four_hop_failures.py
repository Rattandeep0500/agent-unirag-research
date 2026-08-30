import csv
import json
import time
import requests

RESULTS = r".\reproduction\musique_baseline_results.csv"
DATASET = r".\original\Agent-UniRAG\inference\processed_data\musique\dev_500_subsampled.jsonl"
URL = "http://127.0.0.1:8000/retrieve/"
CORPUS = "musique_dev"
TOP_K = 100
OUTPUT = r".\reproduction\results\four_hop_candidate_100_diagnosis.csv"


def norm(text):
    return " ".join(str(text).lower().split())


def passage_key(item):
    return (
        norm(item.get("title", "")),
        norm(item.get("paragraph_text", ""))
    )


# Load baseline results and identify failed 4-hop questions.
failed_ids = set()

with open(RESULTS, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        qid = row["question_id"]

        if qid.startswith("4hop") and row["recall@10"] == "0":
            failed_ids.add(qid)

print("Failed 4-hop questions:", len(failed_ids))


# Load dataset records.
records = {}

with open(DATASET, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        qid = record.get("question_id", "")

        if qid in failed_ids:
            records[qid] = record


print("Loaded failure records:", len(records))


diagnoses = []

for i, qid in enumerate(sorted(failed_ids), 1):

    record = records.get(qid)

    if record is None:
        print(f"[{i}/{len(failed_ids)}] MISSING DATA: {qid}")
        continue

    question = record.get("question_text", "")

    gold = []

    for ctx in record.get("contexts", []):
        if ctx.get("is_supporting", False):
            gold.append({
                "title": ctx.get("title", ""),
                "paragraph_text": ctx.get("paragraph_text", ""),
            })

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
        print(f"[{i}/{len(failed_ids)}] ERROR {qid}: {e}")
        continue

    latency = time.perf_counter() - start

    retrieved = data.get("retrieval", [])

    if isinstance(retrieved, dict):
        retrieved = [retrieved]

    retrieved = [
        item for item in retrieved
        if isinstance(item, dict)
    ]

    gold_keys = {
        passage_key(x)
        for x in gold
    }

    hit_ranks = []

    for rank, item in enumerate(retrieved, 1):
        if passage_key(item) in gold_keys:
            hit_ranks.append(rank)

    first_hit = min(hit_ranks) if hit_ranks else ""

    row = {
        "question_id": qid,
        "question": question,
        "latency_seconds": round(latency, 4),
        "gold_count": len(gold),
        "gold_titles": " || ".join(
            x["title"] for x in gold
        ),
        "first_gold_rank": first_hit,
    }

    for rank in range(1, TOP_K + 1):

        if rank <= len(retrieved):
            item = retrieved[rank - 1]

            row[f"rank{rank}_title"] = item.get("title", "")
            row[f"rank{rank}_score"] = item.get("score", "")
            row[f"rank{rank}_is_gold"] = (
                passage_key(item) in gold_keys
            )
        else:
            row[f"rank{rank}_title"] = ""
            row[f"rank{rank}_score"] = ""
            row[f"rank{rank}_is_gold"] = False

    diagnoses.append(row)

    print(
        f"[{i}/{len(failed_ids)}] "
        f"{qid} | "
        f"gold={len(gold)} | "
        f"first_gold={first_hit if first_hit else 'NOT IN TOP10'}"
    )


if not diagnoses:
    raise RuntimeError("No diagnostic results produced.")


with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=diagnoses[0].keys()
    )

    writer.writeheader()
    writer.writerows(diagnoses)


print()
print("========== FAILURE DIAGNOSIS COMPLETE ==========")
print("Diagnosed:", len(diagnoses))
print("Saved:", OUTPUT)
