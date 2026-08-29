import ssl
ssl.SSLContext.load_default_certs = lambda self, purpose=ssl.Purpose.SERVER_AUTH: None

import json
import sys
import time
import requests

sys.path.insert(0, r".\reproduction\experiments")
from cpu_dense_reranker import CPUDenseReranker

DATASET = r".\original\Agent-UniRAG\inference\processed_data\musique\dev_500_subsampled.jsonl"

with open(DATASET, "r", encoding="utf-8") as f:
    record = next(json.loads(line) for line in f if line.strip())

body = {
    "retrieval_method": "retrieve_from_elasticsearch",
    "query_text": record["question_text"],
    "max_hits_count": 100,
    "document_type": "paragraph_text",
    "corpus_name": "musique_dev",
}

response = requests.post(
    "http://127.0.0.1:8000/retrieve/",
    json=body,
    timeout=120,
)
response.raise_for_status()

candidates = response.json()["retrieval"]

docs = [
    c.get("title", "") + "\n" + c.get("paragraph_text", "")
    for c in candidates
]

r = CPUDenseReranker()

# Warm-up
r.model.encode(
    ["warmup"],
    convert_to_tensor=True,
    device="cpu",
    show_progress_bar=False,
    normalize_embeddings=True,
)

print("Real candidates:", len(docs))

for batch_size in [8, 16, 32]:
    start = time.perf_counter()

    query_embedding = r.model.encode(
        record["question_text"],
        convert_to_tensor=True,
        device="cpu",
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    doc_embeddings = r.model.encode(
        docs,
        convert_to_tensor=True,
        device="cpu",
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    scores = doc_embeddings @ query_embedding

    elapsed = time.perf_counter() - start

    print(
        f"batch_size={batch_size} | "
        f"time={elapsed:.3f}s | "
        f"top_score={scores.max().item():.6f}"
    )
