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
    records = [json.loads(line) for line in f if line.strip()]

question = records[0]["question_text"]

body = {
    "retrieval_method": "retrieve_from_elasticsearch",
    "query_text": question,
    "max_hits_count": 100,
    "document_type": "paragraph_text",
    "corpus_name": "musique_dev",
}

t0 = time.perf_counter()

response = requests.post(
    "http://127.0.0.1:8000/retrieve/",
    json=body,
    timeout=120,
)
response.raise_for_status()

candidates = response.json()["retrieval"]

retrieval_time = time.perf_counter() - t0

docs = [
    c.get("title", "") + "\n" + c.get("paragraph_text", "")
    for c in candidates
]

r = CPUDenseReranker()

t1 = time.perf_counter()

out = r.rerank(
    question,
    docs,
    top_k=10,
)

rerank_time = time.perf_counter() - t1

print("Candidates:", len(docs))
print(f"Retrieval time: {retrieval_time:.3f}s")
print(f"E5 rerank time: {rerank_time:.3f}s")
print(f"Total: {retrieval_time + rerank_time:.3f}s")
print("Top result:", out[0])
