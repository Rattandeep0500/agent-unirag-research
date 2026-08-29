import ssl
ssl.SSLContext.load_default_certs = lambda self, purpose=ssl.Purpose.SERVER_AUTH: None

import sys
import time
sys.path.insert(0, r".\reproduction\experiments")

from cpu_dense_reranker import CPUDenseReranker

query = "Which county shares a border with the county where the most populous city in the state where WEKL operates is located?"

docs = [
    f"Candidate passage {i}: This is a representative document about geography, cities, counties, states, borders, and related entities."
    for i in range(100)
]

for batch_size in [8, 16, 32]:
    r = CPUDenseReranker()

    start = time.perf_counter()

    query_embedding = r.model.encode(
        query,
        convert_to_tensor=True,
        device="cpu",
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

    print(f"batch_size={batch_size} | time={elapsed:.3f}s | best={scores.max().item():.4f}")
