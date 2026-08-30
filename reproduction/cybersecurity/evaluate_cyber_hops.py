import csv
import json
import ssl
import sys
import time
from pathlib import Path

import requests


# ============================================================
# WINDOWS SSL WORKAROUND
# ============================================================

ssl.SSLContext.load_default_certs = (
    lambda self, purpose=ssl.Purpose.SERVER_AUTH: None
)


# ============================================================
# IMPORT E5 RERANKER
# ============================================================

sys.path.insert(
    0,
    r".\reproduction\experiments",
)

from cpu_dense_reranker import CPUDenseReranker


# ============================================================
# PATHS / CONFIG
# ============================================================

BENCHMARK = Path(
    r".\cyber_data\processed\cyber_benchmark.json"
)

OUTPUT = Path(
    r".\reproduction\results\cyber_hop_results.csv"
)

URL = "http://127.0.0.1:8000/retrieve/"
CORPUS = "mitre_attack"

RETRIEVAL_K = 25
FINAL_K = 10


# ============================================================
# LOAD BENCHMARK
# ============================================================

def load_benchmark():

    if not BENCHMARK.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {BENCHMARK}"
        )

    with BENCHMARK.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    top_k=RETRIEVAL_K,
):

    payload = {
        "retrieval_method": (
            "retrieve_from_elasticsearch"
        ),
        "query_text": query,
        "max_hits_count": top_k,
        "document_type": "paragraph_text",
        "corpus_name": CORPUS,
    }

    response = requests.post(
        URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    results = data.get(
        "retrieval",
        [],
    )

    if isinstance(
        results,
        dict,
    ):
        results = [results]

    return [
        item
        for item in results
        if isinstance(item, dict)
    ]


# ============================================================
# DOCUMENT TEXT
# ============================================================

def get_text(item):

    parts = []

    for key in [
        "title",
        "paragraph_text",
        "text",
    ]:

        value = item.get(
            key,
            "",
        )

        if value:
            parts.append(
                str(value)
            )

    return "\n".join(parts)


# ============================================================
# RELATIONSHIP MATCHING
# ============================================================

def relationship_text_matches(
    item,
    source_name,
    target_name,
    relationship,
):

    text = (
        get_text(item)
        .lower()
    )

    source = source_name.lower()
    target = target_name.lower()
    relation = relationship.lower()

    return (
        source in text
        and target in text
        and relation in text
    )


def first_matching_rank(
    documents,
    source_name,
    target_name,
    relationship,
):

    for rank, item in enumerate(
        documents,
        start=1,
    ):

        if relationship_text_matches(
            item,
            source_name,
            target_name,
            relationship,
        ):

            return rank

    return ""


# ============================================================
# METRICS
# ============================================================

def hit_at(
    rank,
    k,
):

    if rank == "":
        return 0

    return int(
        int(rank) <= k
    )


def reciprocal_rank(
    rank,
):

    if rank == "":
        return 0.0

    return 1.0 / int(rank)


# ============================================================
# E5 RERANKING
# ============================================================

def rerank_with_e5(
    query,
    candidates,
    reranker,
):

    if not candidates:
        return []

    docs = [
        get_text(item)
        for item in candidates
    ]

    ranked = reranker.rerank(
        query,
        docs,
        top_k=min(
            FINAL_K,
            len(candidates),
        ),
    )

    return [
        candidates[
            int(result["doc_id"])
        ]
        for result in ranked
    ]


# ============================================================
# EVALUATE ONE HOP
# ============================================================

def evaluate_hop(
    query,
    source_name,
    target_name,
    relationship,
    reranker,
):

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    start = time.perf_counter()

    bm25_docs = retrieve(
        query,
        RETRIEVAL_K,
    )

    bm25_latency = (
        time.perf_counter()
        - start
    )

    bm25_rank = first_matching_rank(
        bm25_docs,
        source_name,
        target_name,
        relationship,
    )

    # --------------------------------------------------------
    # BM25 + E5
    # --------------------------------------------------------

    start = time.perf_counter()

    e5_docs = rerank_with_e5(
        query,
        bm25_docs,
        reranker,
    )

    e5_latency = (
        time.perf_counter()
        - start
    )

    e5_rank = first_matching_rank(
        e5_docs,
        source_name,
        target_name,
        relationship,
    )

    return {
        "bm25_rank": bm25_rank,
        "bm25_recall@1": hit_at(
            bm25_rank,
            1,
        ),
        "bm25_recall@5": hit_at(
            bm25_rank,
            5,
        ),
        "bm25_recall@10": hit_at(
            bm25_rank,
            10,
        ),
        "bm25_mrr": reciprocal_rank(
            bm25_rank
        ),
        "bm25_latency": bm25_latency,

        "e5_rank": e5_rank,
        "e5_recall@1": hit_at(
            e5_rank,
            1,
        ),
        "e5_recall@5": hit_at(
            e5_rank,
            5,
        ),
        "e5_recall@10": hit_at(
            e5_rank,
            10,
        ),
        "e5_mrr": reciprocal_rank(
            e5_rank
        ),
        "e5_latency": e5_latency,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    benchmark = load_benchmark()

    print(
        "========== CYBER RELATIONSHIP HOP EVALUATION =========="
    )

    print(
        "Questions:",
        len(benchmark),
    )

    print()
    print(
        "Loading E5 reranker..."
    )

    reranker = CPUDenseReranker()

    print(
        "E5 reranker ready."
    )

    rows = []

    for index, record in enumerate(
        benchmark,
        start=1,
    ):

        chain = record[
            "chain"
        ]

        source = chain[0][
            "name"
        ]

        technique = chain[1][
            "name"
        ]

        mitigation = chain[2][
            "name"
        ]

        print()
        print(
            f"[{index}/{len(benchmark)}] "
            f"{record['question_id']}"
        )

        # ====================================================
        # HOP 1:
        #
        # source -> technique
        # ====================================================

        hop1_query = (
            f"{source} {technique}"
        )

        hop1 = evaluate_hop(
            query=hop1_query,
            source_name=source,
            target_name=technique,
            relationship="uses",
            reranker=reranker,
        )

        print(
            "  HOP 1:",
            f"{source} -> {technique}"
        )

        print(
            "    BM25 rank:",
            hop1["bm25_rank"],
            "| E5 rank:",
            hop1["e5_rank"],
        )

        # ====================================================
        # HOP 2:
        #
        # technique -> mitigation
        # ====================================================

        hop2_query = (
            f"{technique} {mitigation}"
        )

        hop2 = evaluate_hop(
            query=hop2_query,
            source_name=mitigation,
            target_name=technique,
            relationship="mitigates",
            reranker=reranker,
        )

        print(
            "  HOP 2:",
            f"{mitigation} -> {technique}"
        )

        print(
            "    BM25 rank:",
            hop2["bm25_rank"],
            "| E5 rank:",
            hop2["e5_rank"],
        )

        # ====================================================
        # SAVE BOTH HOPS
        # ====================================================

        rows.append(
            {
                "question_id": record[
                    "question_id"
                ],
                "hop": 1,
                "source": source,
                "target": technique,
                "relationship": "uses",
                "query": hop1_query,

                "bm25_rank": hop1[
                    "bm25_rank"
                ],
                "bm25_recall@1": hop1[
                    "bm25_recall@1"
                ],
                "bm25_recall@5": hop1[
                    "bm25_recall@5"
                ],
                "bm25_recall@10": hop1[
                    "bm25_recall@10"
                ],
                "bm25_mrr": hop1[
                    "bm25_mrr"
                ],
                "bm25_latency": hop1[
                    "bm25_latency"
                ],

                "e5_rank": hop1[
                    "e5_rank"
                ],
                "e5_recall@1": hop1[
                    "e5_recall@1"
                ],
                "e5_recall@5": hop1[
                    "e5_recall@5"
                ],
                "e5_recall@10": hop1[
                    "e5_recall@10"
                ],
                "e5_mrr": hop1[
                    "e5_mrr"
                ],
                "e5_latency": hop1[
                    "e5_latency"
                ],
            }
        )

        rows.append(
            {
                "question_id": record[
                    "question_id"
                ],
                "hop": 2,
                "source": mitigation,
                "target": technique,
                "relationship": "mitigates",
                "query": hop2_query,

                "bm25_rank": hop2[
                    "bm25_rank"
                ],
                "bm25_recall@1": hop2[
                    "bm25_recall@1"
                ],
                "bm25_recall@5": hop2[
                    "bm25_recall@5"
                ],
                "bm25_recall@10": hop2[
                    "bm25_recall@10"
                ],
                "bm25_mrr": hop2[
                    "bm25_mrr"
                ],
                "bm25_latency": hop2[
                    "bm25_latency"
                ],

                "e5_rank": hop2[
                    "e5_rank"
                ],
                "e5_recall@1": hop2[
                    "e5_recall@1"
                ],
                "e5_recall@5": hop2[
                    "e5_recall@5"
                ],
                "e5_recall@10": hop2[
                    "e5_recall@10"
                ],
                "e5_mrr": hop2[
                    "e5_mrr"
                ],
                "e5_latency": hop2[
                    "e5_latency"
                ],
            }
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "question_id",
        "hop",
        "source",
        "target",
        "relationship",
        "query",
        "bm25_rank",
        "bm25_recall@1",
        "bm25_recall@5",
        "bm25_recall@10",
        "bm25_mrr",
        "bm25_latency",
        "e5_rank",
        "e5_recall@1",
        "e5_recall@5",
        "e5_recall@10",
        "e5_mrr",
        "e5_latency",
    ]

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "========== FINAL HOP RESULTS =========="
    )

    for hop in [1, 2]:

        hop_rows = [
            row
            for row in rows
            if row["hop"] == hop
        ]

        n = len(
            hop_rows
        )

        def avg(field):

            return (
                sum(
                    float(row[field])
                    for row in hop_rows
                )
                / n
            )

        print()
        print(
            f"HOP {hop}"
        )

        print(
            "Evaluated:",
            n,
        )

        print(
            "BM25"
        )

        print(
            f"  Recall@1 : "
            f"{avg('bm25_recall@1'):.4f}"
        )

        print(
            f"  Recall@5 : "
            f"{avg('bm25_recall@5'):.4f}"
        )

        print(
            f"  Recall@10: "
            f"{avg('bm25_recall@10'):.4f}"
        )

        print(
            f"  MRR      : "
            f"{avg('bm25_mrr'):.4f}"
        )

        print(
            f"  Latency  : "
            f"{avg('bm25_latency'):.4f}s"
        )

        print(
            "BM25 + E5"
        )

        print(
            f"  Recall@1 : "
            f"{avg('e5_recall@1'):.4f}"
        )

        print(
            f"  Recall@5 : "
            f"{avg('e5_recall@5'):.4f}"
        )

        print(
            f"  Recall@10: "
            f"{avg('e5_recall@10'):.4f}"
        )

        print(
            f"  MRR      : "
            f"{avg('e5_mrr'):.4f}"
        )

        print(
            f"  Latency  : "
            f"{avg('e5_latency'):.4f}s"
        )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()