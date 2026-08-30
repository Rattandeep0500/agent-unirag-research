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
# PROJECT IMPORTS
# ============================================================

sys.path.insert(
    0,
    r".\reproduction\experiments",
)

from cpu_dense_reranker import CPUDenseReranker


# ============================================================
# CONFIGURATION
# ============================================================

BENCHMARK = Path(
    r".\cyber_data\processed\cyber_benchmark.json"
)

STIX_FILE = Path(
    r".\cyber_data\raw\enterprise-attack-19.1.json"
)

OUTPUT = Path(
    r".\reproduction\results\cyber_retrieval_results.csv"
)

URL = "http://127.0.0.1:8000/retrieve/"
CORPUS = "mitre_attack"

BASELINE_K = 10
E5_CANDIDATES = 100
FINAL_K = 10
ORACLE_HOP_K = 25


# ============================================================
# NORMALIZATION
# ============================================================

def norm(text):
    return " ".join(
        str(text).lower().split()
    )


# ============================================================
# LOAD DATA
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

        data = json.load(file)

    if not isinstance(data, list):
        raise RuntimeError(
            "Benchmark must contain a JSON list."
        )

    return data


def load_stix():

    if not STIX_FILE.exists():
        raise FileNotFoundError(
            f"STIX file not found: {STIX_FILE}"
        )

    with STIX_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        bundle = json.load(file)

    objects = bundle.get(
        "objects",
        [],
    )

    objects_by_id = {}

    for obj in objects:

        object_id = obj.get(
            "id"
        )

        if object_id:
            objects_by_id[
                object_id
            ] = obj

    return (
        objects,
        objects_by_id,
    )


# ============================================================
# GOLD MITIGATIONS
# ============================================================

def build_gold_mitigations(
    objects,
    benchmark_record,
):
    """
    Return every ATT&CK mitigation associated with the
    benchmark's gold technique.

    This avoids incorrectly treating only the selected
    mitigation in the benchmark as valid.
    """

    chain = benchmark_record.get(
        "chain",
        [],
    )

    if len(chain) != 3:
        raise RuntimeError(
            f"Invalid chain for "
            f"{benchmark_record.get('question_id')}"
        )

    technique_id = chain[1].get(
        "id",
        "",
    )

    if not technique_id:
        raise RuntimeError(
            f"Missing technique ID for "
            f"{benchmark_record.get('question_id')}"
        )

    mitigation_ids = set()

    for obj in objects:

        if obj.get("type") != "relationship":
            continue

        if obj.get(
            "relationship_type"
        ) != "mitigates":
            continue

        source_id = obj.get(
            "source_ref"
        )

        target_id = obj.get(
            "target_ref"
        )

        if (
            target_id == technique_id
            and source_id
            and source_id.startswith(
                "course-of-action--"
            )
        ):

            mitigation_ids.add(
                source_id
            )

    # Always retain the benchmark-selected mitigation.
    selected_id = chain[2].get(
        "id",
        "",
    )

    if selected_id:
        mitigation_ids.add(
            selected_id
        )

    return mitigation_ids


def build_gold_names(
    objects_by_id,
    mitigation_ids,
):

    names = set()

    for mitigation_id in mitigation_ids:

        obj = objects_by_id.get(
            mitigation_id
        )

        if not obj:
            continue

        name = obj.get(
            "name"
        )

        if name:
            names.add(
                norm(name)
            )

    return names


# ============================================================
# API RETRIEVAL
# ============================================================

def retrieve(
    query,
    top_k,
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

    normalized = []

    for item in results:

        if isinstance(
            item,
            str,
        ):

            normalized.append(
                {
                    "paragraph_text": item
                }
            )

        elif isinstance(
            item,
            dict,
        ):

            normalized.append(
                item
            )

    return normalized


# ============================================================
# DOCUMENT TEXT
# ============================================================

def document_text(item):

    parts = []

    for field in [
        "title",
        "paragraph_text",
        "text",
        "name",
    ]:

        value = item.get(
            field,
            "",
        )

        if value:
            parts.append(
                str(value)
            )

    # Remove duplicate pieces while preserving order.
    output = []
    seen = set()

    for part in parts:

        key = norm(part)

        if key in seen:
            continue

        seen.add(key)
        output.append(part)

    return "\n".join(output)


# ============================================================
# GOLD MATCHING
# ============================================================

def document_matches_gold(
    item,
    gold_names,
    technique_name,
):

    text = norm(
        document_text(item)
    )

    if not text:
        return False

    # --------------------------------------------------------
    # Direct mitigation entity:
    #
    # e.g.
    # "Audit"
    # --------------------------------------------------------

    for mitigation_name in gold_names:

        if (
            norm(mitigation_name)
            == norm(text)
        ):

            return True

    # --------------------------------------------------------
    # Relationship document:
    #
    # e.g.
    # "Audit mitigates Scheduled Task."
    #
    # The API may return only paragraph_text, so inspect the
    # text instead of relying on Elasticsearch metadata.
    # --------------------------------------------------------

    normalized_technique = norm(
        technique_name
    )

    if (
        normalized_technique
        and normalized_technique not in text
    ):
        return False

    for mitigation_name in gold_names:

        normalized_mitigation = norm(
            mitigation_name
        )

        if (
            normalized_mitigation
            and normalized_mitigation in text
        ):

            return True

    return False


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    ranked_docs,
    gold_names,
    technique_name,
):

    hit_ranks = []

    for rank, item in enumerate(
        ranked_docs,
        start=1,
    ):

        if document_matches_gold(
            item,
            gold_names,
            technique_name,
        ):

            hit_ranks.append(
                rank
            )

    first_hit = (
        min(hit_ranks)
        if hit_ranks
        else None
    )

    return {
        "recall@1": int(
            any(
                rank <= 1
                for rank in hit_ranks
            )
        ),

        "recall@5": int(
            any(
                rank <= 5
                for rank in hit_ranks
            )
        ),

        "recall@10": int(
            any(
                rank <= 10
                for rank in hit_ranks
            )
        ),

        "mrr": (
            1.0 / first_hit
            if first_hit
            else 0.0
        ),

        "first_gold_rank": (
            first_hit
            if first_hit
            else ""
        ),

        "gold_in_candidates": int(
            len(hit_ranks) > 0
        ),
    }


# ============================================================
# UNIQUE DOCUMENTS
# ============================================================

def unique_candidates(
    documents,
):

    output = []
    seen = set()

    for item in documents:

        text = norm(
            document_text(item)
        )

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        output.append(item)

    return output


# ============================================================
# ORACLE MULTI-HOP + E5
# ============================================================

def oracle_multihop_e5(
    record,
    reranker,
):

    chain = record.get(
        "chain",
        [],
    )

    source_name = chain[0].get(
        "name",
        "",
    )

    technique_name = chain[1].get(
        "name",
        "",
    )

    # Controlled oracle retrieval.
    queries = [
        source_name,
        f"{source_name} {technique_name}",
        f"{technique_name} mitigation",
    ]

    all_candidates = []

    start = time.perf_counter()

    for query in queries:

        results = retrieve(
            query,
            ORACLE_HOP_K,
        )

        all_candidates.extend(
            results
        )

    candidates = unique_candidates(
        all_candidates
    )

    if not candidates:

        return (
            [],
            time.perf_counter() - start,
        )

    docs = [
        document_text(item)
        for item in candidates
    ]

    reranked = reranker.rerank(
        record["question"],
        docs,
        top_k=min(
            FINAL_K,
            len(docs),
        ),
    )

    final_docs = [
        candidates[
            int(result["doc_id"])
        ]
        for result in reranked
    ]

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        final_docs,
        elapsed,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "========== CYBER RETRIEVAL EVALUATION =========="
    )

    benchmark = load_benchmark()

    objects, objects_by_id = load_stix()

    print(
        "Benchmark questions:",
        len(benchmark),
    )

    print(
        "STIX objects:",
        len(objects),
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

        question_id = record[
            "question_id"
        ]

        question = record[
            "question"
        ]

        chain = record[
            "chain"
        ]

        source_name = chain[0][
            "name"
        ]

        technique_name = chain[1][
            "name"
        ]

        gold_mitigation_ids = (
            build_gold_mitigations(
                objects,
                record,
            )
        )

        gold_names = build_gold_names(
            objects_by_id,
            gold_mitigation_ids,
        )

        print()
        print(
            f"[{index}/{len(benchmark)}] "
            f"{question_id}"
        )

        print(
            "Gold technique:",
            technique_name,
        )

        print(
            "Gold mitigations:",
            ", ".join(
                sorted(gold_names)
            ),
        )

        # ====================================================
        # METHOD 1: BM25
        # ====================================================

        start = time.perf_counter()

        bm25_docs = retrieve(
            question,
            BASELINE_K,
        )

        bm25_latency = (
            time.perf_counter()
            - start
        )

        bm25_metrics = calculate_metrics(
            bm25_docs,
            gold_names,
            technique_name,
        )

        # ====================================================
        # METHOD 2: BM25 + E5
        # ====================================================

        start = time.perf_counter()

        e5_candidates = retrieve(
            question,
            E5_CANDIDATES,
        )

        if e5_candidates:

            texts = [
                document_text(item)
                for item in e5_candidates
            ]

            reranked = reranker.rerank(
                question,
                texts,
                top_k=min(
                    FINAL_K,
                    len(e5_candidates),
                ),
            )

            e5_docs = [
                e5_candidates[
                    int(result["doc_id"])
                ]
                for result in reranked
            ]

        else:

            e5_docs = []

        e5_latency = (
            time.perf_counter()
            - start
        )

        e5_metrics = calculate_metrics(
            e5_docs,
            gold_names,
            technique_name,
        )

        # ====================================================
        # METHOD 3: ORACLE MULTI-HOP + E5
        # ====================================================

        oracle_docs, oracle_latency = (
            oracle_multihop_e5(
                record,
                reranker,
            )
        )

        oracle_metrics = calculate_metrics(
            oracle_docs,
            gold_names,
            technique_name,
        )

        print(
            "  BM25        "
            f"R@10={bm25_metrics['recall@10']} "
            f"gold={bm25_metrics['gold_in_candidates']} "
            f"rank={bm25_metrics['first_gold_rank']}"
        )

        print(
            "  BM25 + E5   "
            f"R@10={e5_metrics['recall@10']} "
            f"gold={e5_metrics['gold_in_candidates']} "
            f"rank={e5_metrics['first_gold_rank']}"
        )

        print(
            "  Oracle + E5 "
            f"R@10={oracle_metrics['recall@10']} "
            f"gold={oracle_metrics['gold_in_candidates']} "
            f"rank={oracle_metrics['first_gold_rank']}"
        )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        rows.extend(
            [
                {
                    "question_id": question_id,
                    "method": "BM25",
                    "recall@1": bm25_metrics[
                        "recall@1"
                    ],
                    "recall@5": bm25_metrics[
                        "recall@5"
                    ],
                    "recall@10": bm25_metrics[
                        "recall@10"
                    ],
                    "mrr": bm25_metrics[
                        "mrr"
                    ],
                    "gold_in_candidates": bm25_metrics[
                        "gold_in_candidates"
                    ],
                    "first_gold_rank": bm25_metrics[
                        "first_gold_rank"
                    ],
                    "latency_seconds": bm25_latency,
                },

                {
                    "question_id": question_id,
                    "method": "BM25+E5",
                    "recall@1": e5_metrics[
                        "recall@1"
                    ],
                    "recall@5": e5_metrics[
                        "recall@5"
                    ],
                    "recall@10": e5_metrics[
                        "recall@10"
                    ],
                    "mrr": e5_metrics[
                        "mrr"
                    ],
                    "gold_in_candidates": e5_metrics[
                        "gold_in_candidates"
                    ],
                    "first_gold_rank": e5_metrics[
                        "first_gold_rank"
                    ],
                    "latency_seconds": e5_latency,
                },

                {
                    "question_id": question_id,
                    "method": "Oracle-MultiHop+E5",
                    "recall@1": oracle_metrics[
                        "recall@1"
                    ],
                    "recall@5": oracle_metrics[
                        "recall@5"
                    ],
                    "recall@10": oracle_metrics[
                        "recall@10"
                    ],
                    "mrr": oracle_metrics[
                        "mrr"
                    ],
                    "gold_in_candidates": oracle_metrics[
                        "gold_in_candidates"
                    ],
                    "first_gold_rank": oracle_metrics[
                        "first_gold_rank"
                    ],
                    "latency_seconds": oracle_latency,
                },
            ]
        )

    # ========================================================
    # WRITE CSV
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "question_id",
        "method",
        "recall@1",
        "recall@5",
        "recall@10",
        "mrr",
        "gold_in_candidates",
        "first_gold_rank",
        "latency_seconds",
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
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "========== FINAL CYBER RESULTS =========="
    )

    for method in [
        "BM25",
        "BM25+E5",
        "Oracle-MultiHop+E5",
    ]:

        method_rows = [
            row
            for row in rows
            if row["method"] == method
        ]

        n = len(
            method_rows
        )

        def average(field):

            return (
                sum(
                    float(row[field])
                    for row in method_rows
                )
                / n
            )

        print()
        print(
            method
        )

        print(
            "Evaluated:",
            n,
        )

        print(
            f"Recall@1 : "
            f"{average('recall@1'):.4f}"
        )

        print(
            f"Recall@5 : "
            f"{average('recall@5'):.4f}"
        )

        print(
            f"Recall@10: "
            f"{average('recall@10'):.4f}"
        )

        print(
            f"MRR      : "
            f"{average('mrr'):.4f}"
        )

        print(
            "Gold candidate recall: "
            f"{average('gold_in_candidates'):.4f}"
        )

        print(
            f"Latency  : "
            f"{average('latency_seconds'):.4f}s"
        )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()