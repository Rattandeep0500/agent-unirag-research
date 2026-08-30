import json
import os
import re
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

DATASET = Path(
    r".\original\Agent-UniRAG\inference\processed_data\musique\dev_500_subsampled.jsonl"
)

HF_ROOT = Path(
    os.path.expandvars(r"%USERPROFILE%\.cache\huggingface\hub")
)

MODEL_ROOT = (
    HF_ROOT / "models--Qwen--Qwen2.5-0.5B-Instruct"
)

SNAPSHOT_ROOT = MODEL_ROOT / "snapshots"

RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve/"
CORPUS_NAME = "musique_dev"

TEST_QUESTIONS = 5
MAX_STEPS = 4

RETRIEVAL_K = 10
EVIDENCE_TO_KEEP = 5


# ============================================================
# LOAD MODEL
# ============================================================

snapshots = [
    path
    for path in SNAPSHOT_ROOT.iterdir()
    if path.is_dir()
]

if not snapshots:
    raise RuntimeError(
        f"No Qwen snapshot found in: {SNAPSHOT_ROOT}"
    )

MODEL_PATH = snapshots[0]

print("Using model:", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    torch_dtype=torch.float32,
)

model.to("cpu")
model.eval()

print("Qwen model loaded.")


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):
    payload = {
        "retrieval_method": "retrieve_from_elasticsearch",
        "query_text": query,
        "max_hits_count": RETRIEVAL_K,
        "document_type": "paragraph_text",
        "corpus_name": CORPUS_NAME,
    }

    response = requests.post(
        RETRIEVAL_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("retrieval", [])

    if isinstance(results, dict):
        results = [results]

    return [
        item
        for item in results
        if isinstance(item, dict)
    ]


# ============================================================
# HELPERS
# ============================================================

def clean_query(text):
    text = text.strip()

    text = re.sub(
        r"^(?:[-*]|\d+[.)])\s*",
        "",
        text,
    )

    text = text.replace('"', "").replace("'", "")

    if "?" in text:
        text = text.split("?", 1)[0].strip()

    text = " ".join(text.split())

    return text


def extract_json(text):
    """
    Extract the first JSON object from model output.
    """
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def valid_query(query):
    if not query:
        return False

    lowered = query.lower()

    banned = [
        "country a",
        "entity a",
        "place a",
        "company a",
        "person a",
        "unknown",
        "cannot determine",
        "i don't know",
    ]

    if any(token in lowered for token in banned):
        return False

    if len(query.split()) < 3:
        return False

    return True


def format_evidence(evidence):
    if not evidence:
        return "NONE"

    chunks = []

    for i, item in enumerate(
        evidence[-EVIDENCE_TO_KEEP:],
        start=1,
    ):
        title = item.get("title", "")
        text = item.get("paragraph_text", "")

        chunks.append(
            f"Evidence {i}\n"
            f"Title: {title}\n"
            f"Text: {text[:700]}"
        )

    return "\n\n".join(chunks)


# ============================================================
# PLANNER
# ============================================================

def generate_next_query(
    original_question,
    evidence,
    previous_queries,
    step_number,
):
    evidence_text = format_evidence(evidence)

    previous_text = (
        "\n".join(
            f"- {query}"
            for query in previous_queries
        )
        if previous_queries
        else "NONE"
    )

    prompt = f"""
You are a retrieval planner for a multi-hop question answering system.

Your task is ONLY to generate the NEXT searchable query.

Original question:
{original_question}

Evidence already retrieved:
{evidence_text}

Queries already used:
{previous_text}

Current step:
{step_number}

Follow these rules exactly:

1. Return ONE query only.
2. The query must retrieve the NEXT missing fact needed to answer
   the original question.
3. Use facts or entities already established in the evidence.
4. At step 1, use only information explicitly present in the original question.
5. At later steps, use an entity established by previous evidence.
6. Never use placeholders such as "country A" or "entity A".
7. Never jump directly to the final answer if an intermediate entity
   is still unknown.
8. Never ask for information that is already established.
9. Do not explain your reasoning.
10. Do not answer the original question.
11. Return JSON only.

Required format:
{{"query": "ONE SEARCHABLE QUESTION"}}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict sequential retrieval planner. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    parsed = extract_json(raw_output)

    if parsed is None:
        return "", raw_output

    query = parsed.get("query", "")

    query = clean_query(query)

    if not valid_query(query):
        return "", raw_output

    return query, raw_output


# ============================================================
# ONE QUESTION
# ============================================================

def run_question(question):
    evidence = []
    previous_queries = []

    for step in range(1, MAX_STEPS + 1):

        query, raw_output = generate_next_query(
            original_question=question,
            evidence=evidence,
            previous_queries=previous_queries,
            step_number=step,
        )

        print()
        print(f"STEP {step}")
        print("Raw planner output:")
        print(raw_output)

        if not query:
            print("Planner failed to produce a valid query.")
            break

        if query in previous_queries:
            print("Planner repeated a previous query.")
            break

        previous_queries.append(query)

        print("Accepted query:")
        print(query)

        try:
            results = retrieve(query)
        except Exception as exc:
            print("Retrieval error:", exc)
            break

        if not results:
            print("No retrieval results.")
            break

        print()
        print("Top retrieved evidence:")

        for rank, item in enumerate(
            results[:5],
            start=1,
        ):
            title = item.get("title", "")
            text = item.get("paragraph_text", "")

            print(
                f"{rank}. {title}"
            )

            print(
                f"   {text[:350]}"
            )

        evidence.extend(results[:EVIDENCE_TO_KEEP])

    return previous_queries


# ============================================================
# MAIN
# ============================================================

records = []

with DATASET.open(
    "r",
    encoding="utf-8",
) as file:
    for line in file:
        if line.strip():
            records.append(json.loads(line))


print()
print("=" * 70)
print("AUTOMATIC DECOMPOSITION VALIDATION")
print("=" * 70)
print("Total questions:", len(records))
print("Testing:", min(TEST_QUESTIONS, len(records)))
print()

for index, record in enumerate(
    records[:TEST_QUESTIONS],
    start=1,
):
    question = record["question_text"]

    print()
    print("=" * 70)
    print(f"QUESTION {index}")
    print("=" * 70)

    print()
    print("ORIGINAL QUESTION:")
    print(question)

    queries = run_question(question)

    print()
    print("GENERATED QUERY CHAIN:")

    for step, query in enumerate(
        queries,
        start=1,
    ):
        print(f"{step}. {query}")


print()
print("=" * 70)
print("VALIDATION TEST COMPLETE")
print("=" * 70)