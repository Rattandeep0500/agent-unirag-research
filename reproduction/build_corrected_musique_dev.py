import hashlib
import json
from elasticsearch import Elasticsearch, helpers

FILES = [
    r".\original\Agent-UniRAG\raw_data\musique\musique_ans_v1.0_dev.jsonl",
    r".\original\Agent-UniRAG\raw_data\musique\musique_full_v1.0_dev.jsonl",
]

INDEX = "musique_dev_corrected"

def documents():
    seen = set()

    for path in FILES:
        print("Reading:", path)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)

                for paragraph in obj.get("paragraphs", []):
                    title = paragraph.get("title", "")
                    text = paragraph.get("paragraph_text", "")

                    if not text:
                        continue

                    full_id = hashlib.sha256(
                        (title + " " + text).encode("utf-8")
                    ).hexdigest()

                    if full_id in seen:
                        continue

                    seen.add(full_id)

                    yield {
                        "_index": INDEX,
                        "_id": full_id[:32],
                        "_source": {
                            "id": full_id[:32],
                            "title": title,
                            "paragraph_index": 0,
                            "paragraph_text": text,
                            "url": "",
                            "is_abstract": True,
                        },
                    }

es = Elasticsearch(
    [{"host": "localhost", "port": 9200}],
    max_retries=20,
    timeout=2000,
    retry_on_timeout=True,
)

if es.indices.exists(INDEX):
    es.indices.delete(index=INDEX)

mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "english"},
            "paragraph_index": {"type": "integer"},
            "paragraph_text": {"type": "text", "analyzer": "english"},
            "url": {"type": "text", "analyzer": "english"},
            "is_abstract": {"type": "boolean"},
        }
    }
}

print("Creating:", INDEX)
es.indices.create(index=INDEX, body=mapping)

success, errors = helpers.bulk(
    es,
    documents(),
    raise_on_error=True,
    raise_on_exception=True,
    max_retries=2,
    request_timeout=500,
)

es.indices.refresh(index=INDEX)

count = es.count(index=INDEX)["count"]

print("\nINDEX BUILD COMPLETE")
print("Documents:", count)
print("Bulk successes:", success)
print("Errors:", len(errors) if errors else 0)

# Direct sanity check for the previously missing entity.
resp = es.search(
    index=INDEX,
    body={
        "query": {
            "match": {
                "paragraph_text": "Purrysburg"
            }
        },
        "size": 3,
    },
)

print("\nPURRYSBURG CHECK")
for hit in resp["hits"]["hits"]:
    print(hit["_source"]["title"])
