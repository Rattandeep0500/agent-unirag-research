import hashlib
import json

from elasticsearch import Elasticsearch, helpers

DATASET = r".\original\Agent-UniRAG\raw_data\musique\musique_full_v1.0_dev.jsonl"
INDEX = "musique_dev"


def make_documents():
    seen = set()

    with open(DATASET, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            instance = json.loads(line)

            for paragraph in instance["paragraphs"]:
                title = paragraph["title"]
                paragraph_text = paragraph["paragraph_text"]

                full_id = hashlib.sha256(
                    (title + " " + paragraph_text).encode("utf-8")
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
                        "paragraph_text": paragraph_text,
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

print("Elasticsearch:", es.info()["version"]["number"])

if es.indices.exists(INDEX):
    print("Deleting existing index:", INDEX)
    es.indices.delete(index=INDEX)

settings = {
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

print("Creating index:", INDEX)
es.indices.create(index=INDEX, body=settings)

print("Inserting MuSiQue dev paragraphs...")
success, errors = helpers.bulk(
    es,
    make_documents(),
    raise_on_error=True,
    raise_on_exception=True,
    max_retries=2,
    request_timeout=500,
)

es.indices.refresh(index=INDEX)

count = es.count(index=INDEX)["count"]

print("INDEX BUILD COMPLETE")
print("Documents:", count)
print("Bulk successes:", success)
print("Errors:", len(errors) if errors else 0)
