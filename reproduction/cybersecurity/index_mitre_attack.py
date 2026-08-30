import ssl

ssl.SSLContext.load_default_certs = (
    lambda self, purpose=ssl.Purpose.SERVER_AUTH: None
)

import json
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


INPUT = Path(
    r".\cyber_data\processed\mitre_attack.jsonl"
)

INDEX = "mitre_attack"
ES_HOST = "http://127.0.0.1:9200"


def load_documents():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT}"
        )

    documents = []

    with INPUT.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():

                documents.append(
                    json.loads(line)
                )

    return documents


def create_index(es):

    if es.indices.exists(index=INDEX):

        print(
            f"Deleting existing index: {INDEX}"
        )

        es.indices.delete(
            index=INDEX
        )

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {

                "id": {
                    "type": "keyword"
                },

                "title": {
                    "type": "text"
                },

                "paragraph_text": {
                    "type": "text"
                },

                "url": {
                    "type": "keyword"
                },

                "is_abstract": {
                    "type": "boolean"
                },

                "paragraph_index": {
                    "type": "integer"
                },

                "paragraph_type": {
                    "type": "keyword"
                },

                "object_type": {
                    "type": "keyword"
                },

                "name": {
                    "type": "text"
                },

                "description": {
                    "type": "text"
                },

                "aliases": {
                    "type": "text"
                },

                "external_ids": {
                    "type": "keyword"
                },

                "platforms": {
                    "type": "keyword"
                },

                "permissions_required": {
                    "type": "keyword"
                },

                "detection": {
                    "type": "text"
                },

                "created": {
                    "type": "keyword"
                },

                "modified": {
                    "type": "keyword"
                },
            }
        },
    }

    es.indices.create(
        index=INDEX,
        body=mapping,
    )


def index_documents(es, documents):

    actions = []

    for document in documents:

        document_id = document.get("id")

        if not document_id:
            continue

        actions.append(
            {
                "_index": INDEX,
                "_id": document_id,
                "_source": document,
            }
        )

    print(
        "Documents prepared:",
        len(actions),
    )

    success, errors = bulk(
        es,
        actions,
        chunk_size=250,
        raise_on_error=False,
    )

    return success, errors


def main():

    print("========== MITRE ATT&CK INDEXING ==========")

    es = Elasticsearch(
        [ES_HOST],
        request_timeout=60,
    )

    if not es.ping():
        raise RuntimeError(
            "Cannot connect to Elasticsearch."
        )

    print(
        "Connected to Elasticsearch."
    )

    documents = load_documents()

    print(
        "Loaded corpus documents:",
        len(documents),
    )

    create_index(es)

    print(
        "Created index:",
        INDEX,
    )

    success, errors = index_documents(
        es,
        documents,
    )

    es.indices.refresh(
        index=INDEX
    )

    count = es.count(
        index=INDEX
    )["count"]

    print()
    print("========== INDEX COMPLETE ==========")
    print("Documents attempted:", len(documents))
    print("Documents indexed:", success)
    print("Errors:", len(errors))
    print("Elasticsearch count:", count)

    if errors:

        print()
        print("First errors:")

        for error in errors[:5]:
            print(error)


if __name__ == "__main__":
    main()