from time import perf_counter

from dense_reranker import DenseReranker
from fastapi import FastAPI, Request
from unified_retriever import UnifiedRetriever

RERANKER_PATH = "dense_reranker_models/multilingual-e5-large"
retriever = UnifiedRetriever(host="http://localhost/", port=9200)
reranker = DenseReranker(RERANKER_PATH)
app = FastAPI()


@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server."}


@app.post("/retrieve/")
async def retrieve(
    arguments: Request,
):  # see the corresponding method in unified_retriever.py
    arguments = await arguments.json()
    retrieval_method = arguments.pop("retrieval_method")
    assert retrieval_method in ("retrieve_from_elasticsearch")
    start_time = perf_counter()
    retrieval = getattr(retriever, retrieval_method)(**arguments)
    end_time = perf_counter()
    time_in_seconds = round(end_time - start_time, 1)
    return {"retrieval": retrieval, "time_in_seconds": time_in_seconds}


@app.post("/retrieve_and_rerank/")
async def retrieve_and_rerank(
    arguments: Request,
):
    arguments = await arguments.json()
    retrieval_method = arguments.pop("retrieval_method")
    assert retrieval_method in ("retrieve_from_elasticsearch")
    start_time = perf_counter()
    docs = getattr(retriever, retrieval_method)(
        **dict(
            query_text=arguments["query_text"],
            max_hits_count=arguments["sparse_max_hits_count"],
            document_type=arguments["sparse_document_type"],
            corpus_name=arguments["corpus_name"],
        )
    )
    docs_text = [doc["title"] + "\n" + doc["paragraph_text"] for doc in docs]
    reranker_results = reranker.rerank(
        query=arguments["query_text"],
        docs=docs_text,
        top_k=arguments["dense_max_hits_count"],
    )
    reraanked_docs = []
    for result in reranker_results:
        doc_id = result["doc_id"]
        docs[doc_id]["dense_score"] = result["dense_score"]
        reraanked_docs.append(docs[doc_id])
    end_time = perf_counter()
    time_in_seconds = round(end_time - start_time, 1)
    return {"retrieval": reraanked_docs, "time_in_seconds": time_in_seconds}
