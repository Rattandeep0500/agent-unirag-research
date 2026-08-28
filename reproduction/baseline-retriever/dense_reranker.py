from sentence_transformers import SentenceTransformer, util

EMBEDDING_RERANKER = "dense_reranker_models/multilingual-e5-large"


class DenseReranker:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_RERANKER)

    def rerank(self, query, docs, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        query_embedding = util.normalize_embeddings(query_embedding)
        doc_embeddings = self.model.encode(docs, convert_to_tensor=True)
        doc_embeddings = util.normalize_embeddings(doc_embeddings)
        scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
        scores = scores.cpu().numpy()[0]
        sorted_indices = scores.argsort()[::-1][:top_k]
        return [
            {
                "doc_id": doc_id,
                "dense_score": float(scores[doc_id]),
            }
            for doc_id in sorted_indices
        ]
