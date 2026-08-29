from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util


MODEL_PATH = Path(
    r".\original\Agent-UniRAG\retriever_server\dense_reranker_models\multilingual-e5-large"
)


class CPUDenseReranker:

    def __init__(self):
        self.device = "cpu"

        self.model = SentenceTransformer(
            str(MODEL_PATH),
            device=self.device,
        )

    def rerank(self, query, docs, top_k=10):

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            device=self.device,
        )

        doc_embeddings = self.model.encode(
            docs,
            convert_to_tensor=True,
            device=self.device,
            batch_size=8,
            show_progress_bar=False,
        )

        query_embedding = util.normalize_embeddings(query_embedding.unsqueeze(0))[0]
        doc_embeddings = util.normalize_embeddings(doc_embeddings)

        scores = util.pytorch_cos_sim(
            query_embedding,
            doc_embeddings,
        )[0]

        sorted_indices = torch.argsort(
            scores,
            descending=True,
        )[:top_k]

        return [
            {
                "doc_id": int(doc_id),
                "dense_score": float(scores[doc_id].item()),
            }
            for doc_id in sorted_indices
        ]
