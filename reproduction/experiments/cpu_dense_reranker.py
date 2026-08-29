from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


MODEL_PATH = Path(
    r".\original\Agent-UniRAG\retriever_server\dense_reranker_models\multilingual-e5-large"
)

MAX_SEQ_LENGTH = 256
BATCH_SIZE = 8


class CPUDenseReranker:

    def __init__(self):
        self.device = "cpu"

        self.model = SentenceTransformer(
            str(MODEL_PATH),
            device=self.device,
        )

        # Limit transformer input length to reduce CPU inference cost.
        self.model.max_seq_length = MAX_SEQ_LENGTH

    def rerank(self, query, docs, top_k=10):

        if not docs:
            return []

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        doc_embeddings = self.model.encode(
            docs,
            convert_to_tensor=True,
            device=self.device,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        scores = doc_embeddings @ query_embedding

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