import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_EMBEDDING_INSTANCE: Optional['EmbeddingService'] = None


class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2

    def load(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            # Determine embedding dimension
            test_emb = self._model.encode(["test"], convert_to_numpy=True)
            self.dimension = int(test_emb.shape[1])
        return self._model

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        model = self.load()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")

    def embed(self, text: str) -> np.ndarray:
        batch = self.embed_batch([text])
        return batch[0]


def get_embedding_service(model_name: str = EMBEDDING_MODEL) -> EmbeddingService:
    global _EMBEDDING_INSTANCE
    if _EMBEDDING_INSTANCE is None or _EMBEDDING_INSTANCE.model_name != model_name:
        _EMBEDDING_INSTANCE = EmbeddingService(model_name=model_name)
    return _EMBEDDING_INSTANCE
