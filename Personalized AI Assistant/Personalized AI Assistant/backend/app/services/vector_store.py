import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

from backend.app.config import STORAGE_DIR

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, index_path: Optional[str] = None):
        if index_path is None:
            self.index_path = str(STORAGE_DIR / "vector_index.faiss")
        else:
            self.index_path = str(Path(index_path).resolve())

        self.metadata_path = self.index_path + ".json"
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Loads FAISS index and metadata from disk if available."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded vector store from {self.index_path} with {len(self.metadata)} chunks.")
            except Exception as e:
                logger.error(f"Failed to load vector store from disk: {e}")
                self.index = None
                self.metadata = []
        else:
            self.index = None
            self.metadata = []

    def _save(self):
        """Persists FAISS index and metadata to disk."""
        if self.index is not None:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved vector store with {len(self.metadata)} chunks to {self.index_path}")

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Adds normalized embeddings and corresponding metadata."""
        if embeddings is None or len(embeddings) == 0:
            return

        np_embeddings = np.asarray(embeddings, dtype="float32")
        dimension = np_embeddings.shape[1]

        if self.index is None:
            # IndexFlatIP calculates inner product (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(np_embeddings)
        self.metadata.extend(metadata_list)
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for top_k most similar vectors.
        Returns list of dicts with 'score', 'metadata', 'index'.
        """
        if self.index is None or len(self.metadata) == 0:
            return []

        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []

        q_vec = np.asarray([query_vector], dtype="float32")
        distances, indices = self.index.search(q_vec, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[int(idx)]
            score = float(distances[0][i])
            results.append({
                "index": int(idx),
                "score": round(score, 4),
                "metadata": meta,
            })
        return results

    def clear(self):
        """Clears all vectors and metadata in memory and on disk."""
        self.index = None
        self.metadata = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)

    def delete_document(self, document_id: str, embedding_service=None):
        """
        Removes all chunks associated with a document_id and rebuilds the FAISS index.
        """
        remaining_meta = [m for m in self.metadata if m.get("document_id") != document_id]
        if len(remaining_meta) == len(self.metadata):
            return  # No change

        self.clear()
        if remaining_meta and embedding_service is not None:
            texts = [m.get("excerpt", m.get("text", "")) for m in remaining_meta]
            embs = embedding_service.embed_batch(texts)
            self.add(embs, remaining_meta)

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics of the current vector index."""
        unique_docs = len({m.get("document_id") for m in self.metadata if m.get("document_id")})
        return {
            "total_chunks": len(self.metadata),
            "unique_documents": unique_docs,
            "dimension": self.index.d if self.index is not None else 0,
            "index_exists": self.index is not None and self.index.ntotal > 0,
            "index_path": self.index_path,
        }
