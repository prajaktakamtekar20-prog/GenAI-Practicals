import os
import tempfile
import numpy as np
import pytest

from backend.app.services.embedding_service import get_embedding_service
from backend.app.services.vector_store import VectorStore
from backend.app.services.rag_service import RAGService


def test_embedding_service():
    emb_service = get_embedding_service()
    vec = emb_service.embed("Hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Verify normalization (L2 norm is approximately 1.0)
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-3


def test_vector_store_add_search_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "test_index.faiss")
        store = VectorStore(index_path=index_path)

        # Mock normalized embeddings
        emb1 = np.array([1.0, 0.0, 0.0], dtype="float32")
        emb2 = np.array([0.0, 1.0, 0.0], dtype="float32")
        embeddings = np.array([emb1, emb2], dtype="float32")

        meta = [
            {"document_id": "doc1", "filename": "doc1.txt", "page_number": 1, "text": "Apples"},
            {"document_id": "doc2", "filename": "doc2.txt", "page_number": 1, "text": "Bananas"}
        ]
        store.add(embeddings, meta)

        # Search for vector close to emb1
        results = store.search(emb1, top_k=2)
        assert len(results) == 2
        assert results[0]["metadata"]["document_id"] == "doc1"
        assert abs(results[0]["score"] - 1.0) < 1e-4

        # Verify disk persistence
        store2 = VectorStore(index_path=index_path)
        assert store2.index is not None
        assert len(store2.metadata) == 2

        # Verify stats
        stats = store2.get_stats()
        assert stats["total_chunks"] == 2
        assert stats["unique_documents"] == 2


def test_rag_service_context_building():
    rag = RAGService()
    mock_hits = [
        {
            "score": 0.89,
            "metadata": {
                "filename": "guide.pdf",
                "page_number": 3,
                "excerpt": "Personalized AI uses RAG to ground outputs in evidence.",
                "chunk_index": 0
            }
        }
    ]
    context, sources = rag.build_rag_context(mock_hits)
    assert "guide.pdf" in context
    assert "Page: 3" in context
    assert len(sources) == 1
    assert sources[0]["filename"] == "guide.pdf"
    assert sources[0]["score"] == 0.89

    sys_inst, user_prompt = rag.format_prompt("What does AI use?", context)
    assert "UNTRUSTED DOCUMENT CONTEXT" in user_prompt
    assert "guide.pdf" in user_prompt
    assert "What does AI use?" in user_prompt
