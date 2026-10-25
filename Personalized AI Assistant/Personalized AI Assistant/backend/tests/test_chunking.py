import pytest
from backend.app.services.chunking import chunk_text, chunk_document_pages


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short():
    text = "This is a brief text."
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_overlapping():
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    # Check overlap between first two chunks
    chunk1_words = chunks[0].split()
    chunk2_words = chunks[1].split()
    # The last words of chunk 1 should match the start of chunk 2
    overlap_words = chunk1_words[-10:]
    assert chunk2_words[:10] == overlap_words


def test_chunk_document_pages():
    pages = [
        {"page_number": 1, "text": "Page one text that has some words for testing."},
        {"page_number": 2, "text": "Page two text with further interesting details."}
    ]
    chunks = chunk_document_pages(
        pages=pages,
        document_id="doc_123",
        filename="test.txt",
        chunk_size=10,
        overlap=2
    )
    assert len(chunks) >= 2
    assert chunks[0]["document_id"] == "doc_123"
    assert chunks[0]["filename"] == "test.txt"
    assert chunks[0]["page_number"] == 1
    assert "Page one" in chunks[0]["text"]
