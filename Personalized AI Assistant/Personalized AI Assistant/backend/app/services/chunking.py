from typing import List, Dict, Any
from backend.app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping chunks based on word count.
    Preserves context across chunk boundaries.
    """
    if not text or not text.strip():
        return []

    words = text.strip().split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break

    return chunks


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    document_id: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Processes a list of page dicts [{"page_number": int, "text": str}],
    chunks the text while maintaining document and page metadata.
    Returns list of chunk objects.
    """
    all_chunks = []
    global_chunk_idx = 0

    for page in pages:
        page_num = page.get("page_number", 1)
        page_text = page.get("text", "")
        if not page_text:
            continue

        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for chunk in page_chunks:
            all_chunks.append({
                "chunk_id": f"{document_id}_p{page_num}_c{global_chunk_idx}",
                "document_id": document_id,
                "filename": filename,
                "page_number": page_num,
                "chunk_index": global_chunk_idx,
                "text": chunk,
            })
            global_chunk_idx += 1

    return all_chunks
