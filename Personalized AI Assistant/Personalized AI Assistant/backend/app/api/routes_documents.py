import datetime
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.config import STORAGE_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
from backend.app.services.document_loader import extract_document_text, save_uploaded_file
from backend.app.services.chunking import chunk_document_pages
from backend.app.services.rag_service import RAGService
from backend.app.utils.file_validation import validate_upload_file

logger = logging.getLogger(__name__)
router = APIRouter()

DOCUMENTS_METADATA_FILE = STORAGE_DIR / "documents.json"


def load_documents_metadata() -> List[Dict[str, Any]]:
    """Loads document records from disk."""
    if DOCUMENTS_METADATA_FILE.exists():
        try:
            with open(DOCUMENTS_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading documents metadata: {e}")
            return []
    return []


def save_documents_metadata(docs: List[Dict[str, Any]]):
    """Saves document records to disk."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCUMENTS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


@router.get('/documents')
async def list_documents():
    docs = load_documents_metadata()
    return {'documents': docs}


@router.get('/documents/stats')
async def get_document_stats():
    rag = RAGService()
    vector_stats = rag.vector_store.get_stats()
    docs = load_documents_metadata()
    total_docs = len(docs)
    total_pages = sum(d.get("page_count", 1) for d in docs)
    return {
        'total_documents': total_docs,
        'total_pages': total_pages,
        'vector_stats': vector_stats,
    }


@router.post('/documents/upload', status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    try:
        validate_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Check for duplicate filename in existing records
    existing_docs = load_documents_metadata()
    for doc in existing_docs:
        if doc.get('filename') == file.filename:
            logger.info(f"Replacing duplicate upload for file: {file.filename}")

    try:
        file_path, stored_name = save_uploaded_file(file, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Could not save uploaded file: {exc}') from exc

    extension = Path(file.filename).suffix.lower()
    try:
        pages, is_scanned = extract_document_text(file_path, extension)
        total_text = " ".join(p.get("text", "") for p in pages)
        preview = total_text[:600] if total_text else ""
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f'Could not extract text: {exc}') from exc

    processing_status = "ready"
    notes = None
    if is_scanned and extension == ".pdf":
        processing_status = "scanned_pdf_warning"
        notes = "This PDF appears to contain scanned images with little or no selectable text. OCR is required to extract text."
    elif not preview:
        processing_status = "empty"
        notes = "Document contains no extractable text."

    record = {
        'document_id': stored_name,
        'filename': file.filename,
        'size_bytes': os.path.getsize(file_path),
        'file_type': extension,
        'upload_status': 'uploaded',
        'processing_status': processing_status,
        'notes': notes,
        'extracted_text_preview': preview,
        'page_count': len(pages),
        'chunk_count': 0,
        'uploaded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Remove any prior duplicate with same stored_name or filename
    filtered_docs = [d for d in existing_docs if d.get('filename') != file.filename]
    filtered_docs.append(record)
    save_documents_metadata(filtered_docs)

    return record


@router.delete('/documents/{document_id}')
async def delete_document(document_id: str):
    docs = load_documents_metadata()
    found = False
    new_docs = []

    for doc in docs:
        if doc['document_id'] == document_id:
            found = True
            # Delete stored file if exists
            stored_file = STORAGE_DIR / document_id
            if stored_file.exists():
                try:
                    os.remove(stored_file)
                except Exception as e:
                    logger.warning(f"Failed to remove file from disk: {e}")
        else:
            new_docs.append(doc)

    if not found:
        raise HTTPException(status_code=404, detail='Document not found')

    save_documents_metadata(new_docs)

    # Also remove from FAISS vector store
    rag = RAGService()
    rag.vector_store.delete_document(document_id, embedding_service=rag.embedding_service)

    return {'status': 'deleted', 'document_id': document_id}


@router.post('/documents/index')
async def index_documents():
    docs = load_documents_metadata()
    if not docs:
        rag = RAGService()
        rag.vector_store.clear()
        return {
            'document_count': 0,
            'chunk_count': 0,
            'vector_index_path': str(STORAGE_DIR / 'vector_index.faiss'),
            'status': 'no_documents'
        }

    rag = RAGService()
    rag.vector_store.clear()
    total_indexed_chunks = 0

    updated_docs = []
    for record in docs:
        file_path = STORAGE_DIR / record['document_id']
        if not file_path.exists():
            continue

        extension = Path(record['filename']).suffix.lower()
        try:
            pages, _ = extract_document_text(str(file_path), extension)
            chunks = chunk_document_pages(
                pages=pages,
                document_id=record['document_id'],
                filename=record['filename'],
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP
            )
            count = rag.index_document_chunks(chunks)
            total_indexed_chunks += count
            record_copy = dict(record)
            record_copy['chunk_count'] = count
            record_copy['processing_status'] = 'indexed'
            updated_docs.append(record_copy)
        except Exception as e:
            logger.error(f"Error indexing document {record['filename']}: {e}")
            record_copy = dict(record)
            record_copy['processing_status'] = 'indexing_error'
            updated_docs.append(record_copy)

    save_documents_metadata(updated_docs)

    return {
        'document_count': len(updated_docs),
        'chunk_count': total_indexed_chunks,
        'vector_index_path': rag.vector_store.index_path,
        'status': 'indexed'
    }
