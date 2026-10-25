from typing import Optional, List, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    file_type: str
    upload_status: str
    processing_status: str
    notes: Optional[str] = None
    extracted_text_preview: Optional[str] = None
    page_count: int = 1
    chunk_count: int = 0
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]


class IndexResponse(BaseModel):
    document_count: int
    chunk_count: int
    vector_index_path: str
    status: str
