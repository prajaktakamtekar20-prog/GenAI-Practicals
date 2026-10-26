from typing import List, Optional, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field("rag", pattern="^(base|rag|fine_tuned)$")
    model_name: Optional[str] = None


class SourceCitation(BaseModel):
    filename: str
    page_number: Optional[int] = None
    excerpt: str
    score: Optional[float] = None
    chunk_index: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    mode: str
    model: str
    sources: List[SourceCitation]
    retrieval_score: Optional[float] = None
    is_unsupported: bool = False
    timestamp: Optional[str] = None
