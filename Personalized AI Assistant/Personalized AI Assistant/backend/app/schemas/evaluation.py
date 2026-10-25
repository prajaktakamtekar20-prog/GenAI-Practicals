from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class EvaluationQuestion(BaseModel):
    id: Optional[int] = None
    question: str
    reference_answer: str
    supporting_documents: List[str] = []
    category: Optional[str] = "general"


class ModeSummary(BaseModel):
    mode: str
    label: str
    questions_evaluated: int
    average_correctness_f1: float
    average_groundedness: float
    average_citation_accuracy: float
    average_latency_ms: float
    unsupported_refusal_rate: float
    details: List[Dict[str, Any]]


class EvaluationResultsResponse(BaseModel):
    status: str
    evaluated_at: Optional[str] = None
    total_duration_seconds: Optional[float] = None
    model_name: Optional[str] = None
    has_adapter: bool = False
    modes_evaluated: List[str] = []
    mode_summaries: List[ModeSummary] = []
