import os
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.services.evaluation_service import EvaluationService
from backend.app.services.fine_tuning_service import FineTuningService

router = APIRouter()


class EvaluationRunPayload(BaseModel):
    modes: Optional[List[str]] = None


class TrainingRunPayload(BaseModel):
    epochs: int = 2
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    batch_size: int = 1
    use_qlora: bool = False


@router.get('/evaluation/results')
async def get_results():
    service = EvaluationService()
    return service.get_results()


@router.post('/evaluation/run')
async def run_evaluation(payload: Optional[EvaluationRunPayload] = None):
    service = EvaluationService()
    try:
        modes = payload.modes if payload else None
        results = service.run_evaluation(modes=modes)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc


@router.get('/evaluation/export')
async def export_results():
    service = EvaluationService()
    csv_file = service.export_csv()
    if not csv_file or not os.path.exists(csv_file):
        raise HTTPException(status_code=404, detail="No evaluation CSV results available. Run evaluation first.")
    return FileResponse(
        path=csv_file,
        filename="model_evaluation_results.csv",
        media_type="text/csv"
    )


@router.get('/training/status')
async def get_training_status():
    ft_service = FineTuningService()
    return ft_service.get_status()


@router.post('/training/run')
async def trigger_training(payload: TrainingRunPayload, background_tasks: BackgroundTasks):
    ft_service = FineTuningService()
    try:
        # Run training in background or directly
        history = ft_service.train(
            epochs=payload.epochs,
            lora_rank=payload.lora_rank,
            lora_alpha=payload.lora_alpha,
            learning_rate=payload.learning_rate,
            batch_size=payload.batch_size,
            use_qlora=payload.use_qlora
        )
        return {"status": "completed", "history": history}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
