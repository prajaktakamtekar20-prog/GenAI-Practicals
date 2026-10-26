from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from backend.app.services.llm_service import get_llm_service

router = APIRouter()


class ModelLoadRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: Optional[str] = None


@router.get('/health')
async def health_check():
    return {
        'status': 'ok',
        'service': 'Personalized AI Assistant',
        'environment': 'local'
    }


@router.get('/model/status')
async def model_status():
    service = get_llm_service()
    try:
        status_info = service.status()
        return {**status_info, 'message': 'Model status retrieved successfully.'}
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}


@router.post('/model/load')
async def load_model(payload: ModelLoadRequest):
    service = get_llm_service(model_name=payload.model_name)
    try:
        service.load(payload.model_name)
        return {
            'status': 'loaded',
            'model_name': service.model_name,
            'device': service.device,
            'message': f"Model '{service.model_name}' loaded successfully into memory."
        }
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}
