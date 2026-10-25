import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / '.env')

from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_documents import router as documents_router
from backend.app.api.routes_chat import router as chat_router
from backend.app.api.routes_evaluation import router as evaluation_router

app = FastAPI(
    title='Personalized AI Assistant API',
    version='1.0.0',
    description='Academic API for Personalized AI Assistant with RAG, FAISS Vector Search, and LoRA/QLoRA Fine-Tuning.'
)

# CORS configuration for local Vite development
origins_env = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173')
allowed_origins = [orig.strip() for orig in origins_env.split(',') if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Include API routers
app.include_router(health_router, prefix='/api', tags=['Health & Model'])
app.include_router(documents_router, prefix='/api', tags=['Documents & Knowledge Base'])
app.include_router(chat_router, prefix='/api', tags=['Chat & Assistant'])
app.include_router(evaluation_router, prefix='/api', tags=['Evaluation & Fine-Tuning'])


@app.get('/')
async def root():
    return {
        'service': 'Personalized AI Assistant API',
        'status': 'running',
        'documentation': '/docs',
        'aim': 'To investigate RAG and LoRA fine-tuning on open-source LLM response quality.'
    }
