import os
from pathlib import Path
from dotenv import load_dotenv

# Locate project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / '.env')

BACKEND_DIR = BASE_DIR / 'backend'
STORAGE_DIR = Path(os.getenv('LOCAL_STORAGE_DIR', str(BACKEND_DIR / 'storage'))).resolve()
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', str(BACKEND_DIR / 'outputs'))).resolve()

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
TOP_K = int(os.getenv('TOP_K', '5'))

MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
USE_CUDA = os.getenv('USE_CUDA', 'false').strip().lower() in {'1', 'true', 'yes'}
ENABLE_QLORA = os.getenv('ENABLE_QLORA', 'false').strip().lower() in {'1', 'true', 'yes'}
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',')
