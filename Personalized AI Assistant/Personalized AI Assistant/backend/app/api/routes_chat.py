import datetime
import json
import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.config import STORAGE_DIR, TOP_K
from backend.app.services.llm_service import get_llm_service
from backend.app.services.rag_service import RAGService
from backend.app.services.fine_tuning_service import FineTuningService

logger = logging.getLogger(__name__)
router = APIRouter()

CHAT_HISTORY_FILE = STORAGE_DIR / "chat_history.json"


def load_chat_history() -> List[Dict[str, Any]]:
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            return []
    return []


def save_chat_history(history: List[Dict[str, Any]]):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


class ChatRequestPayload(BaseModel):
    model_config = {"protected_namespaces": ()}
    question: str = Field(..., min_length=1)
    mode: str = Field("rag", pattern="^(base|rag|fine_tuned)$")
    model_name: Optional[str] = None


@router.get('/chat/history')
async def get_history():
    return {'history': load_chat_history()}


@router.delete('/chat/history')
async def clear_history():
    save_chat_history([])
    return {'status': 'cleared', 'message': 'Chat conversation cleared.'}


@router.post('/chat')
async def chat(payload: ChatRequestPayload):
    question = payload.question.strip()
    mode = payload.mode
    model_name = payload.model_name

    llm = get_llm_service(model_name)
    try:
        llm.load()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"LLM could not be loaded: {exc}") from exc

    context = ""
    sources = []
    retrieval_score = None
    is_unsupported = False
    answer = ""

    rag = RAGService()
    ft_service = FineTuningService()

    if mode == 'base':
        # Unload any active adapter for pure base model evaluation
        llm.unload_adapter()
        system_prompt = "You are a helpful, concise AI assistant. Answer the user's question directly."
        answer = llm.generate(
            prompt=question,
            system_instruction=system_prompt,
            max_new_tokens=280,
            temperature=0.2,
            do_sample=False
        )

    elif mode == 'rag':
        llm.unload_adapter()
        hits = rag.retrieve(question, top_k=TOP_K)

        if not hits or len(hits) == 0:
            is_unsupported = True
            answer = (
                "I cannot answer this based on the available documents. "
                "Please upload relevant files or ask a question supported by the indexed knowledge base."
            )
        else:
            retrieval_score = hits[0].get("score")
            # If highest retrieval similarity is very low (e.g. < 0.25), consider it out-of-context
            if retrieval_score is not None and retrieval_score < 0.25:
                is_unsupported = True
                answer = (
                    "I cannot answer this based on the available documents. "
                    "The indexed knowledge base does not contain sufficiently relevant information for this query."
                )
            else:
                context, sources = rag.build_rag_context(hits)
                sys_inst, user_prompt = rag.format_prompt(question, context)
                answer = llm.generate(
                    prompt=user_prompt,
                    system_instruction=sys_inst,
                    max_new_tokens=320,
                    temperature=0.2,
                    do_sample=False
                )
                if "cannot answer" in answer.lower() or "insufficient" in answer.lower():
                    is_unsupported = True

    elif mode == 'fine_tuned':
        if not ft_service.has_trained_adapter():
            answer = (
                "Fine-tuned LoRA adapter is pending. "
                "No trained adapter found in backend/outputs/lora_adapter. "
                "Please run the LoRA fine-tuning script to generate adapter weights."
            )
            is_unsupported = True
        else:
            try:
                llm.load_adapter()
                # Fine-tuned model answers with adapted domain style
                hits = rag.retrieve(question, top_k=TOP_K)
                if hits:
                    retrieval_score = hits[0].get("score")
                    context, sources = rag.build_rag_context(hits)
                    sys_inst, user_prompt = rag.format_prompt(question, context)
                else:
                    sys_inst = "You are a specialized personalized AI assistant fine-tuned on academic AI concepts."
                    user_prompt = question

                answer = llm.generate(
                    prompt=user_prompt,
                    system_instruction=sys_inst,
                    max_new_tokens=300,
                    temperature=0.2,
                    do_sample=False
                )
            except Exception as e:
                logger.error(f"Error executing fine-tuned model: {e}")
                answer = f"Error generating answer with fine-tuned adapter: {e}"

    response_data = {
        'answer': answer,
        'mode': mode,
        'model': llm.model_name,
        'sources': sources,
        'retrieval_score': retrieval_score,
        'is_unsupported': is_unsupported,
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Record message pair in history
    history = load_chat_history()
    history.append({
        'id': f"user_{len(history)+1}",
        'role': 'user',
        'content': question,
        'timestamp': response_data['timestamp'],
    })
    history.append({
        'id': f"assistant_{len(history)+1}",
        'role': 'assistant',
        'content': answer,
        'mode': mode,
        'sources': sources,
        'retrieval_score': retrieval_score,
        'is_unsupported': is_unsupported,
        'timestamp': response_data['timestamp'],
    })
    save_chat_history(history)

    return response_data
