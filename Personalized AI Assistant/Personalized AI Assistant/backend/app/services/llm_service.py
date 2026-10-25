import gc
import logging
import os
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from backend.app.config import MODEL_NAME, USE_CUDA, OUTPUT_DIR

logger = logging.getLogger(__name__)

_LLM_INSTANCE: Optional['LLMService'] = None


class LLMService:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.base_model = None
        self.active_model = None
        self.loaded_adapter_path: Optional[str] = None
        self.device = 'cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu'
        self.loaded = False
        self.load_error: Optional[str] = None

    def status(self) -> Dict[str, Any]:
        """Provides status of model memory, device, and load state."""
        mem = psutil.virtual_memory()
        cuda_avail = bool(torch.cuda.is_available() if hasattr(torch, 'cuda') else False)
        vram_info = None
        if cuda_avail:
            try:
                vram_info = {
                    "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1),
                    "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1),
                }
            except Exception:
                pass

        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.loaded,
            'cuda_available': cuda_avail,
            'loaded_adapter': self.loaded_adapter_path,
            'error': self.load_error,
            'system_ram_available_gb': round(mem.available / (1024 ** 3), 2),
            'system_ram_total_gb': round(mem.total / (1024 ** 3), 2),
            'vram': vram_info,
        }

    def check_memory_sufficiency(self) -> bool:
        """Check if machine has at least 1.5GB of available RAM."""
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 ** 2)
        if available_mb < 1200:
            logger.warning(f"Low system RAM ({available_mb:.1f} MB available). Model loading might encounter issues.")
        return available_mb >= 500

    def load(self, model_name: Optional[str] = None):
        """Loads base model and tokenizer into memory if not already loaded."""
        target_model = model_name or self.model_name
        if self.loaded and self.active_model is not None and self.model_name == target_model:
            return self.active_model

        if not self.check_memory_sufficiency():
            self.load_error = "Insufficient system RAM to load LLM."
            raise RuntimeError(self.load_error)

        try:
            logger.info(f"Loading tokenizer and model: {target_model} on device: {self.device}")
            self.model_name = target_model
            self.tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32

            self.base_model = AutoModelForCausalLM.from_pretrained(
                target_model,
                device_map='auto' if self.device == 'cuda' else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            if self.device == 'cpu':
                self.base_model.to('cpu')

            self.base_model.eval()
            self.active_model = self.base_model
            self.loaded_adapter_path = None
            self.loaded = True
            self.load_error = None
            logger.info(f"Model {target_model} loaded successfully.")
            return self.active_model
        except Exception as exc:
            self.loaded = False
            self.load_error = str(exc)
            logger.error(f"Failed to load model {target_model}: {exc}")
            raise RuntimeError(f"Could not load the model '{target_model}': {exc}")

    def load_adapter(self, adapter_path: Optional[str] = None):
        """
        Loads a fine-tuned LoRA adapter into the base model.
        Uses PEFT PeftModel.
        """
        if adapter_path is None:
            adapter_path = str(OUTPUT_DIR / "lora_adapter")

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"LoRA adapter directory '{adapter_path}' does not exist.")

        # Ensure base model is loaded
        if not self.loaded or self.base_model is None:
            self.load()

        from peft import PeftModel
        logger.info(f"Attaching LoRA adapter from {adapter_path}")
        # Always attach to clean base model
        self.active_model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.active_model.eval()
        self.loaded_adapter_path = adapter_path
        logger.info("LoRA adapter attached successfully.")
        return self.active_model

    def unload_adapter(self):
        """Reverts active model back to the base model."""
        if self.base_model is not None:
            self.active_model = self.base_model
            self.loaded_adapter_path = None

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_new_tokens: int = 300,
        temperature: float = 0.2,
        do_sample: bool = False,
    ) -> str:
        """
        Generates text using chat template or raw prompt.
        """
        if not self.loaded or self.active_model is None or self.tokenizer is None:
            self.load()

        # Build formatted chat template if possible
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback to direct prompt format
            if system_instruction:
                formatted_input = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_input = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(formatted_input, return_tensors='pt').to(self.active_model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self.active_model.generate(**inputs, **gen_kwargs)

        input_len = inputs['input_ids'].shape[1]
        new_tokens = output[0][input_len:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return generated_text


def get_llm_service(model_name: Optional[str] = None) -> LLMService:
    global _LLM_INSTANCE
    target_name = model_name or MODEL_NAME
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = LLMService(model_name=target_name)
    elif model_name and _LLM_INSTANCE.model_name != model_name:
        _LLM_INSTANCE.model_name = model_name
        _LLM_INSTANCE.loaded = False
    return _LLM_INSTANCE
