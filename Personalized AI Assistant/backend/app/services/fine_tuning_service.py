import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from backend.app.config import OUTPUT_DIR, MODEL_NAME
from backend.training.train_lora import train as execute_lora_train

logger = logging.getLogger(__name__)


class FineTuningService:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.adapter_dir = self.output_dir / "lora_adapter"
        self.history_file = self.output_dir / "training_history.json"

    def has_trained_adapter(self) -> bool:
        """Returns True if a valid LoRA adapter has been saved to disk."""
        adapter_weights = self.adapter_dir / "adapter_model.safetensors"
        adapter_bin = self.adapter_dir / "adapter_model.bin"
        adapter_config = self.adapter_dir / "adapter_config.json"
        return (adapter_weights.exists() or adapter_bin.exists()) and adapter_config.exists()

    def get_status(self) -> Dict[str, Any]:
        """
        Reads real training status from disk.
        Returns 'completed' if training ran, else 'pending' or 'not_trained'.
        Never fabricates results.
        """
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                history["has_adapter"] = self.has_trained_adapter()
                return history
            except Exception as e:
                logger.error(f"Error reading training history: {e}")

        return {
            "status": "pending",
            "has_adapter": self.has_trained_adapter(),
            "message": "LoRA fine-tuning has not been executed yet. Run training to create an adapter.",
            "adapter_dir": str(self.adapter_dir),
            "suggested_command": "python -m backend.training.train_lora --epochs 2",
        }

    def train(
        self,
        model_name: Optional[str] = None,
        dataset_path: str = "backend/training/sample_train.jsonl",
        epochs: int = 2,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        learning_rate: float = 2e-4,
        batch_size: int = 1,
        use_qlora: bool = False
    ) -> Dict[str, Any]:
        """
        Executes LoRA training workflow.
        """
        target_model = model_name or MODEL_NAME
        logger.info(f"Triggering LoRA fine-tuning for model: {target_model}")
        history = execute_lora_train(
            model_name=target_model,
            dataset_path=dataset_path,
            output_dir=str(self.adapter_dir),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            use_qlora=use_qlora,
        )
        return history
