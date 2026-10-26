import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_lora")


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.samples = []
        for item in data:
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            # Mask out padding tokens from loss computation
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_path: str = "backend/training/sample_train.jsonl",
    output_dir: str = "backend/outputs/lora_adapter",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    epochs: int = 3,
    use_qlora: bool = False
):
    start_time = time.time()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Validate dataset
    data_file = Path(dataset_path).resolve()
    if not data_file.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_file}")

    raw_data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))

    if not raw_data:
        raise ValueError("Dataset is empty. Provide at least one instruction-response pair.")

    logger.info(f"Loaded {len(raw_data)} training samples from {data_file}")

    # Hardware detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Check QLoRA compatibility
    qlora_active = False
    if use_qlora:
        if device == "cuda":
            try:
                import bitsandbytes
                qlora_active = True
                logger.info("QLoRA 4-bit quantization enabled via bitsandbytes.")
            except ImportError:
                logger.warning("bitsandbytes not installed. Falling back to standard LoRA.")
        else:
            logger.warning("QLoRA requires CUDA GPU. Falling back to standard LoRA on CPU.")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info(f"Loading base model: {model_name}")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    if device == "cpu":
        model.to("cpu")

    # Configure PEFT LoRA
    logger.info(f"Configuring LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable params: {trainable_params:,} || All params: {all_param:,} || "
        f"Trainable %: {100 * trainable_params / all_param:.4f}%"
    )

    # Prepare DataLoader
    dataset = InstructionDataset(raw_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps
    )

    model.train()
    loss_history = []
    epoch_losses = []

    logger.info(f"Starting LoRA training for {epochs} epochs ({total_steps} steps)...")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        step_count = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loss_val = float(loss.item())
            epoch_loss += loss_val
            step_count += 1
            loss_history.append({"epoch": epoch, "step": step + 1, "loss": round(loss_val, 4)})

        avg_loss = epoch_loss / max(1, step_count)
        epoch_losses.append({"epoch": epoch, "average_loss": round(avg_loss, 4)})
        logger.info(f"Epoch {epoch}/{epochs} completed - Average Loss: {avg_loss:.4f}")

    # Save trained PEFT adapter weights
    logger.info(f"Saving fine-tuned LoRA adapter to {out_path}")
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    duration = round(time.time() - start_time, 2)

    # Save training report / history
    history = {
        "status": "completed",
        "model_name": model_name,
        "dataset_path": str(data_file),
        "output_dir": str(out_path),
        "duration_seconds": duration,
        "device": device,
        "qlora_active": qlora_active,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "trainable_parameters": trainable_params,
        "total_parameters": all_param,
        "trainable_percentage": round(100 * trainable_params / all_param, 4),
        "final_loss": epoch_losses[-1]["average_loss"] if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "loss_history": loss_history,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    history_path = out_path.parent / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training successfully completed in {duration}s! History saved to {history_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for Personalized AI Assistant")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="backend/training/sample_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="backend/outputs/lora_adapter")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--use_qlora", action="store_true")

    args = parser.parse_args()
    train(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_qlora=args.use_qlora,
    )
