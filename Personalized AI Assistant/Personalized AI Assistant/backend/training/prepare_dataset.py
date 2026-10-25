import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_dataset")


def load_and_validate_dataset(jsonl_path: str) -> List[Dict[str, str]]:
    """
    Validates that each row in the JSONL dataset contains 'instruction' and 'response'
    non-empty string fields.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    valid_samples = []
    line_number = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line_number += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping line {line_number}: Invalid JSON: {exc}")
                continue

            instruction = str(data.get("instruction", "")).strip()
            response = str(data.get("response", "")).strip()

            if not instruction or not response:
                logger.warning(f"Skipping line {line_number}: Missing 'instruction' or 'response'")
                continue

            valid_samples.append({
                "instruction": instruction,
                "response": response
            })

    logger.info(f"Successfully validated {len(valid_samples)} training samples from {jsonl_path}")
    return valid_samples


def format_for_sft(samples: List[Dict[str, str]], output_path: str) -> str:
    """
    Formats the samples into standard chat or prompt-response pairs and saves as JSON.
    """
    formatted = []
    for s in samples:
        formatted.append({
            "prompt": f"<|im_start|>user\n{s['instruction']}<|im_end|>\n<|im_start|>assistant\n",
            "completion": f"{s['response']}<|im_end|>",
            "instruction": s["instruction"],
            "response": s["response"]
        })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(formatted)} prepared samples to {output_path}")
    return str(out_file)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "backend/training/sample_train.jsonl"
    dest = sys.argv[2] if len(sys.argv) > 2 else "backend/outputs/prepared_train.json"
    samples = load_and_validate_dataset(src)
    format_for_sft(samples, dest)
