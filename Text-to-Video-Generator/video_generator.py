from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


MODEL_ID = "THUDM/CogVideoX-2b"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
NUM_FRAMES = 49
FPS = 8


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(device: str) -> CogVideoXPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

    if device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)

    return pipeline


def generate_video(
    pipeline: CogVideoXPipeline,
    prompt: str,
    steps: int,
    seed: int,
) -> Path:
    if not prompt.strip():
        raise ValueError("The prompt must not be empty.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"video_{timestamp}_{uuid4().hex[:8]}.mp4"

    result = pipeline(
        prompt=prompt.strip(),
        num_frames=NUM_FRAMES,
        num_inference_steps=steps,
        guidance_scale=6,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    export_to_video(result.frames[0], str(output_path), fps=FPS)
    return output_path
