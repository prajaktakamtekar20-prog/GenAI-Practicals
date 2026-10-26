# Text-to-Video Generator

A Streamlit app that turns a text prompt into a short MP4 using the pretrained,
open-source [CogVideoX-2B](https://huggingface.co/THUDM/CogVideoX-2b) model via
Hugging Face Diffusers. It performs inference only; it does not collect a
dataset or train/fine-tune the model.

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU is strongly recommended. Generation is resource-intensive; CPU
  generation is supported by the app but can be extremely slow.
- Enough disk space for the model weights, which are downloaded on first use.
- Internet access on first run to download the model.

## Run

From this folder, create and activate a virtual environment, then install the
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your hardware by following the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
if the default `torch` package does not support your GPU.

Start the app:

```powershell
streamlit run app.py
```

Enter a prompt, choose the inference-step count and seed, and select **Generate
video**. The app previews the resulting 49-frame video in the browser and offers
an MP4 download. Generated files are also saved under `outputs/`.

## Notes

- The model is loaded and cached for the lifetime of the Streamlit process.
- A generation uses 49 frames at 8 fps (about six seconds).
- On NVIDIA GPUs, model CPU offloading is enabled to reduce VRAM usage; this
  trades some speed for lower GPU memory requirements.
- Model usage is subject to the license and terms published on the model card.
