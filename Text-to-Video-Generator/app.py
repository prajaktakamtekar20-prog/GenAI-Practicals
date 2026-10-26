import streamlit as st
from diffusers import CogVideoXPipeline

from video_generator import MODEL_ID, generate_video, load_pipeline, resolve_device


st.set_page_config(page_title="Text to Video", page_icon="🎬", layout="centered")
st.title("Text to Video")
st.write(
    "Generate a short video from a text prompt with the pretrained "
    "CogVideoX-2B model. No dataset or model training is required."
)

device = resolve_device()
st.caption(f"Compute device: **{device}** · Model: **{MODEL_ID}**")
if device == "cpu":
    st.warning(
        "No supported GPU was detected. Generation on CPU can take a very long "
        "time and may require substantial memory."
    )


@st.cache_resource(show_spinner=False)
def get_pipeline(device_name: str) -> CogVideoXPipeline:
    return load_pipeline(device_name)


with st.form("generation_form"):
    prompt = st.text_area(
        "Describe the video",
        placeholder="A small sailboat crossing a calm lake at sunset, cinematic",
        max_chars=500,
        help="Be specific about the subject, action, setting, and visual style.",
    )
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider(
            "Inference steps",
            min_value=10,
            max_value=50,
            value=30,
            help="More steps can improve detail but take longer.",
        )
    with col2:
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            help="Reuse a seed to reproduce a generation.",
        )
    submitted = st.form_submit_button("Generate video", type="primary")

if submitted:
    if not prompt.strip():
        st.error("Enter a text prompt before generating a video.")
    else:
        with st.spinner("Loading the model and generating your video…"):
            pipeline = get_pipeline(device)
            video_path = generate_video(
                pipeline=pipeline,
                prompt=prompt.strip(),
                steps=steps,
                seed=int(seed),
            )

        st.success("Video generated.")
        st.video(str(video_path))
        st.download_button(
            "Download MP4",
            data=video_path.read_bytes(),
            file_name=video_path.name,
            mime="video/mp4",
        )
