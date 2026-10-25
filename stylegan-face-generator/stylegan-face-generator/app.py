import os

# Reduce CPU memory usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, render_template, jsonify, request

import torch
import numpy as np
from PIL import Image
import colorsys

from model import Generator, MappingNetwork


app = Flask(__name__)

# ============================================================
# SETTINGS
# ============================================================

device = torch.device("cpu")

Z_DIM = 256
W_DIM = 256

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "stylegan.pth"
)

GENERATED_FOLDER = os.path.join(
    app.static_folder,
    "generated"
)

os.makedirs(
    GENERATED_FOLDER,
    exist_ok=True
)


# ============================================================
# CREATE MODELS
# ============================================================

generator = Generator(
    w_dim=W_DIM,
    base_channels=256,
    img_channels=3
)

mapping_network = MappingNetwork(
    z_dim=Z_DIM,
    w_dim=W_DIM,
    n_layers=6
)


# ============================================================
# LOAD STYLEGAN MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)


# Support checkpoint saved as:
# {
#    "generator": ...,
#    "mapping_network": ...
# }

if isinstance(checkpoint, dict):

    if "generator" in checkpoint:

        generator.load_state_dict(
            checkpoint["generator"]
        )

    else:

        # If generator state_dict itself was saved
        try:
            generator.load_state_dict(
                checkpoint,
                strict=True
            )
        except RuntimeError:
            raise RuntimeError(
                "The stylegan.pth file does not contain "
                "a compatible Generator checkpoint."
            )


    if "mapping_network" in checkpoint:

        mapping_network.load_state_dict(
            checkpoint["mapping_network"]
        )

else:

    raise RuntimeError(
        "Invalid stylegan.pth checkpoint."
    )


generator.to(device)
mapping_network.to(device)

generator.eval()
mapping_network.eval()

print("==========================================")
print("StyleGAN model loaded successfully!")
print("==========================================")


# ============================================================
# TENSOR → PIL IMAGE
# ============================================================

def tensor_to_pil(tensor):

    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)

    image = tensor[0].cpu()

    image = image.permute(
        1, 2, 0
    ).numpy()

    image = (
        image * 255
    ).astype(np.uint8)

    return Image.fromarray(image)


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(image, filename):

    path = os.path.join(
        GENERATED_FOLDER,
        filename
    )

    image.save(
        path
    )

    return filename


# ============================================================
# COLOR HELPERS
# ============================================================

def adjust_pixel_color(
    rgb,
    target_hue,
    saturation_strength,
    brightness
):

    r, g, b = rgb / 255.0

    h, s, v = colorsys.rgb_to_hsv(
        r,
        g,
        b
    )

    # Keep some original variation
    new_h = target_hue

    new_s = (
        s * (1 - saturation_strength)
        + 0.55 * saturation_strength
    )

    new_v = np.clip(
        v * brightness,
        0,
        1
    )

    nr, ng, nb = colorsys.hsv_to_rgb(
        new_h,
        new_s,
        new_v
    )

    return np.array(
        [
            nr * 255,
            ng * 255,
            nb * 255
        ]
    )


# ============================================================
# SKIN TONE
# ============================================================

def apply_skin_tone(image, tone):

    if tone == "original":
        return image

    arr = np.array(
        image
    ).astype(np.float32)

    h, w, _ = arr.shape

    # Central face region
    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    cx = w / 2
    cy = h * 0.53

    rx = w * 0.32
    ry = h * 0.40

    face_mask = (
        ((xx - cx) / rx) ** 2
        +
        ((yy - cy) / ry) ** 2
        <= 1
    )

    # RGB skin-like condition
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    skin_mask = (
        (r > g * 1.03)
        &
        (g > b * 1.02)
        &
        (r > 60)
        &
        (g > 40)
    )

    mask = (
        face_mask
        &
        skin_mask
    )

    # Strength
    if tone == "light":

        target = np.array(
            [235, 190, 165]
        )

        strength = 0.38

    elif tone == "medium":

        target = np.array(
            [190, 135, 105]
        )

        strength = 0.32

    elif tone == "dark":

        target = np.array(
            [115, 75, 55]
        )

        strength = 0.35

    else:

        return image

    arr[mask] = (
        arr[mask] * (1 - strength)
        +
        target * strength
    )

    return Image.fromarray(
        np.clip(
            arr,
            0,
            255
        ).astype(np.uint8)
    )


# ============================================================
# HAIR COLOR
# ============================================================

def apply_hair_color(image, color):

    if color == "original":
        return image

    arr = np.array(
        image
    ).astype(np.float32)

    h, w, _ = arr.shape

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    # Hair is mainly around upper/head region
    head_region = (
        (yy < h * 0.47)
        &
        (xx > w * 0.12)
        &
        (xx < w * 0.88)
    )

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Dark hair candidates
    dark_hair = (
        (r < 130)
        &
        (g < 130)
        &
        (b < 130)
    )

    hair_mask = (
        head_region
        &
        dark_hair
    )

    if color == "black":

        target = np.array(
            [25, 22, 20]
        )

        strength = 0.70


    elif color == "brown":

        target = np.array(
            [105, 55, 30]
        )

        strength = 0.65


    elif color == "blonde":

        target = np.array(
            [205, 165, 90]
        )

        strength = 0.65


    else:

        return image


    arr[hair_mask] = (
        arr[hair_mask] * (1 - strength)
        +
        target * strength
    )

    return Image.fromarray(
        np.clip(
            arr,
            0,
            255
        ).astype(np.uint8)
    )


# ============================================================
# NORMAL GENERATION
# ============================================================

@app.route("/generate")
@torch.no_grad()
def generate():

    skin = request.args.get(
        "skin",
        "original"
    )

    hair = request.args.get(
        "hair",
        "original"
    )

    z = torch.randn(
        1,
        Z_DIM,
        device=device
    )

    w = mapping_network(z)

    image = generator(w)

    image = tensor_to_pil(
        image
    )

    # Apply selected controls
    image = apply_skin_tone(
        image,
        skin
    )

    image = apply_hair_color(
        image,
        hair
    )

    filename = "generated_face.png"

    save_image(
        image,
        filename
    )

    return jsonify({
        "success": True,
        "image":
            "/static/generated/"
            + filename
    })


# ============================================================
# STYLE MIXING
# ============================================================

@app.route("/style-mix")
@torch.no_grad()
def style_mix():

    skin = request.args.get(
        "skin",
        "original"
    )

    hair = request.args.get(
        "hair",
        "original"
    )

    z1 = torch.randn(
        1,
        Z_DIM,
        device=device
    )

    z2 = torch.randn(
        1,
        Z_DIM,
        device=device
    )

    w1 = mapping_network(z1)

    w2 = mapping_network(z2)

    num_styles = generator.num_style_inputs

    ws = [w1] * num_styles

    # Mix later synthesis layers
    mix_point = num_styles // 2

    for i in range(
        mix_point,
        num_styles
    ):

        ws[i] = w2

    image = generator(
        ws
    )

    image = tensor_to_pil(
        image
    )

    image = apply_skin_tone(
        image,
        skin
    )

    image = apply_hair_color(
        image,
        hair
    )

    filename = "generated_face.png"

    save_image(
        image,
        filename
    )

    return jsonify({
        "success": True,
        "image":
            "/static/generated/"
            + filename
    })


# ============================================================
# NOISE VARIATION
# ============================================================

@app.route("/noise-variation")
@torch.no_grad()
def noise_variation():

    skin = request.args.get(
        "skin",
        "original"
    )

    hair = request.args.get(
        "hair",
        "original"
    )

    # Same latent style
    z = torch.randn(
        1,
        Z_DIM,
        device=device
    )

    w = mapping_network(z)

    # Generator samples fresh noise
    # internally during each forward pass
    image = generator(w)

    image = tensor_to_pil(
        image
    )

    image = apply_skin_tone(
        image,
        skin
    )

    image = apply_hair_color(
        image,
        hair
    )

    filename = "generated_face.png"

    save_image(
        image,
        filename
    )

    return jsonify({
        "success": True,
        "image":
            "/static/generated/"
            + filename
    })


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )