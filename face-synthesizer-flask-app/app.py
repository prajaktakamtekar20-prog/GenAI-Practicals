from flask import Flask, render_template, jsonify
import torch
from torchvision.utils import save_image
import os
from model import Generator


app = Flask(__name__)

device = torch.device("cpu")

noise_dim = 100


generator = Generator(noise_dim)

generator.load_state_dict(
    torch.load(
        "generator.pth",
        map_location=device
    )
)

generator.to(device)
generator.eval()

print("Generator model loaded successfully!")


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/generate")
def generate_face():

    noise = torch.randn(
        1,
        noise_dim,
        1,
        1,
        device=device
    )

    with torch.no_grad():

        generated_face = generator(noise)

    os.makedirs(
        "static/generated",
        exist_ok=True
    )

    image_path = "static/generated/generated_face.png"

    save_image(
        generated_face,
        image_path,
        normalize=True
    )

    return jsonify({
        "success": True,
        "image": "/static/generated/generated_face.png"
    })


if __name__ == "__main__":
    
    app.run(
        debug=False,
        use_reloader=False
    )