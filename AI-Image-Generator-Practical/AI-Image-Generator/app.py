import os
import uuid
import requests

from urllib.parse import quote
from flask import Flask, render_template, request, send_from_directory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

# Flask application
app = Flask(__name__)

# Image generation model
MODEL_ID = "flux"

# Generated images folder
GENERATED_FOLDER = os.path.join(
    app.root_path,
    "generated_images"
)

os.makedirs(GENERATED_FOLDER, exist_ok=True)


# Home page
@app.route("/", methods=["GET", "POST"])
def home():

    image_path = None
    error = None

    if request.method == "POST":

        prompt = request.form.get("prompt", "").strip()

        if not prompt:
            error = "Please enter a prompt."

        elif len(prompt) > 2000:
            error = "Prompt is too long. Please use fewer than 2000 characters."

        elif not POLLINATIONS_API_KEY:
            error = "API key is missing. Please configure POLLINATIONS_API_KEY in your .env file."

        else:
            try:
                print("Generating image for:", prompt)

                image_url = (
                    f"https://gen.pollinations.ai/image/{quote(prompt)}"
                    f"?model={MODEL_ID}"
                )

                response = requests.get(
                    image_url,
                    headers={
                        "Authorization": f"Bearer {POLLINATIONS_API_KEY}"
                    },
                    timeout=180
                )

                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type", ""
                )

                if not content_type.startswith("image/"):
                    raise ValueError(
                        "The API did not return a valid image."
                    )

                # Save generated image
                filename = f"{uuid.uuid4()}.jpg"

                save_path = os.path.join(
                    GENERATED_FOLDER,
                    filename
                )

                with open(save_path, "wb") as image_file:
                    image_file.write(response.content)

                image_path = f"/image/{filename}"

                print("Image generated successfully!")

            except requests.exceptions.Timeout:
                error = "Image generation timed out. Please try again."

            except requests.exceptions.HTTPError as e:
                status = (
                    e.response.status_code
                    if e.response is not None
                    else "unknown"
                )

                print("API HTTP error:", status)

                if status == 401:
                    error = "Invalid API key. Please check your .env file."

                elif status == 402:
                    error = "API credits or payment are required. Check your provider account."

                elif status == 403:
                    error = "API access denied. Check your account permissions."

                elif status == 429:
                    error = "Rate limit reached. Please try again later."

                else:
                    error = f"Image generation failed (HTTP {status})."

            except requests.exceptions.RequestException as e:
                print("Network error:", e)
                error = "Unable to connect to the image generation service."

            except Exception as e:
                print("Unexpected error:", e)
                error = "Something went wrong while generating the image."

    return render_template(
        "index.html",
        image_path=image_path,
        error=error
    )


# Serve generated images
@app.route("/image/<filename>")
def image(filename):

    return send_from_directory(
        GENERATED_FOLDER,
        filename
    )


# Run Flask application
if __name__ == "__main__":

    print("=" * 40)
    print("AI IMAGE GENERATOR")
    print("Model:", MODEL_ID)
    print("Starting Flask server...")
    print("=" * 40)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )