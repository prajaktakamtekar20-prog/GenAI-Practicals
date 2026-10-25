import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Flask application
app = Flask(__name__)

# Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

# Gemini client
client = genai.Client(api_key=API_KEY)

# Gemini foundation model
MODEL_NAME = "gemini-3-flash-preview"


# ---------------------------------------------------------
# AI Instructions
# ---------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are a helpful Generative AI chatbot for college students.

Follow these rules:

1. Give short and clear answers.
2. Answer in simple English.
3. Usually answer in 2 to 5 sentences.
4. Do not give unnecessary details.
5. If the user asks for a definition, give a short definition.
6. If the user asks for an explanation, explain simply.
7. Use bullet points when they make the answer easier to understand.
8. Give an example only when it is useful.
9. Do not repeat the user's question.
10. Give a detailed answer only if the user specifically asks for details.
11. Focus on Python, AI, Machine Learning, Deep Learning,
    TensorFlow and Generative AI when relevant.
"""


# ---------------------------------------------------------
# Home Page
# ---------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------
# Chat API
# ---------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        user_message = data.get("message", "").strip()

        if not user_message:

            return jsonify({
                "reply": "Please enter a question."
            })


        # -------------------------------------------------
        # Create prompt with AI instructions
        # -------------------------------------------------

        prompt = SYSTEM_INSTRUCTION + "\n\nUser Question:\n" + user_message


        # -------------------------------------------------
        # Send request to Gemini
        # -------------------------------------------------

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )


        # -------------------------------------------------
        # Return AI response
        # -------------------------------------------------

        return jsonify({
            "reply": response.text
        })


    except Exception as error:

        print("Error:", error)

        return jsonify({
            "reply": "Sorry, something went wrong. Please try again."
        }), 500


# ---------------------------------------------------------
# Run Application
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)