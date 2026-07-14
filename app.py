"""
app.py
------
Flask web app for the Student Performance Predictor.

Serves a form-based UI where a user enters study hours, attendance, and
previous score, and gets back a predicted final score, letter grade, and
pass/fail probability from the trained model.

Usage:
    python3 app.py
    then open http://127.0.0.1:5000 in a browser
"""

import os
import sys

from flask import Flask, render_template, request, jsonify

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import StudentPerformanceModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "performance_model.pkl")

app = Flask(__name__)

# Load model once at startup
model = StudentPerformanceModel()
if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)
else:
    model = None


def score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    elif score >= 40:
        return "E"
    else:
        return "F"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained yet. Run train.py first."}), 500

    try:
        data = request.get_json()
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance"])
        previous_score = float(data["previous_score"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid input. Please check your values."}), 400

    # Basic sanity clamping so wild inputs don't break the UI
    study_hours = max(0, min(study_hours, 80))
    attendance = max(0, min(attendance, 100))
    previous_score = max(0, min(previous_score, 100))

    result = model.predict(study_hours, attendance, previous_score)
    result["grade"] = score_to_grade(result["predicted_final_score"])

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
