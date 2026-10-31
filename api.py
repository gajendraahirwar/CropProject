import pickle

import pandas as pd
from flask import Flask, jsonify, request

# Flask serves both the API and the frontend files from the "frontend" folder
app = Flask(__name__, static_folder="frontend", static_url_path="")

# Load the trained model once, when the server starts
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Input fields, in the same order used during training, with allowed ranges
FIELDS = {
    "N": (0, 200),
    "P": (0, 200),
    "K": (0, 200),
    "temperature": (0, 60),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 500),
}


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    # Validate every field: it must exist, be a number and be inside its range
    values = []
    for name, (low, high) in FIELDS.items():
        try:
            value = float(data[name])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": f"'{name}' is missing or not a number"}), 400
        if not low <= value <= high:
            return jsonify({"error": f"'{name}' must be between {low} and {high}"}), 400
        values.append(value)

    # Build a one-row table with the same column names used in training
    row = pd.DataFrame([values], columns=list(FIELDS.keys()))

    # Probability of every crop, then pick the best three
    probabilities = model.predict_proba(row)[0]
    best = probabilities.argsort()[::-1][:3]
    top = [{"crop": model.classes_[i], "probability": round(float(probabilities[i]) * 100, 1)}
           for i in best]

    return jsonify({"crop": top[0]["crop"], "top": top})


if __name__ == "__main__":
    app.run(debug=True)