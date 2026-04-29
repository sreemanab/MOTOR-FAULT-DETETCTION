import os
import numpy as np
import joblib
from flask import Flask, render_template, request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "Motor_Fault_Detection", "models", "random_forest.pkl")


model = joblib.load(MODEL_PATH)

app = Flask(__name__)

class_labels = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Outer Race Fault"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["peak"]),
        float(request.form["rms"]),
        float(request.form["mean"]),
        float(request.form["std"]),
    ]

    prediction = model.predict([features])[0]
    confidence = np.max(model.predict_proba([features])) * 100

    return render_template(
        "result.html",
        result=class_labels[prediction],
        confidence=round(confidence, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)