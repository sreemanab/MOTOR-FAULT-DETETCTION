import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
DATA_PATH = os.path.join("data", "processed", "motor_vibration_dataset.csv")
MODEL_PATH = os.path.join("ml", "saved_model.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1)
y = df["label"]

# Load model
model = joblib.load(MODEL_PATH)

# Predict
y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
