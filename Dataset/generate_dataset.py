# -------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# -------------------------------
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# -------------------------------
# IMPORTS
# -------------------------------
import pandas as pd

from Simulation.normal_vibration import normal_vibration
from Simulation.faulty_vibration import (
    imbalance_vibration,
    bearing_fault_vibration
)
from Simulation.feature_extraction import extract_features

# -------------------------------
# ENSURE OUTPUT DIRECTORY EXISTS
# -------------------------------
OUTPUT_DIR = os.path.join("data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# DATASET GENERATION PARAMETERS
# -------------------------------
SAMPLES_PER_CLASS = 150

data = []

# -------------------------------
# GENERATE DATA
# -------------------------------
for _ in range(SAMPLES_PER_CLASS):
    # Normal motor condition
    signal = normal_vibration()
    features = extract_features(signal)
    features["label"] = 0
    data.append(features)

    # Imbalance fault
    signal = imbalance_vibration()
    features = extract_features(signal)
    features["label"] = 1
    data.append(features)

    # Bearing fault
    signal = bearing_fault_vibration()
    features = extract_features(signal)
    features["label"] = 2
    data.append(features)

# -------------------------------
# CREATE DATAFRAME & SAVE
# -------------------------------
df = pd.DataFrame(data)

output_path = os.path.join(
    OUTPUT_DIR,
    "motor_vibration_dataset.csv"
)

df.to_csv(output_path, index=False)

print("✅ Dataset created successfully!")
print("📁 Saved at:", output_path)
print("📊 Dataset shape:", df.shape)
print(df.head())
