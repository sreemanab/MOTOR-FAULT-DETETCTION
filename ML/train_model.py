import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("data/processed/motor_vibration_dataset.csv")

X = df.drop("label", axis=1)   # Features
y = df["label"]                # Target

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# RANDOM FOREST CLASSIFIER
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,      # number of trees
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = rf_model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(rf_model, "ml/saved_model.pkl")
print("\n✅ Random Forest model saved successfully")
