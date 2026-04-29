import os
import numpy as np
import pandas as pd
import matplotlib

# Force GUI backend (fix for plot not opening)
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1. Safe Dataset Path Handling
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "real_vibration_dataset.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

data = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# -----------------------------
# 2. Split Features and Labels
# -----------------------------
train_files = ["97.mat", "98.mat", "105.mat", "171.mat"]
test_files = ["99.mat", "100.mat", "108.mat", "211.mat"]

train_mask = data["source"].isin(train_files)
test_mask = data["source"].isin(test_files)

X = data.drop(columns=["label", "source"])
y = data["label"]

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print("Unique labels   :", np.unique(y))

# -----------------------------
# 3. SVM with GridSearch (Multi-Class)
# -----------------------------
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        probability=True,
        decision_function_shape="ovr",
        class_weight="balanced",
        random_state=42
    ))
])

svm_param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.01, 0.001],
    "svm__kernel": ["rbf","poly"]
}

svm_grid = GridSearchCV(
    svm_pipeline,
    svm_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train, y_train)
best_svm = svm_grid.best_estimator_
svm_pred = best_svm.predict(X_test)

# -----------------------------
# 4. Random Forest with GridSearch
# -----------------------------
rf_param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 10],      # ← Force shallower trees
    "class_weight": ["balanced"] # ← ADD THIS
}


rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
rf_pred = best_rf.predict(X_test)

# -----------------------------
# 5. Evaluation
# -----------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Results")
    print("-" * 50)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n",
          classification_report(y_true, y_pred, zero_division=0))


evaluate_model("SVM", y_test, svm_pred)
evaluate_model("Random Forest", y_test, rf_pred)

# -----------------------------
# 6. Cross Validation
# -----------------------------
svm_cv = cross_val_score(best_svm, X_train, y_train, cv=5)
rf_cv = cross_val_score(best_rf, X_train, y_train, cv=5)

print("\nCross Validation Accuracy")
print("SVM Mean:", np.mean(svm_cv))
print("RF Mean :", np.mean(rf_cv))

# -----------------------------
# 7. Multi-Class ROC Curve
# -----------------------------

n_classes = len(np.unique(y))

# Binarize labels
y_test_bin = label_binarize(y_test, classes=np.unique(y))

# Get probability predictions (use Random Forest)
rf_probs = best_rf.predict_proba(X_test)

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve (Random Forest)")
plt.legend()
plt.tight_layout()

roc_path = os.path.join(BASE_DIR, "roc_curve.png")
plt.savefig(roc_path)
print(f"ROC curve saved at: {roc_path}")

plt.show(block=True)
print("RF Mean :", np.mean(rf_cv))
print("RF Mean :", np.mean(rf_cv))




# -----------------------------
# 7. Feature Importance Plot
# -----------------------------
importances = best_rf.feature_importances_

plt.figure(figsize=(8, 6))
plt.barh(X.columns, importances)
plt.xlabel("Importance Score")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plot_path = os.path.join(BASE_DIR, "feature_importance.png")
plt.savefig(plot_path)
print(f"Feature importance graph saved at: {plot_path}")

plt.show(block=True)

# -----------------------------
# 8. Save Models
# -----------------------------
models_folder = os.path.join(BASE_DIR, "models")

if not os.path.exists(models_folder):
    os.makedirs(models_folder)

joblib.dump(best_rf, os.path.join(models_folder, "random_forest.pkl"))
joblib.dump(best_svm, os.path.join(models_folder, "svm.pkl"))

print("Models saved successfully.")


