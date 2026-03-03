import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

for d in (OUTPUT_DIR, MODEL_DIR, METRIC_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

ZERO_MISSING_COLS = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]

def load_and_clean_data():
    """
    Loads Kaggle Pima diabetes CSV and applies cleaning:
    - replace invalid 0s with NaN for certain clinical columns
    - median imputation
    """
    df = pd.read_csv(DATA_PATH)

    if "Outcome" not in df.columns:
        raise ValueError("Expected 'Outcome' column in diabetes.csv. Check the file headers.")

    # Replace invalid zeros with NaN
    for col in ZERO_MISSING_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Median imputation for all columns
    imputer = SimpleImputer(strategy="median")
    df[df.columns] = imputer.fit_transform(df[df.columns])

    df["Outcome"] = df["Outcome"].round().astype(int)
    return df


def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
        "Classification Report": classification_report(y_true, y_pred, zero_division=0)
    }