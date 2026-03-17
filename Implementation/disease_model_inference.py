import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("outputs/models")

diabetes_model = joblib.load(MODEL_DIR / "rf_reduced_4_tuned.pkl")
diabetes_features = joblib.load(MODEL_DIR / "diabetes_feature_columns.pkl")

threshold_path = MODEL_DIR / "diabetes_threshold.pkl"
diabetes_threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.50


def predict_diabetes_risk(glucose, bmi, blood_pressure, age):
    if glucose is None:
        return {
            "prediction": None,
            "risk_score": None,
            "confidence_score": None,
            "risk_level": "Insufficient Data",
            "reasons": ["Glucose value could not be extracted from the report."],
        }

    row = pd.DataFrame([{
        "Glucose": float(glucose),
        "BMI": float(bmi) if bmi is not None else 0.0,
        "BloodPressure": float(blood_pressure) if blood_pressure is not None else 0.0,
        "Age": int(age) if age is not None else 0,
    }])

    row = row[diabetes_features]

    proba = float(diabetes_model.predict_proba(row)[0][1])
    pred = 1 if proba >= diabetes_threshold else 0

    if proba < 0.30:
        level = "Low"
    elif proba < 0.60:
        level = "Moderate"
    else:
        level = "High"

    reasons = []

    if glucose >= 126:
        reasons.append(f"Glucose is elevated ({glucose})")
    elif glucose >= 100:
        reasons.append(f"Glucose is above ideal range ({glucose})")

    if bmi is not None:
        if bmi >= 30:
            reasons.append(f"BMI is in the obese range ({bmi})")
        elif bmi >= 25:
            reasons.append(f"BMI is in the overweight range ({bmi})")

    if blood_pressure is not None:
        if blood_pressure >= 90:
            reasons.append(f"Blood pressure is elevated ({blood_pressure})")
        elif blood_pressure >= 80:
            reasons.append(f"Blood pressure is above ideal range ({blood_pressure})")

    if age is not None and age >= 45:
        reasons.append(f"Age is a contributing factor ({age})")

    # Clinical override for realism
    if glucose >= 126:
        level = "High"
        proba = max(proba, 0.70)
    elif glucose >= 100:
        level = "Moderate"
        proba = max(proba, 0.40)

    return {
        "prediction": int(pred),
        "risk_score": round(proba, 2),
        "confidence_score": round(max(proba, 1 - proba), 2),
        "risk_level": level,
        "reasons": reasons if reasons else ["No strong contributing indicators detected."],
    }
