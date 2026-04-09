from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional


MODEL_DIR = Path("outputs/models")


# -----------------------------
# LOAD DIABETES MODEL
# -----------------------------
diabetes_model = joblib.load(MODEL_DIR / "rf_reduced_4_tuned.pkl")
diabetes_features = joblib.load(MODEL_DIR / "diabetes_feature_columns.pkl")

threshold_path = MODEL_DIR / "diabetes_threshold.pkl"
diabetes_threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.50


# -----------------------------
# LOAD HEART MODEL
# -----------------------------
heart_model = joblib.load(MODEL_DIR / "best_heart_model_pipeline.pkl")


# -----------------------------
# HELPERS
# -----------------------------
def _insufficient_result(message: str) -> Dict[str, Any]:
    return {
        "prediction": None,
        "risk_score": None,
        "confidence_score": None,
        "risk_level": "Insufficient Data",
        "reasons": [message],
    }


def _risk_level_from_probability(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Moderate"
    return "High"


# -----------------------------
# DIABETES ML PREDICTION
# -----------------------------
def predict_diabetes_risk(
    glucose: Optional[float],
    bmi: Optional[float],
    blood_pressure: Optional[float],
    age: Optional[int],
) -> Dict[str, Any]:

    if glucose is None:
        return _insufficient_result("Glucose value could not be extracted from the report.")

    row = pd.DataFrame([{
        "Glucose": float(glucose),
        "BMI": float(bmi) if bmi is not None else 0.0,
        "BloodPressure": float(blood_pressure) if blood_pressure is not None else 0.0,
        "Age": int(age) if age is not None else 0,
    }])

    row = row[diabetes_features]

    proba = float(diabetes_model.predict_proba(row)[0][1])
    pred = 1 if proba >= diabetes_threshold else 0
    level = _risk_level_from_probability(proba)

    reasons: List[str] = []

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


# -----------------------------
# HEART ML PREDICTION
# -----------------------------
def predict_heart_risk(
    age: Optional[int],
    sex: Optional[str],
    chest_pain_type: Optional[str],
    resting_bp: Optional[float],
    cholesterol: Optional[float],
    fasting_bs: Optional[int],
    resting_ecg: Optional[str],
    max_hr: Optional[float],
    exercise_angina: Optional[str],
    oldpeak: Optional[float],
    st_slope: Optional[str],
) -> Dict[str, Any]:

    required_fields = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }

    missing = [k for k, v in required_fields.items() if v is None]
    if missing:
        return _insufficient_result(
            f"Missing heart-risk inputs: {', '.join(missing)}."
        )

    row = pd.DataFrame([{
        "Age": int(age),
        "Sex": str(sex),
        "ChestPainType": str(chest_pain_type),
        "RestingBP": float(resting_bp),
        "Cholesterol": float(cholesterol),
        "FastingBS": int(fasting_bs),
        "RestingECG": str(resting_ecg),
        "MaxHR": float(max_hr),
        "ExerciseAngina": str(exercise_angina),
        "Oldpeak": float(oldpeak),
        "ST_Slope": str(st_slope),
    }])

    proba = float(heart_model.predict_proba(row)[0][1])
    pred = 1 if proba >= 0.50 else 0
    level = _risk_level_from_probability(proba)

    reasons: List[str] = []

    if chest_pain_type == "ASY":
        reasons.append("Asymptomatic chest pain pattern is associated with higher heart risk")
    if fasting_bs == 1:
        reasons.append("Fasting blood sugar is elevated")
    if exercise_angina == "Y":
        reasons.append("Exercise-induced angina increases cardiovascular concern")
    if oldpeak is not None and float(oldpeak) >= 1.0:
        reasons.append(f"Oldpeak is elevated ({oldpeak})")
    if st_slope in {"Flat", "Down"}:
        reasons.append(f"ST slope pattern ({st_slope}) may indicate elevated cardiovascular risk")
    if cholesterol is not None and float(cholesterol) >= 200:
        reasons.append(f"Cholesterol is above ideal range ({cholesterol})")
    if resting_bp is not None and float(resting_bp) >= 140:
        reasons.append(f"Resting blood pressure is elevated ({resting_bp})")
    if age is not None and int(age) >= 45:
        reasons.append(f"Age is a contributing cardiovascular factor ({age})")

    return {
        "prediction": int(pred),
        "risk_score": round(proba, 2),
        "confidence_score": round(max(proba, 1 - proba), 2),
        "risk_level": level,
        "reasons": reasons if reasons else ["No strong contributing indicators detected."],
    }


# -----------------------------
# CO-OCCURRENCE FUSION
# -----------------------------
def compute_cooccurrence_risk(
    diabetes_result: Dict[str, Any],
    heart_result: Dict[str, Any],
) -> Dict[str, Any]:

    diabetes_score = diabetes_result.get("risk_score")
    heart_score = heart_result.get("risk_score")

    if diabetes_score is None and heart_score is None:
        return _insufficient_result("Diabetes and heart disease risk scores are unavailable.")

    if diabetes_score is None:
        final_score = float(heart_score)
        reasons = ["Only heart disease risk was available, so co-occurrence risk is based on heart risk alone."]
    elif heart_score is None:
        final_score = float(diabetes_score)
        reasons = ["Only diabetes risk was available, so co-occurrence risk is based on diabetes risk alone."]
    else:
        combined_average = (float(diabetes_score) + float(heart_score)) / 2.0
        interaction_effect = float(diabetes_score) * float(heart_score)
        final_score = (combined_average + interaction_effect) / 2.0

        reasons: List[str] = []

        if diabetes_score >= 0.60:
            reasons.append("Diabetes risk is elevated.")
        elif diabetes_score >= 0.30:
            reasons.append("Diabetes-related indicators are moderately elevated.")

        if heart_score >= 0.60:
            reasons.append("Heart disease risk is elevated.")
        elif heart_score >= 0.30:
            reasons.append("Cardiovascular indicators are moderately elevated.")

        if diabetes_score >= 0.60 and heart_score >= 0.60:
            reasons.append("Both risks are elevated, increasing possible co-occurrence risk.")
        elif diabetes_score >= 0.40 and heart_score >= 0.40:
            reasons.append("Both conditions show overlapping moderate risk patterns.")

    level = _risk_level_from_probability(final_score)
    prediction = 1 if final_score >= 0.50 else 0
    confidence = max(final_score, 1 - final_score)

    return {
        "prediction": int(prediction),
        "risk_score": round(final_score, 2),
        "confidence_score": round(confidence, 2),
        "risk_level": level,
        "reasons": reasons if reasons else ["No strong overlapping chronic disease indicators detected."],
    }


# -----------------------------
# MAIN COMBINED FUNCTION
# -----------------------------
def compute_combined_ml_risk(patient_inputs: Dict[str, Any]) -> Dict[str, Any]:

    diabetes_result = predict_diabetes_risk(
        glucose=patient_inputs.get("glucose"),
        bmi=patient_inputs.get("bmi"),
        blood_pressure=patient_inputs.get("blood_pressure"),
        age=patient_inputs.get("age"),
    )

    heart_result = predict_heart_risk(
        age=patient_inputs.get("age"),
        sex=patient_inputs.get("sex"),
        chest_pain_type=patient_inputs.get("chest_pain_type"),
        resting_bp=patient_inputs.get("resting_bp"),
        cholesterol=patient_inputs.get("cholesterol"),
        fasting_bs=patient_inputs.get("fasting_bs"),
        resting_ecg=patient_inputs.get("resting_ecg"),
        max_hr=patient_inputs.get("max_hr"),
        exercise_angina=patient_inputs.get("exercise_angina"),
        oldpeak=patient_inputs.get("oldpeak"),
        st_slope=patient_inputs.get("st_slope"),
    )

    cooccurrence_result = compute_cooccurrence_risk(diabetes_result, heart_result)

    return {
        "diabetes": diabetes_result,
        "heart": heart_result,
        "cooccurrence": cooccurrence_result,
        "general_notes": [
            "These outputs are informational and non-diagnostic.",
            "Co-occurrence risk reflects overlapping chronic disease patterns.",
            "Discuss concerning results with a qualified healthcare professional.",
        ],
    }
