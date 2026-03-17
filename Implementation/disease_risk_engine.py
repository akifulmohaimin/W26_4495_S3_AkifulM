# Stage 3: Simple rule-based risk scoring + explanations (non-diagnostic)
# Diabetes + Heart Disease version
# Plug this into the pipeline after Stage 2 indicator extraction.

from __future__ import annotations
from typing import Dict, Any, List, Optional


def _safe_flag(indicators: Dict[str, Dict[str, Any]], key: str) -> str:
    return str(indicators.get(key, {}).get("flag", "")).strip()


def _safe_value(indicators: Dict[str, Dict[str, Any]], key: str) -> Optional[float]:
    value = indicators.get(key, {}).get("value")
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _is_high(indicators: Dict[str, Dict[str, Any]], key: str) -> bool:
    return _safe_flag(indicators, key).lower() == "high"


def _is_low(indicators: Dict[str, Dict[str, Any]], key: str) -> bool:
    return _safe_flag(indicators, key).lower() == "low"


def _yes(value: Any) -> bool:
    return str(value).strip().lower() == "yes"


def _risk_level_from_score(score: float) -> str:
    if score < 0.25:
        return "Low"
    elif score < 0.55:
        return "Moderate"
    return "High"


def _build_insufficient_result(message: str) -> Dict[str, Any]:
    return {
        "risk_level": "Insufficient Data",
        "confidence_score": None,
        "risk_score": None,
        "abnormal_count": 0,
        "reasons": [message],
    }


def _compute_diabetes_risk(
    indicators: Dict[str, Dict[str, Any]],
    patient_inputs: Dict[str, Any]
) -> Dict[str, Any]:
    reasons: List[str] = []
    score = 0.0
    abnormal_count = 0

    found_diabetes_data = False

    glucose = _safe_value(indicators, "glucose")
    hba1c = _safe_value(indicators, "hba1c")
    bmi = patient_inputs.get("bmi")
    age = patient_inputs.get("age")

    if glucose is not None:
        found_diabetes_data = True
        if glucose >= 126:
            score += 0.35
            abnormal_count += 1
            reasons.append(f"Glucose is elevated ({glucose})")
        elif glucose >= 100:
            score += 0.20
            abnormal_count += 1
            reasons.append(f"Glucose is above ideal range ({glucose})")

    if hba1c is not None:
        found_diabetes_data = True
        if hba1c >= 6.5:
            score += 0.35
            abnormal_count += 1
            reasons.append(f"HbA1c is elevated ({hba1c}%)")
        elif hba1c >= 5.7:
            score += 0.20
            abnormal_count += 1
            reasons.append(f"HbA1c is above ideal range ({hba1c}%)")

    if bmi is not None:
        try:
            bmi = float(bmi)
            if bmi >= 30:
                score += 0.15
                reasons.append(f"BMI is in the obese range ({bmi})")
            elif bmi >= 25:
                score += 0.08
                reasons.append(f"BMI is in the overweight range ({bmi})")
        except Exception:
            pass

    if age is not None:
        try:
            age = int(age)
            if age >= 45:
                score += 0.08
                reasons.append(f"Age is a contributing diabetes risk factor ({age})")
        except Exception:
            pass

    if _yes(patient_inputs.get("family_history_diabetes")):
        score += 0.12
        reasons.append("Family history of diabetes reported")

    symptoms = [str(s).strip().lower() for s in patient_inputs.get("symptoms", [])]
    diabetes_symptom_hits = {
        "frequent urination",
        "increased thirst",
        "blurred vision",
        "fatigue",
    }
    matched_symptoms = [s for s in symptoms if s in diabetes_symptom_hits]
    if matched_symptoms:
        score += min(0.12, 0.04 * len(matched_symptoms))
        reasons.append("Reported symptoms may align with diabetes-related concerns")

    # If neither glucose nor hba1c exists, treat as insufficient disease-specific data
    if not found_diabetes_data:
        return _build_insufficient_result("No diabetes-related indicators detected in the uploaded report.")

    score = min(1.0, score)
    level = _risk_level_from_score(score)
    confidence = min(0.95, 0.55 + score * 0.4)

    return {
        "risk_level": level,
        "confidence_score": round(confidence, 2),
        "risk_score": round(score, 2),
        "abnormal_count": abnormal_count,
        "reasons": reasons if reasons else ["No strong diabetes-related abnormalities detected."],
    }


def _compute_heart_risk(
    indicators: Dict[str, Dict[str, Any]],
    patient_inputs: Dict[str, Any]
) -> Dict[str, Any]:
    reasons: List[str] = []
    score = 0.0
    abnormal_count = 0

    found_heart_data = False

    total_chol = _safe_value(indicators, "cholesterol_total")
    ldl = _safe_value(indicators, "ldl")
    hdl = _safe_value(indicators, "hdl")
    triglycerides = _safe_value(indicators, "triglycerides")
    systolic_bp = _safe_value(indicators, "systolic_bp")
    diastolic_bp = _safe_value(indicators, "diastolic_bp")
    bmi = patient_inputs.get("bmi")
    age = patient_inputs.get("age")

    if total_chol is not None:
        found_heart_data = True
        if total_chol >= 240:
            score += 0.18
            abnormal_count += 1
            reasons.append(f"Total cholesterol is high ({total_chol})")
        elif total_chol >= 200:
            score += 0.10
            abnormal_count += 1
            reasons.append(f"Total cholesterol is above ideal range ({total_chol})")

    if ldl is not None:
        found_heart_data = True
        if ldl >= 160:
            score += 0.22
            abnormal_count += 1
            reasons.append(f"LDL is high ({ldl})")
        elif ldl >= 100:
            score += 0.12
            abnormal_count += 1
            reasons.append(f"LDL is above ideal range ({ldl})")

    if hdl is not None:
        found_heart_data = True
        if hdl < 40:
            score += 0.18
            abnormal_count += 1
            reasons.append(f"HDL is low ({hdl})")

    if triglycerides is not None:
        found_heart_data = True
        if triglycerides >= 200:
            score += 0.18
            abnormal_count += 1
            reasons.append(f"Triglycerides are high ({triglycerides})")
        elif triglycerides >= 150:
            score += 0.10
            abnormal_count += 1
            reasons.append(f"Triglycerides are above ideal range ({triglycerides})")

    if systolic_bp is not None:
        found_heart_data = True
        if systolic_bp >= 140:
            score += 0.18
            abnormal_count += 1
            reasons.append(f"Systolic blood pressure is high ({systolic_bp})")
        elif systolic_bp >= 120:
            score += 0.10
            abnormal_count += 1
            reasons.append(f"Systolic blood pressure is above ideal range ({systolic_bp})")

    if diastolic_bp is not None:
        found_heart_data = True
        if diastolic_bp >= 90:
            score += 0.18
            abnormal_count += 1
            reasons.append(f"Diastolic blood pressure is high ({diastolic_bp})")
        elif diastolic_bp >= 80:
            score += 0.10
            abnormal_count += 1
            reasons.append(f"Diastolic blood pressure is above ideal range ({diastolic_bp})")

    if bmi is not None:
        try:
            bmi = float(bmi)
            if bmi >= 30:
                score += 0.12
                reasons.append(f"BMI is in the obese range ({bmi})")
            elif bmi >= 25:
                score += 0.06
                reasons.append(f"BMI is in the overweight range ({bmi})")
        except Exception:
            pass

    if age is not None:
        try:
            age = int(age)
            if age >= 45:
                score += 0.08
                reasons.append(f"Age is a contributing cardiovascular risk factor ({age})")
        except Exception:
            pass

    if _yes(patient_inputs.get("smoker")):
        score += 0.14
        reasons.append("Smoking status increases heart disease risk")

    if _yes(patient_inputs.get("family_history_heart_disease")):
        score += 0.12
        reasons.append("Family history of heart disease reported")

    symptoms = [str(s).strip().lower() for s in patient_inputs.get("symptoms", [])]
    heart_symptom_hits = {
        "chest pain",
        "shortness of breath",
        "dizziness",
        "palpitations",
    }
    matched_symptoms = [s for s in symptoms if s in heart_symptom_hits]
    if matched_symptoms:
        score += min(0.12, 0.04 * len(matched_symptoms))
        reasons.append("Reported symptoms may align with cardiovascular-related concerns")

    if not found_heart_data:
        return _build_insufficient_result("No heart-related indicators detected in the uploaded report.")

    score = min(1.0, score)
    level = _risk_level_from_score(score)
    confidence = min(0.95, 0.55 + score * 0.4)

    return {
        "risk_level": level,
        "confidence_score": round(confidence, 2),
        "risk_score": round(score, 2),
        "abnormal_count": abnormal_count,
        "reasons": reasons if reasons else ["No strong heart-related abnormalities detected."],
    }


def compute_diabetes_heart_risk(
    indicators: Dict[str, Dict[str, Any]],
    patient_inputs: Dict[str, Any]
) -> Dict[str, Any]:
    if not indicators:
        return {
            "diabetes": _build_insufficient_result("No structured indicators were extracted."),
            "heart": _build_insufficient_result("No structured indicators were extracted."),
            "general_notes": [
                "Try a clearer report scan or a different report format.",
                "This tool is for informational support only.",
            ],
        }

    diabetes_result = _compute_diabetes_risk(indicators, patient_inputs)
    heart_result = _compute_heart_risk(indicators, patient_inputs)

    return {
        "diabetes": diabetes_result,
        "heart": heart_result,
        "general_notes": [
            "These outputs are informational and non-diagnostic.",
            "They are based only on the extracted report values and structured patient inputs provided.",
            "Discuss concerning results with a qualified healthcare professional.",
        ],
    }