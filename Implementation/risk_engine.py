
# Stage 3: Simple rule-based risk scoring + explanations (non-diagnostic)
# Plugging this into the pipeline after Stage 2 CBC extraction.

from __future__ import annotations
from typing import Dict, Any, List, Optional


def _is_abnormal(flag: Optional[str]) -> bool:
    return (flag or "").strip().lower() in {"low", "high"}


def compute_risk_from_cbc(cbc: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:

    if not cbc:
        return {
            "risk_score": 0.0,
            "risk_level": "Unknown",
            "abnormal_count": 0,
            "reasons": ["No CBC indicators were extracted."],
            "recommendations": [
                "Try a clearer report scan or a different report format.",
                "This tool is for informational support only."
            ],
        }

    reasons: List[str] = []
    abnormal_count = 0

    # Score weighting 
    # - Each abnormal indicator adds points
    # - Certain “key” indicators get a bit more weight
    key_weights = {
        "Hemoglobin": 0.10,
        "Total Leukocyte Count": 0.10,
        "Platelet Count": 0.10,
    }
    default_weight = 0.06

    score = 0.0

    for name, info in cbc.items():
        flag = info.get("flag")
        if _is_abnormal(flag):
            abnormal_count += 1
            val = info.get("value")
            reasons.append(f"{name} is {flag} ({val})")

            w = key_weights.get(name, default_weight)
            score += w

    # Cap score at 1.0
    score = min(1.0, score)

    # Map score or abnormal_count → risk level.
    
    if abnormal_count == 0:
        level = "Low"
    elif abnormal_count <= 2 and score < 0.25:
        level = "Low"
    elif abnormal_count <= 4 and score < 0.55:
        level = "Moderate"
    else:
        level = "High"

    recommendations = [
        "This output is not a diagnosis. Discuss results with a qualified healthcare professional if concerned.",
        "Abnormal flags can also occur due to temporary factors (recent illness, dehydration, lab variation).",
    ]

    return {
        "risk_score": round(score, 2),
        "risk_level": level,
        "abnormal_count": abnormal_count,
        "reasons": reasons if reasons else ["No abnormal CBC flags detected."],
        "recommendations": recommendations,
    }
