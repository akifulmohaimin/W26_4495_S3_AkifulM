import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from disease_pipeline import run_extraction_pipeline, extract_disease_indicators
from disease_risk_engine import compute_combined_ml_risk
from authdb import init_db, create_user, verify_user, save_report, load_reports_for_user

st.set_page_config(
    page_title="HealthLens",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        OPENAI_API_KEY = None

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def init_session():
    defaults = {
        "logged_in": False,
        "user_id": None,
        "username": None,
        "record": None,
        "indicators": None,
        "risk": None,
        "chat_history": [],
        "history": [],
        "dev_mode": False,
        "patient_inputs": {
            "age": 25,
            "sex": "Prefer not to say",
            "height_cm": 170.0,
            "weight_kg": 65.0,
            "smoker": "No",
            "family_history_diabetes": "No",
            "family_history_heart_disease": "No",
            "symptoms": [],
            "consent": True,
            "chest_pain_type": "ATA",
            "resting_ecg": "Normal",
            "exercise_angina": "N",
            "st_slope": "Up",
            "max_hr": 150.0,
            "oldpeak": 0.0,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


def reset_analysis_state():
    st.session_state["record"] = None
    st.session_state["indicators"] = None
    st.session_state["risk"] = None
    st.session_state["chat_history"] = []


def logout():
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None
    st.session_state["username"] = None
    st.session_state["history"] = []
    reset_analysis_state()


def apply_custom_css():
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, #fff1f2 0%, transparent 18%),
                    radial-gradient(circle at top right, #eff6ff 0%, transparent 20%),
                    linear-gradient(180deg, #f8fbff 0%, #f3f7fb 100%);
                color: #162235;
            }
            .main .block-container {
                max-width: 1180px;
                padding-top: 0.7rem;
                padding-bottom: 1.2rem;
            }
            header[data-testid="stHeader"] { background: transparent; }
            #MainMenu, footer { visibility: hidden; }
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #edf4fb 0%, #e6eef7 100%);
                border-right: 1px solid #d7e4ef;
            }
            h1, h2, h3, h4 {
                color: #162235 !important;
                font-weight: 800 !important;
                letter-spacing: -0.02em;
            }
            p, label, div { color: #314155; }
            .stButton > button,
            div[data-testid="stFormSubmitButton"] > button {
                background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
                color: white !important;
                border: none !important;
                border-radius: 14px !important;
                font-weight: 700 !important;
                min-height: 2.7rem !important;
                box-shadow: 0 10px 24px rgba(37,99,235,0.18);
            }
            .stButton > button *,
            div[data-testid="stFormSubmitButton"] > button * {
                color: white !important;
                fill: white !important;
            }
            section[data-testid="stSidebar"] .stButton > button,
            section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] > button {
                background: white !important;
                color: #162235 !important;
                border: 1px solid #dce6f1 !important;
                box-shadow: none !important;
            }
            section[data-testid="stSidebar"] .stButton > button *,
            section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] > button * {
                color: #162235 !important;
                fill: #162235 !important;
            }
            button[data-baseweb="tab"] { color: #506075 !important; font-weight: 700 !important; }
            button[data-baseweb="tab"][aria-selected="true"] { color: #2563eb !important; }
            .hl-hero {
                background: linear-gradient(135deg, #11213a 0%, #18365f 45%, #2563eb 100%);
                color: white;
                border-radius: 26px;
                padding: 22px 24px;
                margin-bottom: 14px;
                box-shadow: 0 16px 34px rgba(17,33,58,0.20);
            }
            .hl-hero * { color: white !important; }
            .hl-hero-badge {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 999px;
                background: rgba(255,255,255,0.14);
                font-size: 0.8rem;
                font-weight: 700;
                margin-bottom: 10px;
            }
            .hl-card, .hl-mini-card, .hl-step {
                background: rgba(255,255,255,0.97);
                border: 1px solid #dce6f1;
                border-radius: 18px;
                padding: 14px;
                box-shadow: 0 8px 22px rgba(15,23,42,0.05);
            }
            .hl-mini-card { min-height: 118px; }
            .hl-disclaimer {
                background: #fff7ed;
                border: 1px solid #fed7aa;
                color: #9a3412;
                border-radius: 16px;
                padding: 12px 14px;
                font-size: 0.9rem;
                margin: 8px 0 14px 0;
            }
            .hl-pill {
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                font-size: 0.76rem;
                font-weight: 700;
                margin-right: 6px;
                margin-bottom: 6px;
                border: 1px solid #d6e2ee;
                background: rgba(255,255,255,0.92);
                color: #435269 !important;
            }
            .hl-muted { color: #62748b !important; font-size: 0.88rem; }
            .hl-step-num {
                width: 30px;
                height: 30px;
                border-radius: 999px;
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                color: #1d4ed8;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: 800;
                margin-bottom: 8px;
            }
            div[data-testid="stTextInput"] input,
            div[data-testid="stNumberInput"] input,
            div[data-testid="stTextArea"] textarea {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cfdbe8 !important;
                border-radius: 12px !important;
            }
            div[data-baseweb="select"] > div {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cfdbe8 !important;
                border-radius: 12px !important;
            }
            [data-baseweb="popover"] [role="listbox"] {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cfdbe8 !important;
            }
            [data-baseweb="popover"] [role="option"] {
                background: #ffffff !important;
                color: #0f172a !important;
            }
            [data-baseweb="popover"] [aria-selected="true"] {
                background: #eff6ff !important;
                color: #0f172a !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_css()


def compute_bmi(height_cm: float, weight_kg: float) -> Tuple[Optional[float], str]:
    if height_cm <= 0 or weight_kg <= 0:
        return None, "Invalid"
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return round(bmi, 1), category


def to_float(value, default=None):
    try:
        if value in [None, "", "—"]:
            return default
        return float(value)
    except Exception:
        return default


def map_sex_for_model(sex: str) -> Optional[str]:
    if sex == "Male":
        return "M"
    if sex == "Female":
        return "F"
    return None


def status_style(level: str) -> Dict[str, Any]:
    level = (level or "").lower()
    if "insufficient" in level:
        return {"label": "Insufficient Data", "color": "#64748b", "bg": "#f8fafc", "border": "#cbd5e1", "dot": "#94a3b8", "value": 18}
    if "high" in level or "attention" in level:
        return {"label": "High", "color": "#dc2626", "bg": "#fef2f2", "border": "#fecaca", "dot": "#dc2626", "value": 88}
    if "moderate" in level or "monitor" in level or "medium" in level:
        return {"label": "Medium", "color": "#d97706", "bg": "#fffbeb", "border": "#fde68a", "dot": "#f59e0b", "value": 58}
    if "low" in level or "stable" in level:
        return {"label": "Low", "color": "#059669", "bg": "#ecfdf5", "border": "#a7f3d0", "dot": "#10b981", "value": 26}
    return {"label": "Unknown", "color": "#64748b", "bg": "#f8fafc", "border": "#cbd5e1", "dot": "#94a3b8", "value": 18}


def visual_flag(flag: str):
    flag = (flag or "").lower()
    if flag == "high":
        return {"label": "High", "color": "#dc2626", "bg": "#fef2f2", "border": "#fecaca", "icon": "▲"}
    if flag == "low":
        return {"label": "Low", "color": "#b45309", "bg": "#fff7ed", "border": "#fed7aa", "icon": "▼"}
    return {"label": "Normal", "color": "#059669", "bg": "#ecfdf5", "border": "#a7f3d0", "icon": "●"}


def compact_metric_card(title: str, value: str, subtitle: str = "", accent: str = "#2563eb"):
    st.markdown(f"""
        <div class="hl-mini-card" style="border-top:4px solid {accent};">
            <div style="font-size:0.78rem; color:#66778f; margin-bottom:4px; font-weight:700;">{title}</div>
            <div style="font-size:1.45rem; font-weight:800; color:#132238; margin-bottom:2px;">{value}</div>
            <div style="font-size:0.8rem; color:#73839a;">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


def info_card(title: str, body: str, color: str = "#2563eb", icon: str = "•"):
    st.markdown(f"""
        <div class="hl-card" style="border-left:5px solid {color};">
            <div style="font-weight:800; color:#132238; margin-bottom:6px;">{icon} {title}</div>
            <div style="color:#3f4f63; line-height:1.5; font-size:0.93rem;">{body}</div>
        </div>
    """, unsafe_allow_html=True)


def empty_state_card(title: str, body: str):
    st.markdown(f"""
        <div class="hl-card" style="background:#fbfdff; border-style:dashed;">
            <div style="font-weight:800; margin-bottom:6px;">{title}</div>
            <div class="hl-muted">{body}</div>
        </div>
    """, unsafe_allow_html=True)


def render_gauge_card(title: str, level: str, subtitle: str = ""):
    s = status_style(level)
    percent = s["value"]
    st.markdown(f"""
        <div class="hl-card" style="text-align:center;">
            <div style="font-size:0.88rem; color:#617089; font-weight:700; margin-bottom:10px;">{title}</div>
            <div style="width:122px; height:122px; margin:0 auto 10px auto; border-radius:50%; background:conic-gradient({s['dot']} 0 {percent}%, #e8eef5 {percent}% 100%); display:flex; align-items:center; justify-content:center;">
                <div style="width:88px; height:88px; border-radius:50%; background:white; border:1px solid #e2e8f0; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                    <div style="font-size:1.1rem; font-weight:800; color:{s['color']};">{s['label']}</div>
                    <div style="font-size:0.72rem; color:#64748b;">Model Risk</div>
                </div>
            </div>
            <div style="font-size:0.82rem; color:#6b7a8f;">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


def count_abnormal_indicators(indicators: Dict[str, Dict[str, Any]]) -> int:
    if not indicators:
        return 0
    count = 0
    for item in indicators.values():
        flag = str(item.get("flag", "")).lower()
        if flag in ["high", "low"]:
            count += 1
    return count


def extract_key_indicator_cards(indicators: Dict[str, Dict[str, Any]]):
    if not indicators:
        st.info("No disease indicators detected yet.")
        return
    abnormal = []
    normal = []
    priority_order = ["glucose", "hba1c", "systolic_bp", "diastolic_bp", "cholesterol_total", "ldl", "hdl", "triglycerides"]
    for key in priority_order:
        if key in indicators:
            item = indicators[key]
            flag = str(item.get("flag", "") or "Normal")
            if flag.lower() == "normal":
                normal.append((key, item))
            else:
                abnormal.append((key, item))
    selected = (abnormal + normal)[:6]
    cols = st.columns(3)
    for idx, (key, item) in enumerate(selected):
        flag = str(item.get("flag", "") or "Normal")
        value = item.get("value", "—")
        unit = item.get("unit", "")
        ref = item.get("ref_raw", "")
        vf = visual_flag(flag)
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="hl-card" style="min-height:150px;">
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
                        <div style="font-size:0.88rem; color:#66778f; font-weight:700;">{key.replace('_', ' ').title()}</div>
                        <div style="display:inline-block; padding:5px 9px; border-radius:999px; background:{vf['bg']}; color:{vf['color']}; border:1px solid {vf['border']}; font-size:0.76rem; font-weight:800;">{vf['icon']} {vf['label']}</div>
                    </div>
                    <div style="font-size:1.6rem; font-weight:800; color:#132238; line-height:1.1; margin-bottom:8px;">{value} <span style="font-size:0.86rem; font-weight:700; color:#6b7a8f;">{unit}</span></div>
                    <div class="hl-muted">Reference: {ref if ref else 'Not available'}</div>
                </div>
            """, unsafe_allow_html=True)


def parse_patient_info_from_text(raw_text: str) -> Dict[str, Any]:
    info = {}
    age_match = re.search(r"\bAge:\s*(\d{1,3})\b", raw_text, re.IGNORECASE)
    sex_match = re.search(r"\bSex:\s*(Male|Female)\b", raw_text, re.IGNORECASE)
    if age_match:
        info["age"] = int(age_match.group(1))
    if sex_match:
        s = sex_match.group(1).strip().lower()
        info["sex"] = "Male" if s == "male" else "Female"
    return info


def get_indicator_value(indicators: Dict[str, Dict[str, Any]], *keys, default=None):
    for key in keys:
        item = indicators.get(key)
        if item and item.get("value") not in [None, ""]:
            return item.get("value")
    return default


def normalize_category(value, allowed: List[str], fallback):
    if value is None:
        return fallback
    value = str(value).strip()
    return value if value in allowed else fallback


def generate_lifestyle_recommendations(risk: Dict[str, Any], patient_inputs: Dict[str, Any], indicators: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    diabetes = risk.get("diabetes", {})
    heart = risk.get("heart", {})
    co = risk.get("cooccurrence", {})
    bmi = patient_inputs.get("bmi")
    smoker = patient_inputs.get("smoker", "No")
    symptoms = patient_inputs.get("symptoms", [])
    actions = []

    def add_action(category: str, title: str, why: str, difficulty: str, priority: int):
        actions.append({"category": category, "title": title, "why": why, "difficulty": difficulty, "priority": priority})

    diabetes_level = str(diabetes.get("risk_level", "")).lower()
    heart_level = str(heart.get("risk_level", "")).lower()
    co_level = str(co.get("risk_level", "")).lower()
    glucose_flag = str(indicators.get("glucose", {}).get("flag", "")).lower()
    systolic_flag = str(indicators.get("systolic_bp", {}).get("flag", "")).lower()

    if glucose_flag == "high" or diabetes_level in ["moderate", "high"]:
        add_action("Food", "Swap one sugary drink today for water or unsweetened tea.", "This matters because blood sugar looks like your main issue right now.", "Easy", 1)
        add_action("Activity", "Take a 10-minute walk after dinner.", "A short walk after meals can help bring blood sugar down.", "Easy", 2)
        add_action("Food", "Choose higher-fiber meals and keep meal timing steadier.", "Fiber and steadier meals may help reduce blood sugar spikes.", "Medium", 3)

    if heart_level in ["moderate", "high"] or systolic_flag == "high":
        add_action("Daily Habits", "Track your blood pressure at regular times if you have access to a monitor.", "This helps you see whether the pattern continues between reports.", "Medium", 2)
        add_action("Food", "Cut down one salty or heavily processed food this week.", "That can support blood pressure and heart health.", "Easy", 3)

    if bmi is not None and bmi >= 25:
        add_action("Activity", "Aim for 20 to 30 minutes of comfortable walking on most days.", "Regular movement supports both blood sugar and heart health.", "Medium", 4)

    if smoker == "Yes":
        add_action("Daily Habits", "Reduce smoking triggers where possible this week.", "Smoking can increase heart strain over time.", "Hard", 5)

    if "Chest pain" in symptoms or "Shortness of breath" in symptoms:
        add_action("Follow-Up", "Bring up chest pain or breathing symptoms with a healthcare professional soon.", "Those symptoms are more important than lifestyle tweaks alone.", "Medium", 1)

    if patient_inputs.get("family_history_diabetes") == "Yes":
        add_action("Follow-Up", "Ask whether regular diabetes screening should happen more often.", "Family history can make blood sugar changes more worth watching.", "Easy", 4)

    if patient_inputs.get("family_history_heart_disease") == "Yes":
        add_action("Follow-Up", "Ask how often heart-health follow-up makes sense for you.", "Family history adds more context to your current results.", "Easy", 4)

    if not actions:
        add_action("Daily Habits", "Keep prioritizing sleep, hydration, and regular movement.", "Your current results do not point to one dominant lifestyle concern.", "Easy", 3)

    actions = sorted(actions, key=lambda x: (x["priority"], x["category"], x["title"]))
    grouped = {"Food": [], "Activity": [], "Daily Habits": [], "Follow-Up": []}
    for action in actions:
        grouped[action["category"]].append(action)

    weekly_plan = actions[:4]
    top_action = actions[0]
    next_report_days = 30 if co_level in ["high", "moderate"] or diabetes_level in ["high", "moderate"] or heart_level in ["high", "moderate"] else 60
    return {"actions": actions, "grouped": grouped, "weekly_plan": weekly_plan, "top_action": top_action, "next_report_days": next_report_days}


def generate_priorities(risk: Dict[str, Any], indicators: Dict[str, Dict[str, Any]]) -> List[str]:
    priorities = []
    diabetes_level = str(risk.get("diabetes", {}).get("risk_level", "")).lower()
    heart_level = str(risk.get("heart", {}).get("risk_level", "")).lower()
    co_level = str(risk.get("cooccurrence", {}).get("risk_level", "")).lower()
    if co_level in ["high", "moderate"]:
        priorities.append("Address combined diabetes and heart risk together.")
    if diabetes_level in ["high", "moderate"]:
        priorities.append("Focus on blood sugar.")
    if heart_level in ["high", "moderate"]:
        priorities.append("Improve blood pressure and cholesterol.")
    if str(indicators.get("ldl", {}).get("flag", "")).lower() == "high":
        priorities.append("Work on lowering LDL cholesterol.")
    if str(indicators.get("hdl", {}).get("flag", "")).lower() == "low":
        priorities.append("Support heart health through activity.")
    if not priorities:
        priorities.append("Keep healthy habits and keep monitoring.")
    return priorities[:3]


def generate_doctor_questions(risk: Dict[str, Any], indicators: Dict[str, Dict[str, Any]]) -> List[str]:
    questions = []
    if str(risk.get("diabetes", {}).get("risk_level", "")).lower() in ["moderate", "high"]:
        questions.append("Should I repeat my blood sugar or HbA1c test?")
    if str(risk.get("heart", {}).get("risk_level", "")).lower() in ["moderate", "high"]:
        questions.append("Do these BP and cholesterol values need follow-up?")
    if str(risk.get("cooccurrence", {}).get("risk_level", "")).lower() in ["moderate", "high"]:
        questions.append("Do these results together suggest a higher overall chronic disease risk?")
    if str(indicators.get("glucose", {}).get("flag", "")).lower() == "high":
        questions.append("Could this suggest prediabetes or diabetes risk?")
    if str(indicators.get("cholesterol_total", {}).get("flag", "")).lower() == "high" or str(indicators.get("ldl", {}).get("flag", "")).lower() == "high":
        questions.append("What lifestyle change should I start with first?")
    if not questions:
        questions.append("Are any follow-up tests needed based on this report?")
    return questions[:4]


def overall_health_summary(risk: Dict[str, Any]) -> Tuple[str, str]:
    diabetes_level = (risk.get("diabetes", {}).get("risk_level", "") or "").lower()
    heart_level = (risk.get("heart", {}).get("risk_level", "") or "").lower()
    co_level = (risk.get("cooccurrence", {}).get("risk_level", "") or "").lower()
    if "high" in [diabetes_level, heart_level, co_level]:
        return "Needs Attention", "Some findings may need closer follow-up."
    if "moderate" in [diabetes_level, heart_level, co_level]:
        return "Monitor Closely", "A few indicators deserve attention."
    if "low" in [diabetes_level, heart_level, co_level]:
        return "Generally Stable", "No major model risk signal appears dominant."
    return "No Analysis Yet", "Upload a report and run the pipeline."


def patient_friendly_explanation(risk: Dict[str, Any], indicators: Dict[str, Dict[str, Any]], patient_inputs: Dict[str, Any]) -> str:
    diabetes = risk.get("diabetes", {})
    heart = risk.get("heart", {})
    co = risk.get("cooccurrence", {})
    d_level = str(diabetes.get("risk_level", "unknown")).lower()
    h_level = str(heart.get("risk_level", "unknown")).lower()
    c_level = str(co.get("risk_level", "unknown")).lower()
    lines = [
        f"Your report suggests {d_level} diabetes model risk, {h_level} heart model risk, and {c_level} combined model risk.",
        "HealthLens is meant to make that easier to understand, not to diagnose you.",
    ]
    if str(indicators.get("glucose", {}).get("flag", "")).lower() == "high":
        lines.append("Your blood sugar looks higher than expected, so that is one of the main things to discuss.")
    if str(indicators.get("systolic_bp", {}).get("flag", "")).lower() == "high":
        lines.append("Your blood pressure may also be worth reviewing because it can affect long-term heart health.")
    if patient_inputs.get("family_history_diabetes") == "Yes" or patient_inputs.get("family_history_heart_disease") == "Yes":
        lines.append("Family history adds extra reason to keep an eye on these results.")
    return " ".join(lines)


def emotional_support_line(risk: Dict[str, Any]) -> str:
    levels = [
        str(risk.get("diabetes", {}).get("risk_level", "")).lower(),
        str(risk.get("heart", {}).get("risk_level", "")).lower(),
        str(risk.get("cooccurrence", {}).get("risk_level", "")).lower(),
    ]
    if "high" in levels or "moderate" in levels:
        return "Seeing concerning numbers can feel stressful. Taking time to understand them is already a positive step."
    return "These results look more reassuring right now, but regular follow-up still matters."


def build_patient_context(rec: Dict[str, Any], indicators: Dict[str, Any], risk: Dict[str, Any]) -> str:
    patient_inputs = rec.get("patient_inputs", {}) if rec else {}
    indicator_lines = []
    for name, item in (indicators or {}).items():
        indicator_lines.append(f"- {name}: value={item.get('value')}, unit={item.get('unit', '')}, flag={item.get('flag', '')}, reference={item.get('ref_raw', '')}")
    diabetes = risk.get("diabetes", {}) if risk else {}
    heart = risk.get("heart", {}) if risk else {}
    co = risk.get("cooccurrence", {}) if risk else {}
    return f"""
Patient profile:
- Age: {patient_inputs.get('age')}
- Sex: {patient_inputs.get('sex')}
- BMI: {patient_inputs.get('bmi')}
- BMI Category: {patient_inputs.get('bmi_category')}
- Smoker: {patient_inputs.get('smoker')}
- Family History Diabetes: {patient_inputs.get('family_history_diabetes')}
- Family History Heart Disease: {patient_inputs.get('family_history_heart_disease')}
- Symptoms: {patient_inputs.get('symptoms', [])}

Extracted indicators:
{chr(10).join(indicator_lines)}

Risk outputs:
- Diabetes Risk Level: {diabetes.get('risk_level')}
- Diabetes Confidence: {diabetes.get('confidence_score')}
- Diabetes Risk Score: {diabetes.get('risk_score')}
- Diabetes Reasons: {diabetes.get('reasons', [])}
- Heart Risk Level: {heart.get('risk_level')}
- Heart Confidence: {heart.get('confidence_score')}
- Heart Risk Score: {heart.get('risk_score')}
- Heart Reasons: {heart.get('reasons', [])}
- Combined Risk Level: {co.get('risk_level')}
- Combined Confidence: {co.get('confidence_score')}
- Combined Risk Score: {co.get('risk_score')}
- Combined Reasons: {co.get('reasons', [])}
""".strip()


def ask_llm_about_report(question: str, rec: Dict[str, Any], indicators: Dict[str, Any], risk: Dict[str, Any]) -> str:
    if client is None:
        return "AI chat is not configured yet. Add your API key in a .env file or Streamlit secrets."
    context = build_patient_context(rec, indicators, risk)
    system_prompt = """
You are a patient-friendly health report explainer inside a Clinical Decision Support System prototype.
Do not diagnose. Do not prescribe medication. Use cautious language.
Keep answers concise, calm, and practical.
"""
    user_prompt = f"""
Here is the patient's app context:

{context}

Patient question:
{question}

Answer in simple language for a non-clinical user.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.4,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI response failed: {e}"


def history_value(item: Dict[str, Any], field: str):
    if field in item and item.get(field) not in [None, "", "—"]:
        return item.get(field)
    record = item.get("record") or item
    patient_inputs = record.get("patient_inputs", {}) if isinstance(record, dict) else {}
    indicators = {}
    if isinstance(record, dict):
        indicators = record.get("processed_indicators", {}).get("disease_indicators", {}) or {}
    if field == "bmi":
        return patient_inputs.get("bmi")
    indicator_map = {
        "glucose": ["glucose"],
        "systolic_bp": ["systolic_bp"],
        "diastolic_bp": ["diastolic_bp"],
        "hba1c": ["hba1c"],
        "cholesterol_total": ["cholesterol_total", "total_cholesterol"],
        "ldl": ["ldl"],
        "hdl": ["hdl"],
        "triglycerides": ["triglycerides"],
    }
    for key in indicator_map.get(field, []):
        val = indicators.get(key, {}).get("value")
        if val not in [None, "", "—"]:
            return val
    return None


def history_label(item: Dict[str, Any], idx: int) -> str:
    ts = str(item.get("timestamp") or item.get("created_at") or item.get("saved_at") or "")
    if ts:
        ts = ts.replace("T", " ")
        return ts[:16]
    return f"Report {idx + 1}"


def compare_with_previous(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(history) < 2:
        return {}
    current = history[-1]
    previous = history[-2]

    def delta(key):
        curr = to_float(history_value(current, key))
        prev = to_float(history_value(previous, key))
        if curr is None or prev is None:
            return None
        return round(curr - prev, 1)

    return {
        "glucose_delta": delta("glucose"),
        "systolic_bp_delta": delta("systolic_bp"),
        "bmi_delta": delta("bmi"),
        "cholesterol_total_delta": delta("cholesterol_total"),
        "ldl_delta": delta("ldl"),
    }


def comparison_card(title: str, delta_value: Optional[float], better_when_lower: bool = True):
    if delta_value is None:
        text = "Need 2 reports"; color = "#64748b"; note = "Upload another report to compare changes."
    elif delta_value == 0:
        text = "No change"; color = "#64748b"; note = "Compared with previous report."
    else:
        improved = delta_value < 0 if better_when_lower else delta_value > 0
        color = "#10b981" if improved else "#ef4444"
        text = f"{'Improved' if improved else 'Worse'} by {abs(delta_value)}"
        note = "Compared with previous report."
    st.markdown(f"""
        <div class="hl-card">
            <div style="font-size:0.88rem;color:#64748b;margin-bottom:6px;">{title}</div>
            <div style="font-size:1.2rem;font-weight:800;color:{color};margin-bottom:4px;">{text}</div>
            <div class="hl-muted">{note}</div>
        </div>
    """, unsafe_allow_html=True)


def render_trend_chart(history: List[Dict[str, Any]], field: str, label: str):
    if not history:
        empty_state_card(f"{label} trend unavailable", "Upload reports to start seeing change over time.")
        return
    rows = []
    for idx, item in enumerate(history):
        value = to_float(history_value(item, field))
        if value is not None:
            rows.append({"x": history_label(item, idx), "value": value})
    if len(rows) < 2:
        empty_state_card(f"{label} trend unavailable", "Upload at least two reports with this value to unlock the trend.")
        return
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("x:N", title="Report", sort=None),
            y=alt.Y("value:Q", title=label),
            tooltip=[alt.Tooltip("x:N", title="Report"), alt.Tooltip("value:Q", title=label)]
        )
        .properties(height=250, background="white")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#334155", titleColor="#0f172a", gridColor="#e2e8f0")
    )
    st.altair_chart(chart, use_container_width=True)


def render_upload_steps():
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""<div class="hl-step"><div class="hl-step-num">1</div><div style="font-weight:800; margin-bottom:4px;">Add profile</div><div class="hl-muted">Enter age, body details, symptoms, and family history.</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""<div class="hl-step"><div class="hl-step-num">2</div><div style="font-weight:800; margin-bottom:4px;">Upload report</div><div class="hl-muted">Use a PDF, JPG, or PNG lab or screening report.</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""<div class="hl-step"><div class="hl-step-num">3</div><div style="font-weight:800; margin-bottom:4px;">See patient view</div><div class="hl-muted">Get a visual summary, priorities, trends, and doctor questions.</div></div>""", unsafe_allow_html=True)


def show_auth_screen():
    st.markdown("# HealthLens")
    st.caption("Bringing clarity to medical data")
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            st.subheader("Login")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login", use_container_width=True)
        if login_submit:
            ok, user_id = verify_user(u, p)
            if ok:
                st.session_state["logged_in"] = True
                st.session_state["user_id"] = user_id
                st.session_state["username"] = u.strip().lower()
                st.session_state["history"] = load_reports_for_user(user_id)
                reset_analysis_state()
                st.success("Logged in successfully.")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with register_tab:
        with st.form("register_form", clear_on_submit=False):
            st.subheader("Create account")
            u2 = st.text_input("New username")
            p2 = st.text_input("New password", type="password")
            p3 = st.text_input("Confirm password", type="password")
            register_submit = st.form_submit_button("Create account", use_container_width=True)
        if register_submit:
            u2 = u2.strip().lower()
            if len(u2) < 3:
                st.error("Username must be at least 3 characters.")
            elif len(p2) < 6:
                st.error("Password must be at least 6 characters.")
            elif p2 != p3:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(u2, p2)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


if not st.session_state["logged_in"]:
    show_auth_screen()
    st.stop()

if st.session_state["logged_in"] and not st.session_state["history"]:
    st.session_state["history"] = load_reports_for_user(st.session_state["user_id"])


with st.sidebar:
    st.markdown(f"### Welcome, {st.session_state['username']}")
    if st.button("Logout", use_container_width=True):
        logout(); st.rerun()
    st.markdown("---")
    st.markdown("### Quick View")
    hist = st.session_state["history"]
    st.write(f"Saved reports: **{len(hist)}**")
    if st.session_state["risk"]:
        st.write(f"Diabetes model risk: **{st.session_state['risk'].get('diabetes', {}).get('risk_level', '—')}**")
        st.write(f"Heart model risk: **{st.session_state['risk'].get('heart', {}).get('risk_level', '—')}**")
        st.write(f"Combined model risk: **{st.session_state['risk'].get('cooccurrence', {}).get('risk_level', '—')}**")
    st.markdown("---")
    if client:
        st.success("AI helper connected")
    else:
        st.warning("AI helper not configured")
    st.markdown("---")
    st.markdown("""<div class="hl-disclaimer"><strong>Medical disclaimer:</strong> HealthLens is informational only. It is not a diagnosis and does not replace a doctor or other healthcare professional.</div>""", unsafe_allow_html=True)

    with st.expander("Detailed Results", expanded=False):
        rec = st.session_state.record
        indicators = st.session_state.indicators
        risk = st.session_state.risk
        if rec is None:
            st.caption("No analysis yet.")
        else:
            st.write(f"**Status:** {rec.get('status', '')}")
            st.write(f"**Patient ID:** {rec.get('patient_id', '')}")
            pi = rec.get("patient_inputs", {})
            st.markdown("**Patient Inputs**")
            st.write(f"- Age: {pi.get('age', '')}")
            st.write(f"- BMI: {pi.get('bmi', '—')}")
            st.write(f"- BMI Category: {pi.get('bmi_category', '')}")
            st.write(f"- Smoker: {pi.get('smoker', '')}")
            if indicators:
                rows = [{"Indicator": name, "Value": v.get("value"), "Unit": v.get("unit", ""), "Flag": v.get("flag", "")} for name, v in indicators.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)
            if risk:
                st.write(f"- Diabetes model risk: {risk.get('diabetes', {}).get('risk_level', '')}")
                st.write(f"- Heart model risk: {risk.get('heart', {}).get('risk_level', '')}")
                st.write(f"- Combined model risk: {risk.get('cooccurrence', {}).get('risk_level', '')}")

    if st.session_state.get("dev_mode", False):
        with st.expander("Developer Tools", expanded=False):
            poppler_path = st.text_input("Poppler path (Windows only, optional)", value="", help="If pdf2image fails on Windows, install Poppler and paste its /bin path here.").strip() or None
            show_raw_text = st.checkbox("Show raw extracted text", value=False)
            raw_preview_chars = st.slider("Raw text preview length", 500, 12000, 4000, step=500)
    else:
        poppler_path = None
        show_raw_text = False
        raw_preview_chars = 4000


dashboard_tab, lifestyle_tab, upload_tab, ai_tab = st.tabs(["Dashboard", "Lifestyle", "Upload & Analyze", "Ask AI"])


with dashboard_tab:
    st.markdown("## Dashboard")
    rec = st.session_state.record
    indicators = st.session_state.indicators
    risk = st.session_state.risk
    history = st.session_state["history"]

    if rec is None or risk is None:
        st.markdown("""<div class="hl-hero"><div class="hl-hero-badge">HealthLens • Patient-friendly summary</div><div style="font-size:2rem; font-weight:800; margin-bottom:8px;">Understand the report in seconds</div><div style="font-size:1rem; max-width:760px; line-height:1.55;">HealthLens pulls key values from your report, explains what matters most, and turns a dense medical document into a simpler patient view.</div></div>""", unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1: compact_metric_card("What it finds", "Key values", "Glucose, HbA1c, BP, cholesterol", "#ef4444")
        with a2: compact_metric_card("What it explains", "Main concerns", "What matters now and why", "#3b82f6")
        with a3: compact_metric_card("What it adds", "Doctor prep", "Questions and next-step prompts", "#8b5cf6")
        b1, b2 = st.columns([1.2, 0.8])
        with b1: info_card("Why this feels different", "Instead of making you read the whole report line by line, HealthLens highlights what needs attention, what improved, and what to ask next.", "#2563eb", "✨")
        with b2: empty_state_card("Next step", "Go to Upload & Analyze, add a report, and the dashboard will turn into a visual patient summary.")
    else:
        overall_label, overall_note = overall_health_summary(risk)
        overall_style = status_style(overall_label)
        priorities = generate_priorities(risk, indicators)
        questions = generate_doctor_questions(risk, indicators)
        patient_inputs = rec.get("patient_inputs", {})
        plan = generate_lifestyle_recommendations(risk, patient_inputs, indicators)
        weekly_plan = plan["weekly_plan"]
        abnormal_count = count_abnormal_indicators(indicators)

        st.markdown(f"""<div class="hl-hero"><div class="hl-hero-badge">Current health summary</div><div style="display:flex; justify-content:space-between; gap:18px; align-items:flex-start;"><div style="max-width:760px;"><div style="font-size:2rem; font-weight:800; margin-bottom:6px;">{overall_label}</div><div style="font-size:1rem; line-height:1.55; margin-bottom:12px;">{overall_note}</div><div><span class="hl-pill">Diabetes model risk: {risk.get('diabetes', {}).get('risk_level', '—')}</span><span class="hl-pill">Heart model risk: {risk.get('heart', {}).get('risk_level', '—')}</span><span class="hl-pill">Combined model risk: {risk.get('cooccurrence', {}).get('risk_level', '—')}</span><span class="hl-pill">Abnormal indicators: {abnormal_count}</span><span class="hl-pill">BMI: {patient_inputs.get('bmi', '—')}</span></div></div><div style="min-width:132px; text-align:center;"><div style="width:98px; height:98px; margin:0 auto 8px auto; border-radius:50%; background:rgba(255,255,255,0.12); display:flex; align-items:center; justify-content:center; border:1px solid rgba(255,255,255,0.22);"><div style="width:20px; height:20px; border-radius:50%; background:{overall_style['dot']}; box-shadow:0 0 0 14px rgba(255,255,255,0.12);"></div></div><div style="font-size:0.84rem;">Current status</div></div></div></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="hl-disclaimer"><strong>Important:</strong> The circles below show model-based risk categories. The indicator cards show individual report values, so some lab values may be high even when a model risk circle is low.</div>""", unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        with r1: render_gauge_card("Diabetes Risk", str(risk.get("diabetes", {}).get("risk_level", "—")), "Based on extracted report values + model")
        with r2: render_gauge_card("Heart Risk", str(risk.get("heart", {}).get("risk_level", "—")), "Based on extracted report values + model")
        with r3: render_gauge_card("Combined Risk", str(risk.get("cooccurrence", {}).get("risk_level", "—")), "A combined view of both models")
        with r4: compact_metric_card("Abnormal indicators", str(abnormal_count), "How many values are outside normal range", "#f59e0b")

        st.markdown("### What this means")
        x1, x2, x3 = st.columns(3)
        with x1: info_card("What stands out", "Blood sugar appears to be the main concern right now, and blood pressure may also be worth reviewing.", "#2563eb", "🩺")
        with x2: info_card("Why it matters", "Changes in blood sugar and blood pressure can affect long-term health, but understanding them early gives you a better chance to act.", "#8b5cf6", "💬")
        with x3: info_card("What to do this week", "<br>".join([f"• {item['title']}" for item in weekly_plan[:3]]), "#ef4444", "🗓️")

        st.markdown("### Top priorities")
        pr1, pr2, pr3 = st.columns(3)
        items = priorities[:3] + ["Keep monitoring"] * (3 - len(priorities[:3]))
        with pr1: info_card("Priority 1", items[0], "#f59e0b", "1️⃣")
        with pr2: info_card("Priority 2", items[1], "#f59e0b", "2️⃣")
        with pr3: info_card("Priority 3", items[2], "#f59e0b", "3️⃣")

        st.markdown("### Key indicators")
        extract_key_indicator_cards(indicators)

        st.markdown("### Progress from the last report")
        comparison = compare_with_previous(history)
        c1, c2, c3 = st.columns(3)
        with c1: comparison_card("Glucose", comparison.get("glucose_delta"), True)
        with c2: comparison_card("Systolic BP", comparison.get("systolic_bp_delta"), True)
        with c3: comparison_card("BMI", comparison.get("bmi_delta"), True)

        st.markdown("### Trends")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Glucose trend**")
            render_trend_chart(history, "glucose", "Glucose")
        with t2:
            st.markdown("**Systolic BP trend**")
            render_trend_chart(history, "systolic_bp", "Systolic BP")

        t3, t4 = st.columns(2)
        with t3:
            st.markdown("**Cholesterol Total trend**")
            render_trend_chart(history, "cholesterol_total", "Cholesterol Total")
        with t4:
            st.markdown("**LDL trend**")
            render_trend_chart(history, "ldl", "LDL")

        st.markdown("### Doctor visit prep")
        d1, d2 = st.columns(2)
        with d1: info_card("Questions to ask", "<br>".join([f"• {q}" for q in questions]), "#7c3aed", "❓")
        with d2: info_card("Start this week", "<br>".join([f"• {item['title']}" for item in weekly_plan[:3]]), "#ef4444", "🗓️")

        if show_raw_text:
            raw = rec.get("raw_text", "")
            st.text_area("Raw text", raw[:raw_preview_chars], height=240)


with lifestyle_tab:
    st.markdown("## Lifestyle Recommendations")
    rec = st.session_state.record
    indicators = st.session_state.indicators
    risk = st.session_state.risk
    if rec is None or risk is None:
        empty_state_card("No personalized guidance yet", "Run an analysis first to unlock lifestyle suggestions tailored to the uploaded report.")
    else:
        plan = generate_lifestyle_recommendations(risk, rec.get("patient_inputs", {}), indicators)
        grouped = plan["grouped"]
        weekly_plan = plan["weekly_plan"]
        top_action = plan["top_action"]

        st.markdown(f"""
            <div class="hl-card" style="border-left:6px solid #2563eb; background:linear-gradient(135deg, #f8fbff, #ffffff); margin-bottom:14px;">
                <div style="font-size:0.78rem; color:#5b6f88; font-weight:700; margin-bottom:6px;">MAIN FOCUS RIGHT NOW</div>
                <div style="font-size:1.35rem; font-weight:800; color:#132238; margin-bottom:6px;">{top_action['title']}</div>
                <div style="color:#435269; margin-bottom:8px;">{top_action['why']}</div>
                <div>
                    <span class="hl-pill" style="background:#eff6ff; border-color:#bfdbfe; color:#1d4ed8 !important;">Priority 1</span>
                    <span class="hl-pill" style="background:#f8fafc; border-color:#cbd5e1; color:#475569 !important;">Difficulty: {top_action['difficulty']}</span>
                    <span class="hl-pill" style="background:#ecfdf5; border-color:#a7f3d0; color:#047857 !important;">Next report: {plan['next_report_days']} days</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        a1, a2, a3 = st.columns([1.1, 1.1, 0.8])
        with a1: compact_metric_card("Top priority", top_action["category"], "What to focus on first", "#2563eb")
        with a2: compact_metric_card("Weekly actions", str(len(weekly_plan)), "Small steps you can actually do", "#10b981")
        with a3: compact_metric_card("Follow-up window", f"{plan['next_report_days']} days", "When to check progress again", "#f59e0b")

        st.markdown("### Your action plan")
        cat_order = ["Food", "Activity", "Daily Habits", "Follow-Up"]
        cat_colors = {"Food": "#f97316", "Activity": "#10b981", "Daily Habits": "#6366f1", "Follow-Up": "#ef4444"}
        cat_icons = {"Food": "🍽️", "Activity": "🚶", "Daily Habits": "🧘", "Follow-Up": "📞"}

        for category in cat_order:
            items = grouped.get(category, [])
            if not items:
                continue

            body_parts = []
            for idx, item in enumerate(items[:2], start=1):
                part = (
                    f"<div style='margin-bottom:14px;'>"
                    f"<div style='font-weight:700; color:#132238; margin-bottom:4px;'>{idx}. {item['title']}</div>"
                    f"<div style='font-size:0.9rem; color:#516175; margin:3px 0 8px 0; line-height:1.45;'>{item['why']}</div>"
                    f"<span class='hl-pill' style='background:#f8fafc; border-color:#cbd5e1; color:#475569 !important;'>Difficulty: {item['difficulty']}</span>"
                    f"</div>"
                )
                body_parts.append(part)

            card_html = (
                f"<div class='hl-card' style='border-left:5px solid {cat_colors[category]}; min-height:220px; display:flex; justify-content:space-between; align-items:center; gap:18px;'>"
                f"<div style='flex:1;'>"
                f"<div style='font-weight:800; color:#132238; margin-bottom:10px; font-size:1.05rem;'>{cat_icons[category]} {category}</div>"
                f"<div style='color:#3f4f63; line-height:1.5; font-size:0.93rem;'>{''.join(body_parts)}</div>"
                f"</div>"
                f"<div style='min-width:120px; text-align:center; opacity:0.95;'>"
                f"<div style='font-size:4rem;'>{cat_icons[category]}</div>"
                f"</div>"
                f"</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("### What to do this week")
        checked_count = 0
        total_count = len(weekly_plan)
        for i, item in enumerate(weekly_plan):
            key = f"weekly_action_{i}_{item['title']}"
            checked = st.checkbox(f"{item['title']} — {item['difficulty']}", key=key)
            if checked:
                checked_count += 1
            st.caption(item["why"])

        progress = 0 if total_count == 0 else checked_count / total_count
        st.progress(progress)
        st.caption(f"Completed {checked_count} of {total_count} weekly actions")

        st.markdown("""
            <div class="hl-card" style="border-left:6px solid #8b5cf6; background:#faf5ff; margin-top:12px;">
                <div style="font-weight:800; color:#132238; margin-bottom:4px;">Why this page matters</div>
                <div style="color:#4c5d73;">The goal is not to give you a long list. The goal is to help you know what to do tomorrow morning, what matters most for your report, and when to check progress again.</div>
            </div>
        """, unsafe_allow_html=True)


with upload_tab:
    st.markdown("## Upload & Analyze")
    render_upload_steps()
    st.markdown("""<div class="hl-disclaimer"><strong>Before you continue:</strong> This prototype is for educational and informational use. It helps explain results, but it does not provide medical diagnosis or treatment.</div>""", unsafe_allow_html=True)
    pi = st.session_state.patient_inputs
    left, right = st.columns([1.15, 0.95], gap="large")

    with left:
        st.markdown("### 1) Patient Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            pi["age"] = st.number_input("Age", min_value=0, max_value=120, value=int(pi["age"]))
            pi["sex"] = st.selectbox("Sex", ["Prefer not to say", "Female", "Male", "Other"], index=["Prefer not to say", "Female", "Male", "Other"].index(pi["sex"]))
        with c2:
            pi["height_cm"] = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=float(pi["height_cm"]), step=0.5)
            pi["weight_kg"] = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=float(pi["weight_kg"]), step=0.5)
        with c3:
            pi["smoker"] = st.selectbox("Smoker", ["No", "Yes"], index=["No", "Yes"].index(pi["smoker"]))
            pi["family_history_diabetes"] = st.selectbox("Family History - Diabetes", ["No", "Yes"], index=["No", "Yes"].index(pi["family_history_diabetes"]))
            pi["family_history_heart_disease"] = st.selectbox("Family History - Heart Disease", ["No", "Yes"], index=["No", "Yes"].index(pi["family_history_heart_disease"]))

        st.markdown("### 2) Symptoms")
        symptom_options = ["Fatigue", "Frequent urination", "Increased thirst", "Blurred vision", "Chest pain", "Shortness of breath", "Dizziness", "Palpitations"]
        selected_symptoms = []
        sx_cols = st.columns(2)
        for i, symptom in enumerate(symptom_options):
            with sx_cols[i % 2]:
                checked = st.checkbox(symptom, value=symptom in pi["symptoms"], key=f"symptom_{symptom}")
                if checked:
                    selected_symptoms.append(symptom)
        pi["symptoms"] = selected_symptoms

        st.markdown("### 3) Heart inputs")
        h1, h2, h3 = st.columns(3)
        with h1:
            pi["chest_pain_type"] = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"], index=["ATA", "NAP", "ASY", "TA"].index(pi["chest_pain_type"]))
            pi["resting_ecg"] = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], index=["Normal", "ST", "LVH"].index(pi["resting_ecg"]))
        with h2:
            pi["exercise_angina"] = st.selectbox("Exercise Angina", ["N", "Y"], index=["N", "Y"].index(pi["exercise_angina"]))
            pi["st_slope"] = st.selectbox("ST Slope", ["Up", "Flat", "Down"], index=["Up", "Flat", "Down"].index(pi["st_slope"]))
        with h3:
            pi["max_hr"] = st.number_input("Max Heart Rate", min_value=40.0, max_value=250.0, value=float(pi["max_hr"]), step=1.0)
            pi["oldpeak"] = st.number_input("Oldpeak", min_value=-5.0, max_value=10.0, value=float(pi["oldpeak"]), step=0.1)

        bmi_value, bmi_category = compute_bmi(pi["height_cm"], pi["weight_kg"])
        b1, b2, b3 = st.columns(3)
        with b1: compact_metric_card("BMI", "—" if bmi_value is None else str(bmi_value), "Calculated automatically", "#14b8a6")
        with b2: compact_metric_card("Category", bmi_category, "Weight status", "#3b82f6")
        with b3: compact_metric_card("Quick view", "Needs focus" if bmi_category in ["Overweight", "Obese"] else "Okay", "General body-size signal", "#8b5cf6")
        pi["consent"] = st.checkbox("I understand this is informational and not medical advice.", value=bool(pi["consent"]))

    with right:
        st.markdown("### 4) Upload report")
        st.markdown("""<div class="hl-card" style="background:linear-gradient(135deg, #eef4ff, #ffffff); border-color:#d8e4ff; margin-bottom:12px;"><div style="font-weight:800; color:#1d4ed8;">Ready to upload</div><div class="hl-muted">Use a PDF, JPG, or PNG file. HealthLens will extract the relevant values and generate a patient-friendly view.</div></div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a PDF/JPG/PNG report", type=["pdf", "jpg", "jpeg", "png"], help="Examples: glucose, lipid profile, diabetes, or heart-related report.")
        if uploaded:
            st.success(f"Selected file: {uploaded.name}")
            st.caption(f"File type: {uploaded.type} | Size: {round(uploaded.size / 1024, 1)} KB")
        else:
            st.info("Choose a file to process.")

        st.markdown("### Analysis progress")
        rec = st.session_state.record
        if rec is None:
            st.progress(0)
            st.caption("Waiting for a report upload.")
        else:
            status = rec.get("status", "Unknown")
            progress = 100 if status == "Scored" else 70 if status == "Structured" else 40 if status == "Extracted" else 20
            st.progress(progress)
            st.caption(f"Latest run status: {status}")

        run_btn = st.button("Run Analysis", type="primary", use_container_width=True, disabled=(uploaded is None))

    if run_btn and uploaded is not None:
        suffix = "." + uploaded.name.split(".")[-1].lower()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"uploaded{suffix}"
            tmp_path.write_bytes(uploaded.getbuffer())
            with st.spinner("Stage 1: Extracting text..."):
                rec = run_extraction_pipeline(str(tmp_path), poppler_path=poppler_path)
            if rec.get("status") == "Failed":
                st.session_state.record = rec
                st.session_state.indicators = None
                st.session_state.risk = None
                st.error("Extraction failed.")
                st.code(rec.get("error", "Unknown error"))
            else:
                raw_text = rec.get("raw_text", "")
                with st.spinner("Stage 2: Parsing disease indicators..."):
                    indicators = extract_disease_indicators(raw_text)

                patient_info = parse_patient_info_from_text(raw_text)
                if patient_info.get("age") is not None:
                    pi["age"] = patient_info["age"]
                if patient_info.get("sex") is not None:
                    pi["sex"] = patient_info["sex"]

                bmi_value, bmi_category = compute_bmi(pi["height_cm"], pi["weight_kg"])
                glucose = to_float(get_indicator_value(indicators, "glucose"))
                diastolic_bp = to_float(get_indicator_value(indicators, "diastolic_bp"))
                systolic_bp = to_float(get_indicator_value(indicators, "systolic_bp"))
                cholesterol_total = to_float(get_indicator_value(indicators, "cholesterol_total", "total_cholesterol"))
                fasting_bs_raw = get_indicator_value(indicators, "fastingbs", "fasting_bs")
                chest_pain_type_raw = get_indicator_value(indicators, "chestpaintype", "chest_pain_type")
                resting_ecg_raw = get_indicator_value(indicators, "restingecg", "resting_ecg")
                max_hr_raw = get_indicator_value(indicators, "maxhr", "max_hr")
                exercise_angina_raw = get_indicator_value(indicators, "exerciseangina", "exercise_angina")
                oldpeak_raw = get_indicator_value(indicators, "oldpeak")
                st_slope_raw = get_indicator_value(indicators, "stslope", "st_slope")

                fasting_bs = int(to_float(fasting_bs_raw, 1 if glucose is not None and glucose >= 120 else 0))
                chest_pain_type = normalize_category(chest_pain_type_raw, ["ATA", "NAP", "ASY", "TA"], pi["chest_pain_type"])
                resting_ecg = normalize_category(resting_ecg_raw, ["Normal", "ST", "LVH"], pi["resting_ecg"])
                max_hr = to_float(max_hr_raw, pi["max_hr"])
                exercise_angina = normalize_category(exercise_angina_raw, ["N", "Y"], pi["exercise_angina"])
                oldpeak = to_float(oldpeak_raw, pi["oldpeak"])
                st_slope = normalize_category(st_slope_raw, ["Up", "Flat", "Down"], pi["st_slope"])

                ml_inputs = {
                    "glucose": glucose,
                    "bmi": bmi_value,
                    "blood_pressure": diastolic_bp,
                    "age": pi["age"],
                    "sex": map_sex_for_model(pi["sex"]) or "M",
                    "chest_pain_type": chest_pain_type,
                    "resting_bp": systolic_bp,
                    "cholesterol": cholesterol_total,
                    "fasting_bs": fasting_bs,
                    "resting_ecg": resting_ecg,
                    "max_hr": max_hr,
                    "exercise_angina": exercise_angina,
                    "oldpeak": oldpeak,
                    "st_slope": st_slope,
                }

                with st.spinner("Stage 3: Running ML risk models..."):
                    risk = compute_combined_ml_risk(ml_inputs)

                rec["patient_inputs"] = {
                    "age": pi["age"],
                    "sex": pi["sex"],
                    "height_cm": pi["height_cm"],
                    "weight_kg": pi["weight_kg"],
                    "bmi": bmi_value,
                    "bmi_category": bmi_category,
                    "smoker": pi["smoker"],
                    "family_history_diabetes": pi["family_history_diabetes"],
                    "family_history_heart_disease": pi["family_history_heart_disease"],
                    "symptoms": pi["symptoms"],
                    "consent": pi["consent"],
                    "chest_pain_type": chest_pain_type,
                    "resting_ecg": resting_ecg,
                    "exercise_angina": exercise_angina,
                    "st_slope": st_slope,
                    "max_hr": max_hr,
                    "oldpeak": oldpeak,
                }
                rec.setdefault("processed_indicators", {})
                rec["processed_indicators"]["disease_indicators"] = indicators
                rec["risk_result"] = risk
                rec["model_used"] = {"diabetes": "rf_reduced_4_tuned.pkl", "heart": "best_heart_model_pipeline.pkl", "cooccurrence": "fusion_logic"}
                rec["status"] = "Scored"
                st.session_state.record = rec
                st.session_state.indicators = indicators
                st.session_state.risk = risk
                st.session_state["chat_history"] = []
                save_report(st.session_state["user_id"], rec, indicators, risk)
                st.session_state["history"] = load_reports_for_user(st.session_state["user_id"])
                st.success("Analysis complete. Open the Dashboard to see the visual patient summary.")


with ai_tab:
    st.markdown("## Ask AI")
    if st.session_state.record is None or st.session_state.indicators is None or st.session_state.risk is None:
        empty_state_card("No report loaded yet", "Run an analysis first, then come back here to ask questions about the current report.")
    else:
        st.markdown("""
            <div class="hl-card" style="border-left:6px solid #7c3aed; background:linear-gradient(135deg, #faf5ff, #ffffff); margin-bottom:14px;">
                <div style="font-size:0.78rem; color:#6b5b95; font-weight:700; margin-bottom:6px;">AI REPORT ASSISTANT</div>
                <div style="font-size:1.2rem; font-weight:800; color:#132238; margin-bottom:6px;">Ask questions in simple language</div>
                <div style="color:#475569;">Use this page to understand your report, what matters most, and what to ask your doctor next. This assistant is informational only and does not diagnose disease.</div>
            </div>
        """, unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        with q1: compact_metric_card("Ask about", "Blood sugar", "Glucose, HbA1c, trends", "#7c3aed")
        with q2: compact_metric_card("Ask about", "Heart health", "BP, cholesterol, heart risk", "#2563eb")
        with q3: compact_metric_card("Ask about", "Next steps", "Lifestyle and follow-up", "#10b981")
        quick_questions = [
            "What does my glucose mean?",
            "Should I worry about this result?",
            "What should I do next?",
            "Explain my heart risk simply",
            "Explain my combined risk simply",
            "What should I ask my doctor?",
            "Which result matters most right now?",
        ]
        left, right = st.columns([1.05, 0.95], gap="large")
        with left:
            quick_q = st.selectbox("Quick question", quick_questions, key="ai_tab_quick_q")
            custom_q = st.text_input("Type your question", placeholder="Ask in simple words", key="ai_tab_custom_q")
            a1, a2 = st.columns(2)
            with a1:
                if st.button("Ask AI", key="ai_tab_send", use_container_width=True):
                    question = custom_q.strip() if custom_q.strip() else quick_q
                    st.session_state["chat_history"].append({"role": "user", "content": question})
                    answer = ask_llm_about_report(question, st.session_state.record, st.session_state.indicators, st.session_state.risk)
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                    st.rerun()
            with a2:
                if st.button("Clear chat", key="ai_tab_clear", use_container_width=True):
                    st.session_state["chat_history"] = []
                    st.rerun()
            st.markdown("### Suggested prompts")
            st.markdown("<br>".join([
                "• Explain my results like I have no medical background.",
                "• Tell me which part of my report matters most.",
                "• Give me 3 questions to ask my doctor.",
                "• What should I change this week based on this report?",
            ]), unsafe_allow_html=True)
        with right:
            st.markdown("### Conversation")
            if st.session_state["chat_history"]:
                for msg in st.session_state["chat_history"]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
            else:
                empty_state_card("No questions yet", "Start with a quick question or type your own question to get a patient-friendly explanation.")
        st.markdown("""<div class="hl-disclaimer" style="margin-top:14px;"><strong>Reminder:</strong> AI responses here are for explanation and support only. They are not a substitute for a doctor, diagnosis, or emergency care.</div>""", unsafe_allow_html=True)
