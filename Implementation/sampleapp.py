import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from disease_pipeline import run_extraction_pipeline, extract_disease_indicators
from diabetes_model_inference import predict_diabetes_risk
from disease_risk_engine import compute_diabetes_heart_risk
from authdb import init_db, create_user, verify_user


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="CDSS Patient Health Portal",
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

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
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
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()



def logout():
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None
    st.session_state["username"] = None
    st.session_state["record"] = None
    st.session_state["indicators"] = None
    st.session_state["risk"] = None
    st.session_state["chat_history"] = []
    st.session_state["history"] = []

# ---------------------------------------------------
# STYLING
# ---------------------------------------------------
def apply_custom_css():
    st.markdown(
        """
        <style>
            .stApp {
                background: #f7f9fc;
                color: #0f172a;
            }

            .main .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1250px;
            }

            header[data-testid="stHeader"] {
                background: transparent;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            section[data-testid="stSidebar"] {
                background: #eef4f8;
                border-right: 1px solid #d9e4ec;
            }

            h1, h2, h3, h4 {
                color: #0f172a !important;
                font-weight: 800 !important;
            }

            p, label {
                color: #334155 !important;
            }

            .stButton > button {
                background: #eb6757 !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                font-weight: 700 !important;
                box-shadow: 0 4px 12px rgba(235, 103, 87, 0.18);
            }

            .stButton > button:hover {
                background: #de5747 !important;
                color: white !important;
            }

            section[data-testid="stSidebar"] .stButton > button {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #d7e2ea !important;
                box-shadow: none !important;
            }

            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #dbe7f0;
                border-radius: 16px;
                padding: 12px;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            }

            div[data-testid="stAlert"] {
                border-radius: 14px !important;
            }

            .mini-pill {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 700;
                margin-right: 8px;
                margin-bottom: 8px;
                border: 1px solid #d7e2ea;
                background: #ffffff;
                color: #334155;
            }

            .small-muted {
                color: #64748b !important;
                font-size: 0.92rem;
            }

            div[data-testid="stCheckbox"] label {
                color: #0f172a !important;
                font-weight: 500 !important;
            }

            div[data-testid="stCheckbox"] p {
                color: #0f172a !important;
                font-size: 0.95rem !important;
            }

            div[data-testid="stChatMessage"] {
                background: #ffffff;
                border: 1px solid #dbe7f0;
                border-radius: 16px;
                padding: 10px 12px;
            }

            button[data-baseweb="tab"] {
                color: #334155 !important;
                font-weight: 600 !important;
            }

            button[data-baseweb="tab"][aria-selected="true"] {
                color: #dc2626 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_css()



# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
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


def get_status_color(level: str) -> str:
    level = (level or "").lower()
    if "high" in level or "attention" in level:
        return "#ef4444"
    if "moderate" in level or "monitor" in level:
        return "#f59e0b"
    if "low" in level or "stable" in level:
        return "#10b981"
    return "#64748b"


def metric_card(title: str, value: str, subtitle: str = "", color: str = "#2563eb", icon: str = "📌"):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}14, #ffffff);
            border: 1px solid {color}28;
            border-radius: 18px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            min-height: 118px;
        ">
            <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 8px;">{icon} {title}</div>
            <div style="font-size: 1.9rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">{value}</div>
            <div style="font-size: 0.88rem; color: #64748b;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str, color: str = "#0ea5e9", icon: str = "ℹ️"):
    st.markdown(
        f"""
        <div style="
            background: #ffffff;
            border-left: 6px solid {color};
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 12px;
        ">
            <div style="font-weight: 800; color: #0f172a; margin-bottom: 6px;">{icon} {title}</div>
            <div style="color: #334155; line-height: 1.55;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compact_pills(items: List[str]):
    if not items:
        return
    html = "".join([f'<span class="mini-pill">{item}</span>' for item in items])
    st.markdown(html, unsafe_allow_html=True)


def extract_key_indicator_cards(indicators: Dict[str, Dict[str, Any]]):
    if not indicators:
        st.info("No disease indicators detected yet.")
        return

    priority_order = [
        "glucose",
        "hba1c",
        "systolic_bp",
        "diastolic_bp",
        "cholesterol_total",
        "ldl",
        "hdl",
        "triglycerides",
    ]

    available = [k for k in priority_order if k in indicators]
    fallback = [k for k in indicators.keys() if k not in available]
    selected = (available + fallback)[:4]

    cols = st.columns(len(selected))
    for col, key in zip(cols, selected):
        item = indicators.get(key, {})
        flag = str(item.get("flag", "") or "Normal")
        value = item.get("value", "—")
        unit = item.get("unit", "")
        ref = item.get("ref_raw", "")
        flag_color = (
            "#ef4444" if flag.lower() == "high"
            else "#f59e0b" if flag.lower() not in ["normal", ""]
            else "#10b981"
        )

        with col:
            st.markdown(
                f"""
                <div style="
                    background:#ffffff;
                    border:1px solid #dbe7f0;
                    border-radius:18px;
                    padding:18px;
                    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
                    min-height: 160px;
                ">
                    <div style="font-size:0.92rem; color:#64748b; margin-bottom:10px;">🧪 {key.replace('_', ' ').title()}</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#0f172a;">{value} {unit}</div>
                    <div style="margin:12px 0;">
                        <span style="
                            background:{flag_color}15;
                            color:{flag_color};
                            border:1px solid {flag_color}35;
                            border-radius:999px;
                            padding:6px 12px;
                            font-size:0.8rem;
                            font-weight:700;
                        ">{flag}</span>
                    </div>
                    <div style="font-size:0.82rem; color:#64748b;">Reference: {ref if ref else "Not available"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def generate_lifestyle_recommendations(
    risk: Dict[str, Any],
    patient_inputs: Dict[str, Any],
    indicators: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    diabetes = risk.get("diabetes", {})
    heart = risk.get("heart", {})
    bmi = patient_inputs.get("bmi")
    smoker = patient_inputs.get("smoker", "No")
    symptoms = patient_inputs.get("symptoms", [])

    recs = {
        "Food": [],
        "Activity": [],
        "Daily Habits": [],
        "Follow-Up": [],
    }

    if bmi is not None and bmi >= 25:
        recs["Food"].append("Reduce sugary drinks and processed snacks.")
        recs["Activity"].append("Aim for 30 minutes of walking most days.")

    if diabetes.get("risk_level", "").lower() in ["moderate", "high"]:
        recs["Food"].append("Choose higher-fiber meals and steadier meal timing.")
        recs["Follow-Up"].append("Discuss follow-up blood sugar testing.")

    if heart.get("risk_level", "").lower() in ["moderate", "high"]:
        recs["Food"].append("Limit salty and heavily processed foods.")
        recs["Daily Habits"].append("Track blood pressure regularly.")

    if smoker == "Yes":
        recs["Daily Habits"].append("Cutting down smoking would strongly help heart health.")

    if "Chest pain" in symptoms or "Shortness of breath" in symptoms:
        recs["Follow-Up"].append("Discuss chest pain or breathing symptoms promptly.")

    glucose_flag = str(indicators.get("glucose", {}).get("flag", "")).lower()
    if glucose_flag == "high":
        recs["Food"].append("Reduce added sugar and watch portions.")

    if patient_inputs.get("family_history_diabetes") == "Yes":
        recs["Follow-Up"].append("Family history makes regular diabetes screening important.")

    if patient_inputs.get("family_history_heart_disease") == "Yes":
        recs["Follow-Up"].append("Family history makes heart check-ups more important.")

    if not recs["Activity"]:
        recs["Activity"].append("Stay active with walking, stretching, or moderate exercise.")
    if not recs["Daily Habits"]:
        recs["Daily Habits"].append("Prioritize sleep, hydration, and stress management.")
    if not recs["Follow-Up"]:
        recs["Follow-Up"].append("Use this report as a discussion aid, not a diagnosis.")

    return recs


def generate_priorities(risk: Dict[str, Any], indicators: Dict[str, Dict[str, Any]]) -> List[str]:
    priorities = []

    diabetes_level = str(risk.get("diabetes", {}).get("risk_level", "")).lower()
    heart_level = str(risk.get("heart", {}).get("risk_level", "")).lower()

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

    if str(indicators.get("glucose", {}).get("flag", "")).lower() == "high":
        questions.append("Could this suggest prediabetes or diabetes risk?")

    if str(indicators.get("cholesterol_total", {}).get("flag", "")).lower() == "high" or str(indicators.get("ldl", {}).get("flag", "")).lower() == "high":
        questions.append("What lifestyle changes should I start with?")

    if not questions:
        questions.append("Are any follow-up tests needed based on this report?")

    return questions[:3]


def overall_health_summary(risk: Dict[str, Any]) -> Tuple[str, str]:
    diabetes_level = (risk.get("diabetes", {}).get("risk_level", "") or "").lower()
    heart_level = (risk.get("heart", {}).get("risk_level", "") or "").lower()

    if "high" in [diabetes_level, heart_level]:
        return "Needs Attention", "Some findings may need closer follow-up."
    if "moderate" in [diabetes_level, heart_level]:
        return "Monitor Closely", "A few indicators deserve attention."
    if "low" in [diabetes_level, heart_level]:
        return "Generally Stable", "No major risk signal appears dominant."
    return "No Analysis Yet", "Upload a report and run the pipeline."


def patient_friendly_explanation(risk: Dict[str, Any]) -> str:
    diabetes = risk.get("diabetes", {})
    heart = risk.get("heart", {})
    d_level = diabetes.get("risk_level", "unknown")
    h_level = heart.get("risk_level", "unknown")

    return (
        f"Your report suggests {str(d_level).lower()} diabetes risk and "
        f"{str(h_level).lower()} heart risk. This is an informational summary."
    )


def build_patient_context(rec: Dict[str, Any], indicators: Dict[str, Any], risk: Dict[str, Any]) -> str:
    patient_inputs = rec.get("patient_inputs", {}) if rec else {}

    indicator_lines = []
    for name, item in (indicators or {}).items():
        indicator_lines.append(
            f"- {name}: value={item.get('value')}, unit={item.get('unit', '')}, "
            f"flag={item.get('flag', '')}, reference={item.get('ref_raw', '')}"
        )

    diabetes = risk.get("diabetes", {}) if risk else {}
    heart = risk.get("heart", {}) if risk else {}

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
""".strip()


def ask_llm_about_report(
    question: str,
    rec: Dict[str, Any],
    indicators: Dict[str, Any],
    risk: Dict[str, Any],
) -> str:
    if client is None:
        return (
            "AI chat is not configured yet. Add your API key in a .env file or Streamlit secrets. "
            "The rest of the app will still work normally."
        )

    context = build_patient_context(rec, indicators, risk)

    system_prompt = """
You are a patient-friendly health report explainer inside a Clinical Decision Support System prototype.

Your job:
- Explain results in simple, calm, clear language.
- Help the patient understand indicators, risk summaries, and lifestyle guidance.
- Keep answers concise, practical, and non-technical when possible.

Rules:
- Do not diagnose.
- Do not say the patient definitely has a disease.
- Do not prescribe medication.
- Do not replace a doctor.
- Use cautious language such as: "may indicate", "can be associated with", "suggests", "may deserve follow-up".
- Avoid definitive claims.
- If serious symptoms appear such as chest pain, shortness of breath, or severe dizziness, advise prompt medical attention.
- Mention that the app is informational and non-diagnostic when relevant.
- Focus on: what it means, why it matters, what to do next.
- Use short paragraphs or bullets.
- Keep answers shorter than 120 words unless the user asks for more detail.
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
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI response failed: {e}"


def render_sidebar_export():
    rec = st.session_state.record
    if rec is None:
        st.caption("No structured output available yet.")
        return

    st.caption("Available for project/demo use.")
    json_bytes = json.dumps(rec, indent=2).encode("utf-8")
    st.download_button(
        "Download structured JSON",
        data=json_bytes,
        file_name=f"{rec.get('patient_id', 'record')}_record.json",
        mime="application/json",
        use_container_width=True,
    )
# -------------------------
# TAB 1: Upload
# -------------------------
with tab_upload:
    st.subheader("1) Patient Inputs (Structured)")

    pi = st.session_state.patient_inputs

    c1, c2, c3 = st.columns(3)

    with c1:
        pi["age"] = st.number_input("Age", min_value=0, max_value=120, value=int(pi["age"]))
        pi["sex"] = st.selectbox(
            "Sex",
            ["Prefer not to say", "Female", "Male", "Other"],
            index=["Prefer not to say", "Female", "Male", "Other"].index(pi["sex"]),
        )

    with c2:
        pi["height_cm"] = st.number_input(
            "Height (cm)", min_value=50.0, max_value=250.0, value=float(pi["height_cm"]), step=0.5
        )
        pi["weight_kg"] = st.number_input(
            "Weight (kg)", min_value=10.0, max_value=300.0, value=float(pi["weight_kg"]), step=0.5
        )

    with c3:
        pi["smoker"] = st.selectbox("Smoker", ["No", "Yes"], index=["No", "Yes"].index(pi["smoker"]))
        pi["family_history_diabetes"] = st.selectbox(
            "Family History - Diabetes",
            ["No", "Yes"],
            index=["No", "Yes"].index(pi["family_history_diabetes"]),
        )
        pi["family_history_heart_disease"] = st.selectbox(
            "Family History - Heart Disease",
            ["No", "Yes"],
            index=["No", "Yes"].index(pi["family_history_heart_disease"]),
        )

    pi["symptoms"] = st.multiselect(
        "Symptoms (optional)",
        [
            "Fatigue",
            "Frequent urination",
            "Increased thirst",
            "Blurred vision",
            "Chest pain",
            "Shortness of breath",
            "Dizziness",
            "Palpitations",
            "Other",
        ],
        default=pi["symptoms"],
    )

    bmi_value, bmi_category = compute_bmi(pi["height_cm"], pi["weight_kg"])
    st.metric("BMI (calculated)", "—" if bmi_value is None else str(bmi_value))
    st.caption(f"BMI category: {bmi_category}")

    pi["consent"] = st.checkbox(
        "I understand this is a prototype and not medical advice.",
        value=bool(pi["consent"])
    )

    st.markdown("---")
    st.subheader("2) Upload Medical Report")

    uploaded = st.file_uploader(
        "Upload a PDF/JPG/PNG (e.g., glucose, lipid profile, diabetes or heart-related report)",
        type=["pdf", "jpg", "jpeg", "png"],
        help="File will be processed locally by the pipeline.",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if uploaded:
            st.success(f"Selected file: {uploaded.name}")
            st.json(
                {
                    "filename": uploaded.name,
                    "type": uploaded.type,
                    "size_bytes": uploaded.size,
                    "selected_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        else:
            st.info("Choose a file to run extraction.")

    with col2:
        st.markdown("### Pipeline Status")
        rec = st.session_state.record
        if rec is None:
            st.progress(10)
            st.caption("Waiting for upload")
        else:
            status = rec.get("status", "Unknown")
            if status == "Extracted":
                st.progress(40)
            elif status == "Structured":
                st.progress(70)
            elif status == "Scored":
                st.progress(100)
            else:
                st.progress(60)
            st.caption(f"Last run: {status}")

    run_btn = st.button("Run Pipeline", type="primary", disabled=(uploaded is None))

    if run_btn and uploaded is not None:
        suffix = "." + uploaded.name.split(".")[-1].lower()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"uploaded{suffix}"
            tmp_path.write_bytes(uploaded.getbuffer())

            with st.spinner("Stage 1: Extracting text (PDF text → OCR fallback)..."):
                rec = run_extraction_pipeline(str(tmp_path), poppler_path=poppler_path)

            if rec.get("status") == "Failed":
                st.session_state.record = rec
                st.session_state.indicators = None
                st.session_state.risk = None
                st.error("Extraction failed.")
                st.code(rec.get("error", "Unknown error"))
            else:
                with st.spinner("Stage 2: Parsing diabetes and heart disease indicators..."):
                    indicators = extract_disease_indicators(rec.get("raw_text", ""))

                # Diabetes ML input mapping
                glucose = indicators.get("glucose", {}).get("value")

                # For Pima-style diabetes data, BloodPressure is closer to diastolic BP
                blood_pressure = indicators.get("diastolic_bp", {}).get("value")

                with st.spinner("Stage 3A: Running diabetes ML model..."):
                    diabetes_result = predict_diabetes_risk(
                        glucose=glucose,
                        bmi=bmi_value,
                        blood_pressure=blood_pressure,
                        age=pi["age"],
                    )

                with st.spinner("Stage 3B: Computing heart disease rule-based estimate..."):
                    rule_based_result = compute_diabetes_heart_risk(
                        indicators,
                        {
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
                        }
                    )

                risk = {
                    "diabetes": diabetes_result,
                    "heart": rule_based_result.get("heart", {}),
                    "general_notes": [
                        "Diabetes risk is generated using a trained machine learning model.",
                        "Heart disease risk is currently generated using a rule-based approach.",
                        "These outputs are informational and non-diagnostic.",
                        "Discuss concerning results with a qualified healthcare professional.",
                    ],
                }

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
                }

                rec.setdefault("processed_indicators", {})
                rec["processed_indicators"]["disease_indicators"] = indicators
                rec["risk_result"] = risk
                rec["model_used"] = {
                    "diabetes": "rf_reduced_4_tuned.pkl",
                    "heart": "rule_based_engine",
                }
                rec["status"] = "Scored"

                st.session_state.record = rec
                st.session_state.indicators = indicators
                st.session_state.risk = risk

                st.success("Done! Go to the Results tab to view structured indicators and risk summary.")

# -------------------------
# TAB 2: Results
# -------------------------
with tab_results:
    st.subheader("Results")

    rec = st.session_state.record
    indicators = st.session_state.indicators
    risk = st.session_state.risk

    if rec is None:
        st.info("No run yet. Go to Upload → fill inputs → upload a file → click Run.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Status", rec.get("status", ""))
        m2.metric("Patient ID", rec.get("patient_id", ""))
        m3.metric("Extracted chars", str(len(rec.get("raw_text", ""))))

        st.markdown("### Patient Inputs")
        pi = rec.get("patient_inputs", {})
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Age", str(pi.get("age", "")))
        p2.metric("BMI", "—" if pi.get("bmi") is None else str(pi.get("bmi")))
        p3.metric("BMI Category", str(pi.get("bmi_category", "")))
        p4.metric("Smoker", str(pi.get("smoker", "")))

        st.markdown("### Extracted Diabetes / Heart Indicators")
        if not indicators:
            st.warning("No diabetes or heart-related indicators detected.")
        else:
            rows = []
            for name, v in indicators.items():
                rows.append(
                    {
                        "Indicator": name,
                        "Value": v.get("value"),
                        "Unit": v.get("unit", ""),
                        "Reference": v.get("ref_raw", ""),
                        "Flag": v.get("flag", ""),
                    }
                )
            st.table(sorted(rows, key=lambda r: r["Indicator"]))

        st.markdown("### Risk Summary")

        if not risk:
            st.info("Risk result not available yet.")
        else:
            st.markdown("#### Diabetes Risk (ML Model)")
            d1, d2, d3 = st.columns(3)
            d1.metric("Risk Level", risk.get("diabetes", {}).get("risk_level", ""))
            d2.metric("Confidence Score", str(risk.get("diabetes", {}).get("confidence_score", "")))
            d3.metric("Risk Score", str(risk.get("diabetes", {}).get("risk_score", "")))

            st.markdown("**Contributing Indicators - Diabetes**")
            for item in risk.get("diabetes", {}).get("reasons", []):
                st.write(f"- {item}")

            st.markdown("#### Heart Disease Risk (Rule-Based)")
            h1, h2, h3 = st.columns(3)
            h1.metric("Risk Level", risk.get("heart", {}).get("risk_level", ""))
            h2.metric("Confidence Score", str(risk.get("heart", {}).get("confidence_score", "")))
            h3.metric("Risk Score", str(risk.get("heart", {}).get("risk_score", "")))

            st.markdown("**Contributing Indicators - Heart Disease**")
            for item in risk.get("heart", {}).get("reasons", []):
                st.write(f"- {item}")

            with st.expander("Recommendations / Notes"):
                for tip in risk.get("general_notes", []):
                    st.write(f"- {tip}")

            st.warning(
                "These outputs are informational and non-diagnostic. They are intended to support understanding "
                "of possible risk indicators and should not replace professional medical evaluation."
            )

        if show_raw_text:
            st.markdown("### Raw Extracted Text (Preview)")
            raw = rec.get("raw_text", "")
            st.text_area("Raw text", raw[:raw_preview_chars], height=260)

# -------------------------
# TAB 3: Export
# -------------------------
with tab_export:
    st.subheader("Export JSON")

    rec = st.session_state.record
    if rec is None:
        st.info("Nothing to export yet. Run the pipeline first.")
    else:
        st.caption("This JSON is useful evidence for your progress report.")
        json_bytes = json.dumps(rec, indent=2).encode("utf-8")

        st.download_button(
            "Download structured JSON",
            data=json_bytes,
            file_name=f"{rec.get('patient_id', 'record')}_record.json",
            mime="application/json",
        )

        st.markdown("### JSON Preview")
        st.code(json.dumps(rec, indent=2), language="json")
