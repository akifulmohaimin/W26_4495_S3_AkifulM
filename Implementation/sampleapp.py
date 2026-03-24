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
# FILE-BASED HISTORY
# ---------------------------------------------------
HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)


def get_history_path(username: str) -> Path:
    safe_name = username.replace(" ", "_").replace("/", "_")
    return HISTORY_DIR / f"{safe_name}_history.json"


def load_user_history(username: str) -> List[Dict[str, Any]]:
    path = get_history_path(username)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_user_history(username: str, history: List[Dict[str, Any]]) -> None:
    path = get_history_path(username)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")



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
def render_trend_chart(history: List[Dict[str, Any]], field: str, label: str):
    if not history:
        st.info(f"No data yet for {label}.")
        return

    rows = []
    for item in history:
        value = to_float(item.get(field))
        if value is not None:
            rows.append(
                {
                    "date": item.get("timestamp", "")[:10],
                    "value": value,
                }
            )

    if not rows:
        st.info(f"No valid values yet for {label}.")
        return

    df = pd.DataFrame(rows)
    df = df.groupby("date", as_index=False).last()

    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("date:N", title="Date"),
            y=alt.Y("value:Q", title=label),
            tooltip=["date", "value"],
        )
        .properties(
            height=220,
            background="white",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#334155",
            titleColor="#0f172a",
            gridColor="#e2e8f0",
        )
    )

    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------
# AUTH
# ---------------------------------------------------
st.markdown("## CDSS — Patient Health Portal")
st.caption("Understand your report, key indicators, trends, and next steps in one place.")

if not st.session_state["logged_in"]:
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.subheader("Login")
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", use_container_width=True):
            ok, user_id = verify_user(u, p)
            if ok:
                st.session_state["logged_in"] = True
                st.session_state["user_id"] = user_id
                st.session_state["username"] = u.strip().lower()
                st.session_state["history"] = load_user_history(st.session_state["username"])
                st.success("Logged in successfully.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        st.subheader("Create account")
        u2 = st.text_input("New username", key="reg_user")
        p2 = st.text_input("New password", type="password", key="reg_pass")

        if st.button("Create account", use_container_width=True):
            ok, msg = create_user(u2, p2)
            (st.success if ok else st.error)(msg)

    st.stop()


if st.session_state["logged_in"] and not st.session_state["history"]:
    st.session_state["history"] = load_user_history(st.session_state["username"])


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown(f"### 👋 Welcome, {st.session_state['username']}")

    if st.button("Logout", use_container_width=True):
        logout()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 AI Status")
    if client:
        st.success("AI chat connected")
    else:
        st.warning("AI chat not configured")

    st.markdown("---")
    st.markdown("### 📌 Quick View")
    hist = st.session_state["history"]
    st.write(f"Past reports saved: **{len(hist)}**")
    if st.session_state["risk"]:
        st.write(f"Latest diabetes risk: **{st.session_state['risk'].get('diabetes', {}).get('risk_level', '—')}**")
        st.write(f"Latest heart risk: **{st.session_state['risk'].get('heart', {}).get('risk_level', '—')}**")

    st.markdown("---")
    st.markdown("### ⚙️ App Settings")
    poppler_path = st.text_input(
        "Poppler path (Windows only, optional)",
        value="",
        help="If pdf2image fails on Windows, install Poppler and paste its /bin path here.",
    ).strip() or None

    show_raw_text = st.checkbox("Show raw extracted text", value=False)
    raw_preview_chars = st.slider("Raw text preview length", 500, 12000, 4000, step=500)

    st.markdown("---")
    with st.expander("Disclaimer", expanded=False):
        st.warning(
            "This prototype provides informational and non-diagnostic outputs only. "
            "It should not be used as medical advice or as a replacement for professional evaluation."
        )

    with st.expander("Developer Tools", expanded=False):
        render_sidebar_export()


# ---------------------------------------------------
# NAVIGATION
# ---------------------------------------------------
home_tab, upload_tab, results_tab = st.tabs(
    ["Home Dashboard", "Upload & Analyze", "Detailed Results"]
)


# ---------------------------------------------------
# HOME DASHBOARD
# ---------------------------------------------------
with home_tab:
    st.markdown("## 🏥 Health Dashboard")

    rec = st.session_state.record
    indicators = st.session_state.indicators
    risk = st.session_state.risk
    history = st.session_state["history"]

    if rec is None or risk is None:
        st.info("No analysis yet. Go to Upload & Analyze to process a patient report.")

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Overall Status", "No Analysis Yet", "Upload a report", "#64748b", "📄")
        with c2:
            metric_card("Patient View", "Friendly", "Clear and simple", "#0ea5e9", "✨")
        with c3:
            metric_card("AI Help", "Ready", "Ask after analysis", "#8b5cf6", "🤖")
    else:
        overall_label, overall_note = overall_health_summary(risk)
        overall_color = get_status_color(overall_label)

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {overall_color}14, #ffffff);
                border: 1px solid {overall_color}26;
                border-radius: 22px;
                padding: 22px;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
                margin-bottom: 16px;
            ">
                <div style="font-size: 0.92rem; color: #64748b; margin-bottom: 6px;">🩺 Overall Health Snapshot</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 6px;">{overall_label}</div>
                <div style="font-size: 0.98rem; color: #334155; margin-bottom: 10px;">{overall_note}</div>
                <div style="font-size: 0.88rem; color: #64748b;">Last analysis status: {rec.get('status', 'Unknown')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        compact_pills([
            f"Diabetes: {risk.get('diabetes', {}).get('risk_level', '—')}",
            f"Heart: {risk.get('heart', {}).get('risk_level', '—')}",
            f"BMI: {rec.get('patient_inputs', {}).get('bmi', '—')}",
            f"Symptoms: {len(rec.get('patient_inputs', {}).get('symptoms', []))}",
        ])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card(
                "Diabetes Risk",
                str(risk.get("diabetes", {}).get("risk_level", "—")),
                "ML-based",
                get_status_color(str(risk.get("diabetes", {}).get("risk_level", ""))),
                "🩸",
            )
        with c2:
            metric_card(
                "Heart Risk",
                str(risk.get("heart", {}).get("risk_level", "—")),
                "Rule-based",
                get_status_color(str(risk.get("heart", {}).get("risk_level", ""))),
                "❤️",
            )
        with c3:
            metric_card(
                "BMI",
                str(rec.get("patient_inputs", {}).get("bmi", "—")),
                str(rec.get("patient_inputs", {}).get("bmi_category", "")),
                "#14b8a6",
                "⚖️",
            )
        with c4:
            metric_card(
                "Symptoms",
                str(len(rec.get("patient_inputs", {}).get("symptoms", []))),
                "Reported",
                "#8b5cf6",
                "📝",
            )

        st.markdown("### ✨ Summary")
        info_card("At a glance", patient_friendly_explanation(risk), "#2563eb", "📘")

        priorities = generate_priorities(risk, indicators)
        questions = generate_doctor_questions(risk, indicators)

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            info_card(
                "Top Priorities",
                "<br>".join([f"• {p}" for p in priorities]),
                "#f59e0b",
                "🎯",
            )
        with pcol2:
            info_card(
                "Doctor Questions",
                "<br>".join([f"• {q}" for q in questions]),
                "#8b5cf6",
                "👨‍⚕️",
            )

        st.markdown("### 🧪 Key Health Indicators")
        extract_key_indicator_cards(indicators)

        st.markdown("### 🌿 Lifestyle Suggestions")
        recs = generate_lifestyle_recommendations(risk, rec.get("patient_inputs", {}), indicators)
        rc1, rc2 = st.columns(2)
        with rc1:
            info_card("Food", "<br>".join([f"• {x}" for x in recs["Food"]]), "#f59e0b", "🥗")
            info_card("Activity", "<br>".join([f"• {x}" for x in recs["Activity"]]), "#10b981", "🚶")
        with rc2:
            info_card("Daily Habits", "<br>".join([f"• {x}" for x in recs["Daily Habits"]]), "#0ea5e9", "🛌")
            info_card("Follow-Up", "<br>".join([f"• {x}" for x in recs["Follow-Up"]]), "#ef4444", "📅")

        st.markdown("### 📈 Comparison With Past Report")
        comparison = compare_with_previous(history)
        comp1, comp2, comp3 = st.columns(3)
        with comp1:
            comparison_card("Glucose", comparison.get("glucose_delta"), better_when_lower=True)
        with comp2:
            comparison_card("Systolic BP", comparison.get("systolic_bp_delta"), better_when_lower=True)
        with comp3:
            comparison_card("BMI", comparison.get("bmi_delta"), better_when_lower=True)

        st.markdown("### 📊 Trends Over Time")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown("**Glucose**")
            render_trend_chart(history, "glucose", "Glucose")
        with t2:
            st.markdown("**Systolic BP**")
            render_trend_chart(history, "systolic_bp", "Systolic BP")
        with t3:
            st.markdown("**BMI**")
            render_trend_chart(history, "bmi", "BMI")

        st.markdown("### 🤖 Ask AI About This Report")
        if not st.session_state["chat_history"]:
            st.caption("Ask something simple, like: What does my glucose mean?")

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        chat_col1, chat_col2 = st.columns([5, 1])
        with chat_col1:
            question = st.text_input(
                "Ask a question about the report",
                placeholder="Example: What does my glucose mean?",
                key="ai_question",
            )
        with chat_col2:
            send_clicked = st.button("Send", key="send_ai_question", use_container_width=True)

        if send_clicked and question.strip():
            st.session_state["chat_history"].append({"role": "user", "content": question})
            answer = ask_llm_about_report(question, rec, indicators, risk)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.rerun()

        st.caption("This assistant explains results in simple language and does not provide diagnosis.")


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

# ---------------------------------------------------
# DETAILED RESULTS
# ---------------------------------------------------
with results_tab:
    st.markdown("## 🧾 Detailed Results")

    rec = st.session_state.record
    indicators = st.session_state.indicators
    risk = st.session_state.risk

    if rec is None:
        st.info("No run yet. Go to Upload & Analyze first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Status", rec.get("status", ""))
        with c2:
            st.metric("Patient ID", rec.get("patient_id", ""))
        with c3:
            st.metric("Extracted Characters", str(len(rec.get("raw_text", ""))))

        st.markdown("### Patient Inputs")
        pi = rec.get("patient_inputs", {})
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Age", str(pi.get("age", "")))
        with p2:
            st.metric("BMI", "—" if pi.get("bmi") is None else str(pi.get("bmi")))
        with p3:
            st.metric("BMI Category", str(pi.get("bmi_category", "")))
        with p4:
            st.metric("Smoker", str(pi.get("smoker", "")))

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
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("### Risk Summary")
        if not risk:
            st.info("Risk result not available yet.")
        else:
            st.markdown("#### Diabetes Risk")
            d1, d2, d3 = st.columns(3)
            d1.metric("Risk Level", risk.get("diabetes", {}).get("risk_level", ""))
            d2.metric("Confidence Score", str(risk.get("diabetes", {}).get("confidence_score", "")))
            d3.metric("Risk Score", str(risk.get("diabetes", {}).get("risk_score", "")))

            st.markdown("**Contributing Indicators — Diabetes**")
            reasons_d = risk.get("diabetes", {}).get("reasons", [])
            if reasons_d:
                for item in reasons_d:
                    st.write(f"• {item}")
            else:
                st.write("• No reasons available")

            st.markdown("#### Heart Disease Risk")
            h1, h2, h3 = st.columns(3)
            h1.metric("Risk Level", risk.get("heart", {}).get("risk_level", ""))
            h2.metric("Confidence Score", str(risk.get("heart", {}).get("confidence_score", "")))
            h3.metric("Risk Score", str(risk.get("heart", {}).get("risk_score", "")))

            st.markdown("**Contributing Indicators — Heart Disease**")
            reasons_h = risk.get("heart", {}).get("reasons", [])
            if reasons_h:
                for item in reasons_h:
                    st.write(f"• {item}")
            else:
                st.write("• No reasons available")

            with st.expander("Recommendations / Notes", expanded=False):
                for tip in risk.get("general_notes", []):
                    st.write(f"• {tip}")

            st.warning(
                "These outputs are informational and non-diagnostic. "
                "They are intended to support understanding of possible risk indicators and should not replace professional medical evaluation."
            )

        if show_raw_text:
            st.markdown("### Raw Extracted Text (Preview)")
            raw = rec.get("raw_text", "")
            st.text_area("Raw text", raw[:raw_preview_chars], height=260)
