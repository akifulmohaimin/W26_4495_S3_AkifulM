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



def compute_bmi(height_cm: float, weight_kg: float) -> Tuple[Optional[float], str]:
    """Returns (bmi_value, bmi_category)."""
    if height_cm <= 0 or weight_kg <= 0:
        return None, "Invalid"

    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25:
        cat = "Normal"
    elif bmi < 30:
        cat = "Overweight"
    else:
        cat = "Obese"

    return round(bmi, 1), cat


st.title("CDSS - Patient Portal")

if not st.session_state["logged_in"]:
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            ok, user_id = verify_user(u, p)
            if ok:
                st.session_state["logged_in"] = True
                st.session_state["user_id"] = user_id
                st.session_state["username"] = u.strip().lower()
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.subheader("Register")
        u2 = st.text_input("Username", key="reg_user")
        p2 = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Create account"):
            ok, msg = create_user(u2, p2)
            (st.success if ok else st.error)(msg)

    st.stop()


# -------------------------
# Protected area
# -------------------------
st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
if st.sidebar.button("Logout"):
    logout()
    st.rerun()


# -------------------------
# Page setup
# -------------------------
st.title("Clinical Decision Support System (CDSS) — Prototype")
st.caption("Upload → Extract → Parse indicators → Diabetes ML prediction + Heart rule-based estimate → Export JSON")

with st.expander("Disclaimer", expanded=True):
    st.warning(
        "This is for CSIS-4495 (Applied Research Project Course). Not medical advice. Not a diagnosis. "
        "OCR/extraction accuracy depends on report format and scan quality."
    )

st.sidebar.header("Settings (Optional)")
poppler_path = st.sidebar.text_input(
    "Poppler path (Windows only, optional)",
    value="",
    help="If pdf2image fails on Windows, install Poppler and paste its /bin path here.",
).strip() or None

st.sidebar.markdown("---")
show_raw_text = st.sidebar.checkbox("Show raw extracted text", value=True)
raw_preview_chars = st.sidebar.slider("Raw text preview length", 500, 12000, 6000, step=500)

tab_upload, tab_results, tab_export = st.tabs(["Upload", "Results", "Export"])

if "record" not in st.session_state:
    st.session_state.record = None
if "indicators" not in st.session_state:
    st.session_state.indicators = None
if "risk" not in st.session_state:
    st.session_state.risk = None
if "patient_inputs" not in st.session_state:
    st.session_state.patient_inputs = {
        "age": 25,
        "sex": "Prefer not to say",
        "height_cm": 170.0,
        "weight_kg": 65.0,
        "smoker": "No",
        "family_history_diabetes": "No",
        "family_history_heart_disease": "No",
        "symptoms": [],
        "consent": True,
    }

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
