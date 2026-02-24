# THIS PYTHON UI SCRIPT INCLUDES 

# CDSS Prototype UI (Stage 1 + Stage 2 + Stage 3 + Patient Inputs/BMI)
# - Upload PDF/JPG/PNG
# - Patient inputs (height/weight -> BMI) + a few basic fields
# - Runs extraction pipeline (PDF embedded text -> OCR fallback)
# - Parses CBC indicators + flags

# - Computes simple rule-based risk score + explanations
# - Shows results + raw text preview + JSON export

# REQUIREMENTS :
#   pip install streamlit pytesseract pdf2image pillow pypdf

# RUN:  python -m streamlit run sampleapp.py


import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

from cbc_pipeline import run_extraction_pipeline, extract_cbc_indicators
from risk_engine import compute_risk_from_cbc
from authdb import init_db, create_user, verify_user

init_db()

def logout():
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None
    st.session_state["username"] = None

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None
    st.session_state["username"] = None

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

# ----- Protected area -----
st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
if st.sidebar.button("Logout"):
    logout()
    st.rerun()


# -------------------------
# Helpers
# -------------------------
def compute_bmi(height_cm: float, weight_kg: float) -> Tuple[Optional[float], str]:
    """Returns (bmi_value, bmi_category)."""
    if height_cm <= 0 or weight_kg <= 0:
        return None, "Invalid"
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)

    # WHO categories
    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25:
        cat = "Normal"
    elif bmi < 30:
        cat = "Overweight"
    else:
        cat = "Obese"
    return round(bmi, 1), cat


# -------------------------
# Page setup
# -------------------------

st.set_page_config(page_title="CDSS Prototype (Stage 1+2+3)", layout="wide")

st.title("Clinical Decision Support System (CDSS) — Prototype")
st.caption("Upload → Extract (OCR fallback) → CBC structuring → Risk scoring → Export JSON")

with st.expander("Disclaimer", expanded=True):
    st.warning(
        "This is for CSIS-4495 (Applied Research Project Course). Not medical advice. Not a diagnosis. "
        "OCR/extraction accuracy depends on report format and scan quality."
    )

# Sidebar Controls 
st.sidebar.header("Settings (Optional)")
poppler_path = st.sidebar.text_input(
    "Poppler path (Windows only, optional)",
    value="",
    help="If pdf2image fails on Windows, install Poppler and paste its /bin path here.",
).strip() or None

st.sidebar.markdown("---")
show_raw_text = st.sidebar.checkbox("Show raw extracted text", value=True)
raw_preview_chars = st.sidebar.slider("Raw text preview length", 500, 12000, 6000, step=500)

# Tabs
tab_upload, tab_results, tab_export = st.tabs(["Upload", "Results", "Export"])

# Shared state
if "record" not in st.session_state:
    st.session_state.record = None
if "cbc" not in st.session_state:
    st.session_state.cbc = None
if "risk" not in st.session_state:
    st.session_state.risk = None
if "patient_inputs" not in st.session_state:
    st.session_state.patient_inputs = {
        "age": 25,
        "sex": "Prefer not to say",
        "height_cm": 170.0,
        "weight_kg": 65.0,
        "smoker": "No",
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
        pi["height_cm"] = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=float(pi["height_cm"]), step=0.5)
        pi["weight_kg"] = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=float(pi["weight_kg"]), step=0.5)

    with c3:
        pi["smoker"] = st.selectbox("Smoker", ["No", "Yes"], index=["No", "Yes"].index(pi["smoker"]))
        pi["symptoms"] = st.multiselect(
            "Symptoms (optional)",
            ["Fatigue", "Fever", "Cough", "Shortness of breath", "Dizziness", "Chest pain", "Other"],
            default=pi["symptoms"],
        )

    bmi_value, bmi_category = compute_bmi(pi["height_cm"], pi["weight_kg"])
    st.metric("BMI (calculated)", "—" if bmi_value is None else str(bmi_value))
    st.caption(f"BMI category: {bmi_category}")

    pi["consent"] = st.checkbox("I understand this is a prototype and not medical advice.", value=bool(pi["consent"]))

    st.markdown("---")
    st.subheader("2) Upload Medical Report")

    uploaded = st.file_uploader(
        "Upload a PDF/JPG/PNG (e.g., CBC report)",
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
            if status in {"Extracted"}:
                st.progress(40)
            elif status in {"Structured"}:
                st.progress(70)
            elif status in {"Scored"}:
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

            with st.spinner("Stage 1: Extracting text (PDF text → OCR fallback)"):
                rec = run_extraction_pipeline(str(tmp_path), poppler_path=poppler_path)

            if rec.get("status") == "Failed":
                st.session_state.record = rec
                st.session_state.cbc = None
                st.session_state.risk = None
                st.error("Extraction failed.")
                st.code(rec.get("error", "Unknown error"))
            else:
                with st.spinner("Stage 2: Parsing CBC indicators..."):
                    cbc = extract_cbc_indicators(rec.get("raw_text", ""))

                with st.spinner("Stage 3: Computing rule-based risk (prototype)..."):
                    risk = compute_risk_from_cbc(cbc)

                # Attach structured inputs + BMI into the record (end-to-end)
                rec["patient_inputs"] = {
                    "age": pi["age"],
                    "sex": pi["sex"],
                    "height_cm": pi["height_cm"],
                    "weight_kg": pi["weight_kg"],
                    "bmi": bmi_value,
                    "bmi_category": bmi_category,
                    "smoker": pi["smoker"],
                    "symptoms": pi["symptoms"],
                    "consent": pi["consent"],
                }

                rec.setdefault("processed_indicators", {})
                rec["processed_indicators"]["cbc"] = cbc
                rec["risk_result"] = risk
                rec["status"] = "Scored"

                st.session_state.record = rec
                st.session_state.cbc = cbc
                st.session_state.risk = risk

                st.success("Done! Go to the Results tab to view patient inputs, CBC output, and risk summary.")

# -------------------------
# TAB 2: Results
# -------------------------

with tab_results:
    st.subheader("Results")

    rec = st.session_state.record
    cbc = st.session_state.cbc
    risk = st.session_state.risk

    if rec is None:
        st.info("No run yet. Go to Upload tab → fill inputs → upload a file → click Run.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Status", rec.get("status", ""))
        m2.metric("Patient ID", rec.get("patient_id", ""))
        m3.metric("Extracted chars", str(len(rec.get("raw_text", ""))))

        st.markdown("### Patient Inputs (Saved with Record)")
        pi = rec.get("patient_inputs", {})
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Age", str(pi.get("age", "")))
        p2.metric("BMI", "—" if pi.get("bmi") is None else str(pi.get("bmi")))
        p3.metric("BMI Category", str(pi.get("bmi_category", "")))
        p4.metric("Smoker", str(pi.get("smoker", "")))


        st.markdown("### CBC Indicators")
        if not cbc:
            st.warning("No CBC fields detected. (This can happen if OCR/text format differs.)")
        else:
            rows = []
            for name, v in cbc.items():
                rows.append(
                    {
                        "Indicator": name,
                        "Value": v.get("value"),
                        "Unit": v.get("unit") or "",
                        "Reference": v.get("ref_raw") or "",
                        "Flag": v.get("flag") or "",
                    }
                )
            st.table(sorted(rows, key=lambda r: r["Indicator"]))

        st.markdown("### Risk Summary (Rule-based Prototype)")
        if not risk:
            st.info("Risk score not available yet.")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Risk Level", risk.get("risk_level", ""))
            r2.metric("Risk Score", str(risk.get("risk_score", "")))
            r3.metric("Abnormal Count", str(risk.get("abnormal_count", "")))

            st.markdown("**Why this result?**")
            for reason in risk.get("reasons", []):
                st.write(f"- {reason}")

            with st.expander("Recommendations / Notes"):
                for tip in risk.get("recommendations", []):
                    st.write(f"- {tip}")

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
            file_name=f"{rec.get('patient_id','record')}_record.json",
            mime="application/json",
        )

        st.markdown("### JSON Preview")

        st.code(json.dumps(rec, indent=2), language="json")

