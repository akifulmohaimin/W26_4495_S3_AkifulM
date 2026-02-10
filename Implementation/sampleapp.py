import json
from datetime import datetime
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Clinical Decision Support System (Prototype)",layout="wide")

# ---------------------------------------------------------
# App header
# ---------------------------------------------------------
st.title("Clinical Decision Support System (CDSS)")

# ---------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------
with st.expander("Important Disclaimer", expanded=True):
    st.warning(
        "This system is a prototype and is NOT intended for medical diagnosis or treatment. "
        "It is designed only to demonstrate how patient-accessible medical data *could* be "
        "processed and presented in a future clinical decision support system."
    )

# ---------------------------------------------------------
# Application tabs
# ---------------------------------------------------------
tab_upload, tab_inputs, tab_results, tab_export = st.tabs(["Upload Report", "Patient Inputs", "Results", "Export"])

# =========================================================
# TAB 1: Upload medical report (simulation only)
# =========================================================
with tab_upload:
    st.subheader("Step 1: Upload a Medical Report")

    uploaded = st.file_uploader("Upload a PDF or image file (PDF / JPG / PNG)",type=["pdf", "jpg", "jpeg", "png"])

    if uploaded:
        st.success(f"File uploaded successfully: {uploaded.name}")

        st.json({
            "file_name": uploaded.name,
            "file_type": uploaded.type,
            "file_size_bytes": uploaded.size,
            "upload_time": datetime.now().isoformat(timespec="seconds"),
        })
    else:
        st.info("No file uploaded yet.")

    st.markdown("### Processing Status (Mock)")
    st.progress(25)
    st.caption("Stage 1: File upload and validation (UI simulation only)")

# =========================================================
# TAB 2: Structured patient inputs
# =========================================================
with tab_inputs:
    st.subheader("Step 2: Patient Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=0, max_value=120, value=27)
        sex = st.selectbox("Sex",["Male", "Female", "Other", "Prefer not to say"])

    with c2:
        height_cm = st.number_input("Height (cm)",min_value=50.0,max_value=250.0,value=170.0,step=0.5)
        weight_kg = st.number_input("Weight (kg)",min_value=10.0,max_value=250.0,value=65.0,step=0.5)

    with c3:
        smoker = st.selectbox("Smoking Status", ["No", "Yes"])
        symptoms = st.multiselect("Reported Symptoms (optional)",["Fatigue","Fever","Cough","Shortness of breath","Dizziness","Chest pain","Other"])

    # Calculate BMI
    bmi = weight_kg / ((height_cm / 100) ** 2) if height_cm else 0
    st.metric("Calculated BMI", f"{bmi:.1f}")

    consent = st.checkbox("I understand that this tool is a prototype and does not provide medical advice.",value=True)

# =========================================================
# TAB 3: Results (placeholder / mock data)
# =========================================================
with tab_results:
    st.subheader("Step 3: Results Overview (Mock Data)")

    st.markdown("#### Extracted Laboratory Indicators")
    mock_rows = [
        {"Indicator": "Hemoglobin", "Value": "15", "Unit": "g/dL", "Reference Range": "13–17", "Flag": "Normal"},
        {"Indicator": "Total Leukocyte Count", "Value": "5100", "Unit": "/cumm", "Reference Range": "4800–10800", "Flag": "Normal"},
        {"Indicator": "Lymphocyte Percentage", "Value": "18", "Unit": "%", "Reference Range": "20–40", "Flag": "Low"},
        {"Indicator": "MCHC", "Value": "35.7", "Unit": "%", "Reference Range": "31.5–34.5", "Flag": "High"},
    ]

    st.table(mock_rows)

    cA, cB = st.columns(2)

    with cA:
        st.markdown("#### Overall Risk Indicator (Mock)")
        st.metric("Risk Score", "0.32")
        st.caption("Model not implemented yet (planned future stage)")

    with cB:
        st.markdown("#### Explanation Summary")
        st.write("• Lymphocyte percentage flagged as low")
        st.write("• MCHC flagged as high")
        st.write("• Risk score shown here is illustrative only")

    st.markdown("#### Processing Progress (Mock)")
    st.progress(60)
    st.caption(
        "Stage 2: Structured data extraction (planned) • "
        "Stage 3: Model-based risk scoring (planned)"
    )

# =========================================================
# TAB 4: Export results (mock JSON)
# =========================================================
with tab_export:
    st.subheader("Step 4: Export Results")

    mock_record = {
        "patient_id": "DEMO_ID",
        "system_status": "UI_ONLY_PROTOTYPE",
        "uploaded_file": uploaded.name if uploaded else None,
        "patient_inputs": {
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": round(bmi, 1),
            "smoker": smoker,
            "symptoms": symptoms,
            "consent_acknowledged": consent,
        },
        "results": {
            "cbc_indicators": "placeholder",
            "risk_score": "placeholder",
            "explanations": "placeholder",
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    st.code(json.dumps(mock_record, indent=2), language="json")

    st.download_button(
        label="Download Mock Output (JSON)",
        data=json.dumps(mock_record, indent=2).encode("utf-8"),
        file_name="cdss_mock_output.json",
        mime="application/json"
    )
