# W26_4495_S3_AkifulM
# Student Names
1. AKIFUL MOHAIMIN
   ID: 300369823
   Email: mohaimina@student.douglascollege.ca
3. UTHARA UDAY
   ID: 300395271
   Email: udayu@student.douglascollege.ca

   <br> <br> <br>


# Clinical Decision Support System (CDSS)  
**CSIS 4495 – Applied Research Project (Winter 2026)**  

## Project Overview

The Clinical Decision Support System (CDSS) is a medical intelligence prototype that bridges the gap between raw laboratory data and patient understanding. The system provides a secure, end-to-end pipeline that ingests medical reports (PDF/JPG/PNG), extracts critical health indicators using Optical Character Recognition (OCR), and applies a rule-based risk engine to generate preliminary health insights.

### Core Modules:
- Secure Authentication: A robust login and registration gateway designed to protect sensitive patient profiles.
- Interactive Patient Dashboard: A user-centric interface for tracking demographics, physical metrics (Height/Weight), and current symptoms.
- Intelligent Extraction Pipeline: A hybrid processing engine that utilizes fast-path text extraction for digital files and an OCR fallback for scanned documents.
- Advanced Risk Engine: A rule-based logic system that evaluates extracted biomarkers to compute risk scores for conditions such as Diabetes and Heart Disease.

### Key Features:
- Automated Health Metrics: Real-time calculation of BMI and health categorization based on user profile inputs.
- Indicator Parsing & Flagging: Automatically identifies indicators (e.g., Glucose, HbA1c, Cholesterol) and flags them as Low, Normal, or High against standard reference ranges.
- Detailed Risk Summaries: Provides quantified Risk Levels and Confidence Scores, accompanied by specific clinical explanations for the findings.
- Secure Data Handling: Implements file extension validation and structured upload handling to ensure data integrity.
- Professional Export: Capability to generate and download a full clinical record in structured JSON format for healthcare interoperability.

In short, this system:
- Accepts PDF lab reports (text-based and scanned)
- Uses OCR for scanned documents
- Extracts clinical indicators (glucose, cholesterol, etc.)
- Generates structured features
- Applies supervised ML models (Logistic Regression, Random Forest)
- Displays non-diagnostic, explainable risk levels

This system does NOT provide medical diagnosis. It is for research and educational purposes only.

---

---

## System Requirements

- Python 3.10+
- VS Code
- Windows / macOS / Linux
- Tesseract OCR installed
- Poppler installed

---

# Installation & Setup (Using VS Code)

## Step 1 – Clone Repository

Open VS Code.

Press:
```
Ctrl + Shift + P
```

Type:
```
Git: Clone
```

Paste your repository URL:

```
https://github.com/YOUR-USERNAME/W26_4495_S3_AkifulM.git
```

Open the cloned folder in VS Code.

---

## Step 2 – Create Virtual Environment (Inside VS Code Terminal)

Open Terminal in VS Code:

```
Terminal → New Terminal
```

Run:

### Windows
```
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux
```
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` in the terminal.

---

## Step 3 – Install Python Dependencies

```
pip install -r requirements.txt
```

```
# Running the Application (VS Code)

In VS Code terminal:
streamlit run sampleapp.py
```

Open browser at:

```
http://localhost:8501
```

---


## Ethical Considerations

- Uses anonymized or synthetic data
- Non-diagnostic output
- Explainable risk predictions
- Patient awareness focused

---

## Academic Integrity

Developed for CSIS 4495 – Applied Research Project.  
All implementation work is original and maintained in this repository.





   
