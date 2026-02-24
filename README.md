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

This project implements a patient-centered Clinical Decision Support System (CDSS) designed to help non-clinical users understand potential disease risk indicators from medical reports.

The system:
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
pip install streamlit pytesseract pdf2image pillow pypdf pandas numpy scikit-learn
```

---

## Step 4 – Install External Dependencies

### Install Tesseract OCR

Windows:  
Download from:  
https://github.com/tesseract-ocr/tesseract  

Add installation path to system PATH.

Mac:
```
brew install tesseract
```

Linux:
```
sudo apt install tesseract-ocr
```

---

### Install Poppler

Mac:
```
brew install poppler
```

Linux:
```
sudo apt install poppler-utils
```

Windows:  
Download Poppler binaries and add to PATH.

---

# Running the Application (VS Code)

In VS Code terminal:

```bash
cd Implementation
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





   
