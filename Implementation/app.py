import os
import re
import pytesseract
from pdf2image import convert_from_path

# Defining Initial Data Structures for storing extracted values
extracted_medical_record = {
    "patient_id": "AUTO_GEN_001",
    "raw_text": "",
    "processed_indicators": {},
    "status": "Incomplete"
}

# Implementing Secure PDF Upload Functionality
def secure_upload_check(filename):
    allowed_extensions = {'.pdf', '.jpg', '.png'}
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext in allowed_extensions:
        print(f"Secure Upload: great {filename} is a valid medical document.")
        return True
    print(f"Error: Sorry {file_ext} is not a supported format.")
    return False

# Medical report scanning OCR Pipeline (PDF/JPG/PNG)
def run_ocr_pipeline(file_path):
    print("Starting OCR Text Extraction Here...")
    try:
        # Converting PDF pages to images to read text
        pages = convert_from_path(file_path)
        full_text = ""
        
        for i, page in enumerate(pages):
            # Extracting text on Image page using Tesseract OCR
            page_text = pytesseract.image_to_string(page)
            full_text += f"- Page {i+1} -\n{page_text}\n"
        
        return full_text
    except Exception as e:
        return f"Pipeline Error: {str(e)}"

# Execution
sample_file = "medical_report.pdf" 

if secure_upload_check(sample_file):
    raw_content = run_ocr_pipeline(sample_file)
    extracted_medical_record["raw_text"] = raw_content
    extracted_medical_record["status"] = "Extracted"
    
    print("\n Extracted Content Summary ::")
    print(extracted_medical_record["raw_text"][:5000] + "...") 
