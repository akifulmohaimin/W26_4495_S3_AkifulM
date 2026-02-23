"""
Clinical Decision Support System (CDSS) — Stage 1 + Stage 2

What this script does:

1) Validates an uploaded file (type + size + exists)
2) Extracts text from PDFs using embedded text (fast)
3) Falls back to OCR for scanned PDFs (pdf2image + Tesseract)
4) Pulls CBC indicators using regex + assigns Low/Normal/High flags
5) Saves a structured JSON record to disk

Requirements:
pip install pytesseract pdf2image pillow pypdf

"""

import json
import re
import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pytesseract
from pdf2image import convert_from_path
from PIL import ImageOps
from pypdf import PdfReader

# =========================================================
# Configuration
# =========================================================

UPLOAD_DIR = Path("./uploads").resolve()
OUTPUT_DIR = Path("./outputs").resolve()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_FILE_MB = 10

POPPLER_PATH = None


# =========================================================
# Stage 1: Secure file checks
# =========================================================
def validate_input_file(file_path: str) -> Path:
    """Basic safety checks: file exists, allowed extension, max size."""
    p = Path(file_path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")

    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise ValueError(f"File too large: {size_mb:.2f} MB (max {MAX_FILE_MB} MB)")

    return p


# =========================================================
# Stage 1: PDF text extraction (fast path)
# =========================================================
def extract_embedded_pdf_text(pdf_path: Path) -> str:
    """Extracts text from a text-based PDF using pypdf."""
    reader = PdfReader(str(pdf_path))
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            chunks.append(f"- Page {page_num} -\n{page_text}\n")

    return "".join(chunks)


# =========================================================
# Stage 1: OCR helpers (slow path)
# =========================================================

def ocr_pdf_to_text(pdf_path: Path, poppler_path: Optional[str] = None) -> str:
    """OCR a scanned PDF by converting pages to images and running Tesseract."""
    pages = convert_from_path(str(pdf_path), dpi=300, poppler_path=poppler_path)

    ocr_config = "--oem 3 --psm 6"
    chunks = []

    for page_num, img in enumerate(pages, start=1):
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        page_text = pytesseract.image_to_string(img, config=ocr_config)
        chunks.append(f"- Page {page_num} -\n{page_text}\n")

    return "".join(chunks)


def ocr_image_to_text(image_path: Path) -> str:
    """OCR for standalone image files (JPG/PNG)."""
    ocr_config = "--oem 3 --psm 6"

    return pytesseract.image_to_string(str(image_path), config=ocr_config)





# =========================================================
# Stage 1: Main extraction pipeline
# =========================================================


def extract_report_text(file_path: str, poppler_path: Optional[str] = None) -> Dict[str, Any]:
    """Returns a record dict with patient ID and extracted raw text."""
    p = validate_input_file(file_path)

    record = {
        "patient_id": f"AUTO_GEN_{uuid.uuid4().hex[:8].upper()}",
        "source_file": p.name,
        "raw_text": "",
        "processed_indicators": {},
        "status": "Incomplete",
    }

    try:
        if p.suffix.lower() == ".pdf":
            text = extract_embedded_pdf_text(p)
            if len(text.strip()) < 50:
                text = ocr_pdf_to_text(p, poppler_path=poppler_path)
        else:
            text = ocr_image_to_text(p)

        record["raw_text"] = text
        record["status"] = "Extracted"
        return record

    except Exception as e:
        record["status"] = "Failed"
        record["error"] = repr(e)
        return record


# =========================================================
# Stage 2: CBC parsing + flagging
# =========================================================

def safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def parse_reference_range(ref: str) -> Tuple[Optional[float], Optional[float]]:
    """Supports reference formats like '13-17' or '< 2'."""
    ref = (ref or "").strip()
    m = re.search(r"(\d+(\.\d+)?)\s*[-–]\s*(\d+(\.\d+)?)", ref)
    if m:
        return float(m.group(1)), float(m.group(3))
    m = re.search(r"<\s*(\d+(\.\d+)?)", ref)
    if m:
        return None, float(m.group(1))
    return None, None


def classify_flag(val: Optional[float], low: Optional[float], high: Optional[float]) -> str:
    if val is None:
        return "Unknown"
    if low is not None and val < low:
        return "Low"
    if high is not None and val > high:
        return "High"
    if low is not None or high is not None:
        return "Normal"
    return "Unknown"


CBC_PATTERNS = {
    "Hemoglobin": r"HEMOGLOBIN\s+(\d+(\.\d+)?)\s*(g\/d[lL])?\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "Total Leukocyte Count": r"TOTAL\s+LEU[CK]OCYTE\s+COUNT\s+(\d+)\s*(cumm|\/cumm)?\s+(\d+\s*[-–]\s*\d+)",
    "Neutrophils %": r"NEUTROPHILS\s+(\d+(\.\d+)?)\s*%\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "Lymphocyte %": r"LYMPHOCY?TE\s+(\d+(\.\d+)?)\s*%\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "Eosinophils %": r"EOSINOPHILS\s+(\d+(\.\d+)?)\s*%\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "Monocytes %": r"MONOCYTES\s+(\d+(\.\d+)?)\s*%\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "Basophils %": r"BASOPHILS\s+(\d+(\.\d+)?)\s*%\s+(<\s*\d+(\.\d+)?|\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "Platelet Count": r"PLATELET\s+COUNT\s+(\d+(\.\d+)?)\s*(lakhs\/cumm)?\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "Total RBC Count": r"TOTAL\s+RBC\s+COUNT\s+(\d+(\.\d+)?)\s*(million\/cumm)?\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "HCT %": r"(HCT|HEMATOCRIT\s+VALUE,\s*HCT)\s+(\d+(\.\d+)?)\s*%\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "MCV": r"\bMCV\b\s+(\d+(\.\d+)?)\s*fL\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "MCH": r"\bMCH\b\s+(\d+(\.\d+)?)\s*pg\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
    "MCHC": r"\bMCHC\b\s+(\d+(\.\d+)?)\s*%\s+(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)",
}


def extract_cbc(text: str) -> Dict[str, Dict[str, Any]]:
    """Extract CBC fields from raw text using regex."""
    results = {}
    for indicator, pattern in CBC_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if indicator == "HCT %":
            value = safe_float(match.group(2))
            ref_raw = match.group(4)
            unit = "%"
        else:
            value = safe_float(match.group(1))
            ref_raw = match.group(4) if indicator == "Hemoglobin" else match.group(3)
            unit = None
        low, high = parse_reference_range(ref_raw)
        results[indicator] = {
            "value": value,
            "unit": unit,
            "ref_raw": ref_raw,
            "ref_low": low,
            "ref_high": high,
            "flag": classify_flag(value, low, high),
        }
    return results


# =========================================================
# Save output JSON
# =========================================================

def save_json_record(record: Dict[str, Any], out_dir: Path = OUTPUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record['patient_id']}_record.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return out_path 


# =========================================================
#  Compatibility Aliases 
# =========================================================

def run_extraction_pipeline(file_path: str, poppler_path: Optional[str] = None) -> Dict[str, Any]:
    """Alias for extract_report_text as expected by app.py."""
    return extract_report_text(file_path, poppler_path=poppler_path)


def extract_cbc_indicators(text: str) -> Dict[str, Dict[str, Any]]:
    """Alias for extract_cbc as expected by app.py."""
    return extract_cbc(text)


# =========================================================
# Entry point (demo run)
# =========================================================

if __name__ == "__main__":
    sample_file = "medical_report.pdf" 
    record = run_extraction_pipeline(sample_file)
    print("Status:", record["status"])
    if record["status"] != "Failed":
        record["processed_indicators"]["cbc"] = extract_cbc_indicators(record["raw_text"])
        print("Fields extracted:", len(record["processed_indicators"]["cbc"]))