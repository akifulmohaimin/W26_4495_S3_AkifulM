import sqlite3
import json
from pathlib import Path
import bcrypt

DB_PATH = Path("cdss_auth.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            patient_id TEXT,
            bmi REAL,
            glucose REAL,
            hba1c REAL,
            systolic_bp REAL,
            diastolic_bp REAL,
            cholesterol_total REAL,
            ldl REAL,
            diabetes_risk TEXT,
            heart_risk TEXT,
            cooccurrence_risk TEXT,
            record_json TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        conn.commit()


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()

    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, pw_hash),
            )
            conn.commit()
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def verify_user(username: str, password: str) -> tuple[bool, int | None]:
    username = username.strip().lower()

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        return False, None

    user_id = row["id"]
    pw_hash = row["password_hash"]

    if isinstance(pw_hash, str):
        pw_hash = pw_hash.encode("utf-8")

    ok = bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    return ok, (user_id if ok else None)


def save_report(user_id: int, record: dict, indicators: dict, risk: dict):
    patient_inputs = record.get("patient_inputs", {})

    def as_float(value):
        try:
            if value in [None, "", "—"]:
                return None
            return float(value)
        except Exception:
            return None

    row = {
        "user_id": user_id,
        "patient_id": record.get("patient_id", ""),
        "bmi": as_float(patient_inputs.get("bmi")),
        "glucose": as_float(indicators.get("glucose", {}).get("value")),
        "hba1c": as_float(indicators.get("hba1c", {}).get("value")),
        "systolic_bp": as_float(indicators.get("systolic_bp", {}).get("value")),
        "diastolic_bp": as_float(indicators.get("diastolic_bp", {}).get("value")),
        "cholesterol_total": as_float(indicators.get("cholesterol_total", {}).get("value")),
        "ldl": as_float(indicators.get("ldl", {}).get("value")),
        "diabetes_risk": risk.get("diabetes", {}).get("risk_level"),
        "heart_risk": risk.get("heart", {}).get("risk_level"),
        "cooccurrence_risk": risk.get("cooccurrence", {}).get("risk_level"),
        "record_json": json.dumps(record),
    }

    with get_conn() as conn:
        conn.execute("""
        INSERT INTO reports (
            user_id, patient_id, bmi, glucose, hba1c, systolic_bp, diastolic_bp,
            cholesterol_total, ldl, diabetes_risk, heart_risk, cooccurrence_risk, record_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["user_id"],
            row["patient_id"],
            row["bmi"],
            row["glucose"],
            row["hba1c"],
            row["systolic_bp"],
            row["diastolic_bp"],
            row["cholesterol_total"],
            row["ldl"],
            row["diabetes_risk"],
            row["heart_risk"],
            row["cooccurrence_risk"],
            row["record_json"],
        ))
        conn.commit()


def load_reports_for_user(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
        SELECT id, created_at, patient_id, bmi, glucose, hba1c,
               systolic_bp, diastolic_bp, cholesterol_total, ldl,
               diabetes_risk, heart_risk, cooccurrence_risk, record_json
        FROM reports
        WHERE user_id = ?
        ORDER BY created_at ASC, id ASC
        """, (user_id,)).fetchall()

    history = []
    for row in rows:
        history.append({
            "report_id": row["id"],
            "timestamp": row["created_at"],
            "patient_id": row["patient_id"],
            "bmi": row["bmi"],
            "glucose": row["glucose"],
            "hba1c": row["hba1c"],
            "systolic_bp": row["systolic_bp"],
            "diastolic_bp": row["diastolic_bp"],
            "cholesterol_total": row["cholesterol_total"],
            "ldl": row["ldl"],
            "diabetes_risk": row["diabetes_risk"],
            "heart_risk": row["heart_risk"],
            "cooccurrence_risk": row["cooccurrence_risk"],
        })
    return history
