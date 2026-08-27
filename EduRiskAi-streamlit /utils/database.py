"""
utils/database.py

SQLite schema, connection handling, and CRUD helpers for:
    users, students, predictions, activity_logs

Design notes:
  - Every query uses parameterised placeholders ("?") -- never f-string or
    .format() SQL -- so user input can never be interpreted as SQL.
  - get_connection() opens a fresh connection per call rather than sharing
    one long-lived connection. SQLite connections aren't safe to share
    across threads, and Streamlit can run script reruns on different
    threads, so "one connection per operation" is the safe default here.
  - This file currently exposes only what Authentication (this stage)
    needs. Student/prediction CRUD functions are added incrementally as
    Student Management, Prediction History, etc. are built -- the schema
    below already has all four tables so it doesn't need to change shape
    later, only gain more functions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "student_risk.db"


def get_connection() -> sqlite3.Connection:
    """Open a fresh connection with dict-like row access and FK enforcement on."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK(role IN ('Administrator', 'Lecturer')),
    full_name     TEXT,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS students (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    student_code   TEXT UNIQUE NOT NULL,
    full_name      TEXT NOT NULL,
    course         TEXT,
    academic_year  INTEGER,
    email          TEXT,
    created_by     INTEGER REFERENCES users(id),
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id         INTEGER REFERENCES students(id),
    age                INTEGER,
    gender             INTEGER,
    ethnicity          INTEGER,
    parental_education INTEGER,
    study_time_weekly  REAL,
    absences           INTEGER,
    tutoring           INTEGER,
    parental_support   INTEGER,
    extracurricular    INTEGER,
    sports             INTEGER,
    music              INTEGER,
    volunteering       INTEGER,
    prediction         INTEGER NOT NULL,
    risk_probability   REAL NOT NULL,
    risk_level         TEXT NOT NULL,
    model_version      TEXT,
    predicted_by       INTEGER REFERENCES users(id),
    predicted_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS activity_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES users(id),
    action     TEXT NOT NULL,
    details    TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db() -> None:
    """Create all tables if they don't exist yet. Safe to call on every app start."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def seed_default_admin(username: str, password_hash: str, full_name: str = "Administrator") -> bool:
    """
    Insert a default Administrator account ONLY if the users table is
    completely empty, so a brand-new database always has a way to log in
    without ever duplicating or overwriting an account on later restarts.

    Returns True if a seed account was actually created.
    """
    conn = get_connection()
    try:
        existing = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        if existing["n"] > 0:
            return False
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name) "
            "VALUES (?, ?, 'Administrator', ?)",
            (username, password_hash, full_name),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_user_by_username(username: str) -> sqlite3.Row | None:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()


def upsert_student(
    student_code: str,
    full_name: str,
    course: str | None,
    academic_year: int | None,
    created_by: int | None,
) -> int:
    """
    Insert a new student or update an existing one matched by student_code
    (the human-readable ID like "STU-2026-001"). Returns the internal
    surrogate id -- predictions.student_id references THIS, not
    student_code directly, so correcting a student's code later can never
    orphan their prediction history.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO students (student_code, full_name, course, academic_year, created_by)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(student_code) DO UPDATE SET
                full_name = excluded.full_name,
                course = excluded.course,
                academic_year = excluded.academic_year
            """,
            (student_code, full_name, course, academic_year, created_by),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM students WHERE student_code = ?", (student_code,)
        ).fetchone()
        return row["id"]
    finally:
        conn.close()


def insert_prediction(
    student_id: int | None,
    encoded_input: dict,
    prediction: int,
    probability: float,
    risk_level: str,
    model_version: str,
    predicted_by: int | None,
) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                student_id, age, gender, ethnicity, parental_education,
                study_time_weekly, absences, tutoring, parental_support,
                extracurricular, sports, music, volunteering,
                prediction, risk_probability, risk_level, model_version, predicted_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                encoded_input["Age"], encoded_input["Gender"], encoded_input["Ethnicity"],
                encoded_input["ParentalEducation"], encoded_input["StudyTimeWeekly"],
                encoded_input["Absences"], encoded_input["Tutoring"], encoded_input["ParentalSupport"],
                encoded_input["Extracurricular"], encoded_input["Sports"], encoded_input["Music"],
                encoded_input["Volunteering"], prediction, probability, risk_level,
                model_version, predicted_by,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def log_activity(user_id: int | None, action: str, details: str | None = None) -> None:
    """Record an audit-trail row. Never raises -- a logging failure should
    never be allowed to break the actual operation being logged."""
    try:
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
                (user_id, action, details),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        pass


def get_all_predictions() -> "pd.DataFrame":
    """Predictions joined with student identity, newest first. Used by
    Dashboard, Prediction History, and Analytics."""
    import pandas as pd
    conn = get_connection()
    try:
        return pd.read_sql_query(
            """
            SELECT p.*, s.student_code, s.full_name AS student_name, s.course
            FROM predictions p
            LEFT JOIN students s ON p.student_id = s.id
            ORDER BY p.predicted_at DESC
            """,
            conn,
        )
    finally:
        conn.close()


def get_all_students() -> "pd.DataFrame":
    import pandas as pd
    conn = get_connection()
    try:
        return pd.read_sql_query("SELECT * FROM students ORDER BY created_at DESC", conn)
    finally:
        conn.close()


def batch_insert_predictions(rows: list[dict]) -> int:
    """
    Bulk-insert prediction rows with a single executemany() rather than one
    execute() per row -- meaningfully faster for a few thousand rows (the
    scale of the real dataset this project uses) than a Python-level loop
    of individual INSERTs.

    Each dict in `rows` needs: student_id, the 12 encoded feature keys,
    prediction, probability, risk_level, model_version, predicted_by.
    """
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO predictions (
                student_id, age, gender, ethnicity, parental_education,
                study_time_weekly, absences, tutoring, parental_support,
                extracurricular, sports, music, volunteering,
                prediction, risk_probability, risk_level, model_version, predicted_by
            ) VALUES (:student_id, :Age, :Gender, :Ethnicity, :ParentalEducation,
                      :StudyTimeWeekly, :Absences, :Tutoring, :ParentalSupport,
                      :Extracurricular, :Sports, :Music, :Volunteering,
                      :prediction, :probability, :risk_level, :model_version, :predicted_by)
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def list_users() -> "pd.DataFrame":
    import pandas as pd
    conn = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT id, username, role, full_name, is_active, created_at FROM users ORDER BY created_at",
            conn,
        )
    finally:
        conn.close()


def create_user(username: str, password_hash: str, role: str, full_name: str) -> tuple[bool, str]:
    conn = get_connection()
    try:
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            return False, f"Username '{username}' already exists."
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
            (username, password_hash, role, full_name),
        )
        conn.commit()
        return True, f"Account '{username}' created."
    finally:
        conn.close()


def set_user_active(user_id: int, is_active: bool) -> None:
    conn = get_connection()
    try:
        conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (int(is_active), user_id))
        conn.commit()
    finally:
        conn.close()
