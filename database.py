import os
import sqlite3
from datetime import datetime

from config import DB_PATH


# ── connection helper ─────────────────────────────────────────────────────────

def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA foreign_keys = ON")
    return c


# ── schema / migration ────────────────────────────────────────────────────────

def init_db():
    conn = _conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT UNIQUE NOT NULL,
            registered_at TEXT DEFAULT (datetime('now','localtime'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name  TEXT NOT NULL,
            date          TEXT NOT NULL,
            time          TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'present',
            confidence    REAL,
            session_id    INTEGER,
            UNIQUE(student_name, date)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            date        TEXT NOT NULL,
            start_time  TEXT NOT NULL,
            late_after  TEXT,
            end_time    TEXT
        )
    """)

    # Migration: add columns to attendance if upgrading from old schema
    c.execute("PRAGMA table_info(attendance)")
    cols = {row[1] for row in c.fetchall()}
    if "status" not in cols:
        c.execute("ALTER TABLE attendance ADD COLUMN status TEXT NOT NULL DEFAULT 'present'")
    if "confidence" not in cols:
        c.execute("ALTER TABLE attendance ADD COLUMN confidence REAL")
    if "session_id" not in cols:
        c.execute("ALTER TABLE attendance ADD COLUMN session_id INTEGER")

    conn.commit()
    conn.close()


# ── students ──────────────────────────────────────────────────────────────────

def add_student(name):
    conn = _conn()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (name) VALUES (?)", (name,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_all_students():
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT name, registered_at FROM students ORDER BY name")
    rows = c.fetchall()
    conn.close()
    return rows


def get_student_names():
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT name FROM students ORDER BY name")
    names = [r[0] for r in c.fetchall()]
    conn.close()
    return names


def delete_student(name):
    conn = _conn()
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE name=?", (name,))
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0


# ── attendance ────────────────────────────────────────────────────────────────

def mark_attendance(name, status="present", confidence=None, session_id=None):
    """Insert today's attendance row. Returns True if newly marked, False if duplicate."""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    conn = _conn()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO attendance (student_name, date, time, status, confidence, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, date, time, status, confidence, session_id),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_attendance(date=None):
    conn = _conn()
    c = conn.cursor()
    if date:
        c.execute(
            "SELECT student_name, date, time, status, confidence "
            "FROM attendance WHERE date=? ORDER BY time",
            (date,),
        )
    else:
        c.execute(
            "SELECT student_name, date, time, status, confidence "
            "FROM attendance ORDER BY date DESC, time DESC"
        )
    rows = c.fetchall()
    conn.close()
    return rows


def get_distinct_dates():
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT DISTINCT date FROM attendance ORDER BY date DESC")
    rows = [r[0] for r in c.fetchall()]
    conn.close()
    return rows


def get_per_student_summary():
    """Return list of (name, present, late, absent, total_days) across all sessions."""
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT s.name,
               COALESCE(SUM(CASE WHEN a.status='present' THEN 1 ELSE 0 END), 0) AS present,
               COALESCE(SUM(CASE WHEN a.status='late'    THEN 1 ELSE 0 END), 0) AS late,
               COALESCE(SUM(CASE WHEN a.status='absent'  THEN 1 ELSE 0 END), 0) AS absent
        FROM students s
        LEFT JOIN attendance a ON a.student_name = s.name
        GROUP BY s.name
        ORDER BY s.name
    """)
    out = []
    for name, present, late, absent in c.fetchall():
        total = present + late + absent
        out.append((name, present, late, absent, total))
    conn.close()
    return out


# ── sessions ──────────────────────────────────────────────────────────────────

def start_session(name, late_after=None):
    """Create a session for today. Returns session id."""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    start = now.strftime("%H:%M:%S")
    conn = _conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO sessions (name, date, start_time, late_after) VALUES (?, ?, ?, ?)",
        (name, date, start, late_after),
    )
    sid = c.lastrowid
    conn.commit()
    conn.close()
    return sid


def end_session(session_id):
    """Mark the session ended now and write absent rows for unmarked students."""
    end_time = datetime.now().strftime("%H:%M:%S")
    conn = _conn()
    c = conn.cursor()
    c.execute(
        "UPDATE sessions SET end_time=? WHERE id=? AND end_time IS NULL",
        (end_time, session_id),
    )

    # Find session date and missing students
    c.execute("SELECT date FROM sessions WHERE id=?", (session_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return 0
    date = row[0]

    c.execute("""
        SELECT s.name FROM students s
        WHERE s.name NOT IN (SELECT student_name FROM attendance WHERE date=?)
    """, (date,))
    missing = [r[0] for r in c.fetchall()]

    for name in missing:
        try:
            c.execute(
                "INSERT INTO attendance (student_name, date, time, status, session_id) "
                "VALUES (?, ?, ?, 'absent', ?)",
                (name, date, end_time, session_id),
            )
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    return len(missing)


def get_active_session():
    """Return (id, name, date, start_time, late_after) of the currently open session, or None."""
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, name, date, start_time, late_after FROM sessions
        WHERE end_time IS NULL
        ORDER BY id DESC LIMIT 1
    """)
    row = c.fetchone()
    conn.close()
    return row


def get_recent_sessions(limit=20):
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, name, date, start_time, end_time, late_after
        FROM sessions ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
