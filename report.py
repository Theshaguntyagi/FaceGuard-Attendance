"""
report.py — View and export attendance reports.

Usage:
    python report.py
"""

import csv
import os
from datetime import datetime

from database import get_all_students, get_attendance, init_db


# ── helpers ──────────────────────────────────────────────────────────────────

def _print_table(headers: list, rows: list):
    if not rows:
        print("  (no records found)\n")
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "  +" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "  |" + "|".join(f" {{:<{w}}} " for w in widths) + "|"

    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print(sep)
    print()


def _export_csv(rows: list, filename: str):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])
        writer.writerows(rows)
    print(f"  Exported {len(rows)} record(s) → {filename}\n")


# ── report functions ──────────────────────────────────────────────────────────

def report_today():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n  Attendance — {today}")
    rows = get_attendance(date=today)
    _print_table(["Name", "Date", "Time"], rows)
    print(f"  Total present: {len(rows)}")


def report_all():
    print("\n  Full Attendance History")
    rows = get_attendance()
    _print_table(["Name", "Date", "Time"], rows)
    print(f"  Total records: {len(rows)}")


def report_by_date(date: str):
    print(f"\n  Attendance — {date}")
    rows = get_attendance(date=date)
    _print_table(["Name", "Date", "Time"], rows)
    print(f"  Total present: {len(rows)}")


def report_summary():
    """Per-student attendance count."""
    rows = get_attendance()
    counts: dict = {}
    for name, _, _ in rows:
        counts[name] = counts.get(name, 0) + 1

    print("\n  Attendance Summary (all time)")
    summary = sorted(counts.items(), key=lambda x: -x[1])
    _print_table(["Name", "Days Present"], summary)


def list_students():
    print("\n  Registered Students")
    rows = get_all_students()
    _print_table(["Name", "Registered At"], rows)
    print(f"  Total registered: {len(rows)}")


# ── menu ──────────────────────────────────────────────────────────────────────

def menu():
    init_db()
    options = [
        ("1", "Today's attendance"),
        ("2", "Attendance by specific date"),
        ("3", "Full attendance history"),
        ("4", "Per-student summary"),
        ("5", "List registered students"),
        ("6", "Export today → CSV"),
        ("7", "Export all records → CSV"),
        ("0", "Exit"),
    ]

    while True:
        print("\n=== FaceGuard — Reports ===")
        for key, label in options:
            print(f"  {key}. {label}")

        choice = input("\n  Choice: ").strip()

        if choice == "1":
            report_today()
        elif choice == "2":
            date = input("  Date (YYYY-MM-DD): ").strip()
            report_by_date(date)
        elif choice == "3":
            report_all()
        elif choice == "4":
            report_summary()
        elif choice == "5":
            list_students()
        elif choice == "6":
            today = datetime.now().strftime("%Y-%m-%d")
            rows = get_attendance(date=today)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _export_csv(rows, f"attendance_{today}_{ts}.csv")
        elif choice == "7":
            rows = get_attendance()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _export_csv(rows, f"attendance_all_{ts}.csv")
        elif choice == "0":
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    menu()
