"""
recognize.py — Live webcam recognition and attendance marking (CLI).

Usage:
    python recognize.py

Keys:
    q  — quit
    r  — reload encodings
    s  — print today's session summary
"""

from datetime import datetime

import cv2

from config import DEFAULT_LATE_AFTER, FRAME_SCALE
from database import init_db, mark_attendance
from face_engine import detect_faces, get_embedding, load_encodings, match, warm_up


def _status_for_time(now_str, late_after):
    """Return 'present' or 'late' based on HH:MM:SS comparison with HH:MM cutoff."""
    if not late_after:
        return "present"
    return "late" if now_str[:5] > late_after else "present"


def draw_label(frame, text, origin, bg_color):
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 0.65, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 4, y + base), bg_color, -1)
    cv2.putText(frame, text, (x + 2, y), font, scale, (255, 255, 255), thickness)


def run_recognition(late_after=DEFAULT_LATE_AFTER):
    init_db()
    data = load_encodings()

    if not data or not data["encodings"]:
        print("  No registered faces found. Run register.py first.")
        return

    known_encodings = data["encodings"]
    known_names = data["names"]
    print(f"  Loaded {len(known_names)} face(s): {', '.join(known_names)}")
    print("  Warming up face model...")
    warm_up()
    print("  Keys: [q] quit  [r] reload  [s] summary\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open webcam.")
        return

    marked_session = {}
    last_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % 3 == 0:
            faces = detect_faces(frame, scale=FRAME_SCALE)

            last_results = []
            for (x, y, w, h) in faces:
                face_crop = frame[y : y + h, x : x + w]
                emb = get_embedding(face_crop, normalize=False)

                name, dist = ("Unknown", 1.0)
                if emb is not None:
                    name, dist = match(emb, known_encodings, known_names)

                color = (34, 180, 34) if name != "Unknown" else (50, 50, 210)

                if name != "Unknown" and name not in marked_session:
                    now_str = datetime.now().strftime("%H:%M:%S")
                    status = _status_for_time(now_str, late_after)
                    if mark_attendance(name, status=status, confidence=dist):
                        marked_session[name] = (now_str, status)
                        print(f"  [{now_str}] {status.upper():<7} {name}  (dist={dist:.3f})")

                last_results.append((x, y, w, h, name, color))

        for (x, y, w, h, name, color) in last_results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_label(frame, name, (x, y + h + 20), color)

        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, now_str, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.putText(frame, f"Present: {len(marked_session)}", (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 1)

        cv2.imshow("FaceGuard  |  q=quit  r=reload  s=summary", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            data = load_encodings()
            if data:
                known_encodings = data["encodings"]
                known_names = data["names"]
                print(f"  Reloaded: {len(known_names)} face(s)")
        elif key == ord("s"):
            print(f"\n  --- Session ({datetime.now().strftime('%Y-%m-%d')}) ---")
            if marked_session:
                for n, (t, st) in sorted(marked_session.items()):
                    print(f"    {n:<25} {t}  [{st}]")
            else:
                print("    Nothing marked yet.")
            print()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Session ended — {len(marked_session)} marked.")


if __name__ == "__main__":
    print("=== FaceGuard — Live Attendance ===")
    run_recognition()
