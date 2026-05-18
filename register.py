"""
register.py — Enrol a new person into FaceGuard (CLI).

Usage:
    python register.py
    python register.py --name "Alice"
    python register.py --remove "Alice"
"""

import argparse

import cv2
import numpy as np

from config import MIN_SAMPLES
from database import add_student, delete_student, init_db
from face_engine import (detect_faces, get_embedding, load_encodings,
                         save_encodings, warm_up)


def register_face(name: str, on_progress=None):
    """Capture MIN_SAMPLES frames and store a mean embedding for `name`.

    on_progress: optional callable(count, total, frame_bgr) for GUI integration.
    """
    init_db()

    data = load_encodings()
    if name in data["names"]:
        print(f"  '{name}' is already registered. Remove first to re-register.")
        return False

    print(f"\n  Registering: {name}")
    print(f"  Look at the camera — capturing {MIN_SAMPLES} samples.")
    print("  Press 'q' to abort.\n")

    print("  Loading face model (first run downloads ~100 MB)...")
    warm_up()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open webcam.")
        return False

    collected = []

    while len(collected) < MIN_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        display = frame.copy()

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_crop = frame[y : y + h, x : x + w]
            emb = get_embedding(face_crop)
            if emb is not None:
                collected.append(emb)

            color = (0, 200, 0)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            label = f"Captured: {len(collected)}/{MIN_SAMPLES}"
            cv2.putText(display, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        elif len(faces) > 1:
            cv2.putText(display, "Multiple faces — only one person visible please",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
        else:
            cv2.putText(display, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)

        if MIN_SAMPLES > 0:
            prog = int((len(collected) / MIN_SAMPLES) * frame.shape[1])
            cv2.rectangle(display, (0, frame.shape[0] - 8),
                          (prog, frame.shape[0]), (0, 200, 0), -1)

        if on_progress:
            on_progress(len(collected), MIN_SAMPLES, display)

        cv2.imshow(f"Register — {name}", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(collected) < MIN_SAMPLES // 2:
        print(f"  Registration aborted — only {len(collected)} samples captured.")
        return False

    mean_encoding = np.mean(collected, axis=0)
    mean_encoding /= np.linalg.norm(mean_encoding) + 1e-9

    data["names"].append(name)
    data["encodings"].append(mean_encoding)
    save_encodings(data)
    add_student(name)

    print(f"  OK {name} registered successfully ({len(collected)} samples).")
    return True


def remove_face(name: str):
    data = load_encodings()
    if name not in data["names"]:
        print(f"  '{name}' not found.")
        return False

    idx = data["names"].index(name)
    data["names"].pop(idx)
    data["encodings"].pop(idx)
    save_encodings(data)
    delete_student(name)
    print(f"  OK {name} removed.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceGuard — Register a face")
    parser.add_argument("--name", type=str, help="Person's name to register")
    parser.add_argument("--remove", type=str, help="Person's name to remove")
    args = parser.parse_args()

    print("=== FaceGuard — Register ===")

    if args.remove:
        remove_face(args.remove)
    else:
        name = args.name or input("  Enter name: ").strip()
        if name:
            register_face(name)
        else:
            print("  Name cannot be empty.")
