"""
face_engine.py — shared face detection, embedding, and matching helpers.
"""

import os
import pickle
import sys


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def _ensure_msvc_runtime():
    """On Windows, make co-located MSVC runtime DLLs visible to TensorFlow.

    Some systems have the Office-provided msvcp140.dll but lack the standalone
    VC++ redistributable in System32. We bundle copies in venv\\Scripts and
    register the directory here so TF's import-time DLL check passes.
    """
    if sys.platform != "win32":
        return
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "venv", "Scripts"),
        os.path.dirname(sys.executable),
    ]
    for d in candidates:
        if d and os.path.isdir(d) and os.path.exists(os.path.join(d, "msvcp140.dll")):
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass


_ensure_msvc_runtime()

import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

from config import COSINE_THRESHOLD, ENCODINGS_DIR, ENCODINGS_FILE, MODEL_NAME

_face_cascade = None


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def detect_faces(frame, scale=1.0):
    """Return list of (x, y, w, h) in full-frame coordinates."""
    cascade = get_face_cascade()
    if scale != 1.0:
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    else:
        small = frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(60, 60))
    if scale != 1.0:
        faces = [(int(x / scale), int(y / scale),
                  int(w / scale), int(h / scale)) for (x, y, w, h) in faces]
    return list(faces)


def get_embedding(face_img: np.ndarray, normalize=False):
    """Return embedding vector for a cropped face image, or None.

    If `normalize=True`, the vector is L2-normalised (useful for matching).
    """
    try:
        result = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
        if result:
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            if normalize:
                emb /= np.linalg.norm(emb) + 1e-9
            return emb
    except Exception:
        pass
    return None


def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {"names": [], "encodings": []}


def save_encodings(data):
    os.makedirs(ENCODINGS_DIR, exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def match(embedding, known_encodings, known_names, threshold=COSINE_THRESHOLD):
    """Return (name, distance). name='Unknown' if no match within threshold."""
    if not known_encodings:
        return "Unknown", 1.0
    distances = [cosine(embedding, enc) for enc in known_encodings]
    best_idx = int(np.argmin(distances))
    if distances[best_idx] <= threshold:
        return known_names[best_idx], float(distances[best_idx])
    return "Unknown", float(distances[best_idx])


def warm_up():
    """Force model load so the first real frame isn't slow."""
    try:
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        get_embedding(dummy)
    except Exception:
        pass
