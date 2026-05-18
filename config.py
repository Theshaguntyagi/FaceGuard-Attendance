import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ENCODINGS_DIR = os.path.join(DATA_DIR, "encodings")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "known_faces.pkl")

# Recognition
MODEL_NAME        = "ArcFace"   # ArcFace | Facenet512 | VGG-Face
DETECTOR_BACKEND  = "opencv"    # opencv | ssd | mtcnn | retinaface
COSINE_THRESHOLD  = 0.40        # Lower = stricter match
FRAME_SCALE       = 0.75        # Downscale factor for detection
MIN_SAMPLES       = 10          # Frames per registration

# Sessions / attendance policy
DEFAULT_SESSION_NAME = "Class"
DEFAULT_LATE_AFTER   = "09:15"  # HH:MM — anyone marked after this is "late"
