# FaceGuard Attendance

A desktop face-recognition attendance system for Windows. Enrol students from a
webcam, run a class session, and FaceGuard automatically marks each recognised
face as **present**, **late**, or **absent** — backed by an ArcFace embedding
model and stored in a local SQLite database.

```
┌──────────────────────────────────────────────────────────┐
│  Dashboard │ Register │ Live Attendance │ Reports │ ...  │   ← Tkinter GUI
├──────────────────────────────────────────────────────────┤
│            ┌──────────────────────┐                      │
│            │   webcam preview     │   Marked today:      │
│            │                      │   ┌──────────────┐   │
│            │                      │   │ Alice  09:02 │   │
│            └──────────────────────┘   │ Bob    09:18 │   │
│            [Start session]            └──────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## Features

- **GUI** built in Tkinter with six tabs:
  Dashboard, Register, Live Attendance, Reports, Students, Settings.
- **One-shot enrolment** — capture ~10 frames of a person's face and store a
  mean ArcFace embedding.
- **Session-based attendance** with a configurable late cutoff time (HH:MM).
  Anyone seen after the cutoff is marked **late**; anyone the camera never
  recognises during the session is marked **absent** when you end the session.
- **Per-student summary** view across all sessions (present / late / absent /
  total) plus CSV export.
- **Background workers** keep face detection and ArcFace inference off the Tk
  main thread, so the preview stays smooth (~30 fps) while embeddings happen
  asynchronously.
- **Two entry points** — `python main.py` for the GUI, `python main.py --cli`
  for the legacy text menu.

---

## Requirements

| Component         | Version                                |
|-------------------|----------------------------------------|
| OS                | Windows 10 / 11 (tested on Windows 11) |
| Python            | 3.10 or 3.11                           |
| Webcam            | Any UVC-compatible camera              |
| MSVC C++ runtime  | 14.x (for TensorFlow's DLLs)           |

`requirements.txt`:

```
opencv-python>=4.8.0
deepface>=0.0.93
tf-keras>=2.16.0
numpy>=1.24.0
scipy>=1.11.0
Pillow>=10.0.0
```

> **Disk space.** The first run downloads:
> - TensorFlow wheel (~350 MB) during `pip install`
> - ArcFace weights (~130 MB) into `%USERPROFILE%\.deepface\weights\` on the
>   first embedding call.

---

## Install

```powershell
cd C:\Users\Shagun\FaceGuard
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow fails at import with
`Could not find the DLL(s) 'msvcp140.dll'`, install the
**Microsoft Visual C++ Redistributable (x64)**:
<https://aka.ms/vs/17/release/vc_redist.x64.exe>.

(As a workaround, FaceGuard ships a small bootstrap in `face_engine.py` that
will register any MSVC DLLs sitting in `venv\Scripts` via
`os.add_dll_directory()`, so you can also drop the DLLs there.)

---

## Run

```powershell
python main.py             # GUI (default)
python main.py --cli       # legacy text menu
```

Direct CLI scripts also still work:

```powershell
python register.py --name "Alice"
python recognize.py
python report.py
```

---

## Using the GUI

### Register
1. Click **Start camera** — preview appears immediately; the ArcFace model
   loads in the background (~5–8 s on first launch).
2. Wait for the status to read *Model ready — click Capture*.
3. Type a name → click **Capture** → look at the lens. The status reads
   `Capturing... 4 / 10 samples` as embeddings come in.
4. Done: the mean L2-normalised embedding is saved to
   `data/encodings/known_faces.pkl` and the name is added to `students`.

### Live Attendance
1. Set a session name and *late after* time (default `09:15`).
2. **Start session** — the cam opens, ArcFace runs every third frame.
   Recognised faces get a `present` or `late` row in `attendance`.
3. **End session** — closes the session and inserts an `absent` row for every
   registered student who wasn't seen.

### Reports
- Filter by date and/or status, switch to **Summary view** for per-student
  totals, **Export CSV**.

### Students
- Lists all enrolled people; **Remove selected** deletes both the DB row and
  the face encoding.

### Settings
- Tune `COSINE_THRESHOLD`, `FRAME_SCALE`, `MIN_SAMPLES`, default late-after,
  default session name — **in memory only** for this launch. Persist by
  editing `config.py`.

---

## Configuration (`config.py`)

| Key                    | Default   | Meaning                                          |
|------------------------|-----------|--------------------------------------------------|
| `MODEL_NAME`           | `ArcFace` | DeepFace face-recognition model                  |
| `DETECTOR_BACKEND`     | `opencv`  | Reserved for swapping detection backend          |
| `COSINE_THRESHOLD`     | `0.40`    | Lower = stricter match                           |
| `FRAME_SCALE`          | `0.75`    | Downscale factor for the detection pass          |
| `MIN_SAMPLES`          | `10`      | Embeddings averaged during enrolment             |
| `DEFAULT_SESSION_NAME` | `Class`   | Pre-filled session name                          |
| `DEFAULT_LATE_AFTER`   | `09:15`   | `HH:MM` after which an arrival is *late*         |

---

## Project structure

```
FaceGuard/
├── main.py               # entry point (GUI by default, --cli for text menu)
├── gui.py                # Tkinter app (6 tabs + webcam thread + embedding worker)
├── face_engine.py        # detection, embedding, matching, MSVC DLL bootstrap
├── register.py           # CLI registration
├── recognize.py          # CLI live recognition
├── report.py             # CLI reports & CSV export
├── database.py           # SQLite schema + helpers (auto-migrates old DBs)
├── config.py             # tunable constants
├── requirements.txt
├── setup_guide.txt       # plain-text install steps
├── README.md             # this file
├── DOCUMENTATION.md      # architecture & developer reference
└── data/                 # created at runtime
    ├── attendance.db
    └── encodings/known_faces.pkl
```

---

## Troubleshooting

| Symptom                                                | Cause / Fix                                                                                       |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Start camera shows a dark/black preview                | Webcam backend didn't enumerate. `face_engine` retries CAP_DSHOW → CAP_MSMF → default automatically. |
| `ImportError: Could not find the DLL(s) 'msvcp140.dll'`| Install the VC++ x64 redistributable, or drop `msvcp140.dll` into `venv\Scripts`.                 |
| Capture sits at 0 / 10 forever                         | ArcFace weights failed to download. Manually save `arcface_weights.h5` to `%USERPROFILE%\.deepface\weights\`. |
| Every face shows as Unknown                            | `COSINE_THRESHOLD` too tight — raise to 0.45–0.50. Or re-register under varied lighting.          |
| Wrong person matched                                   | Threshold too loose — lower to 0.35.                                                              |
| GUI freezes during enrolment                           | Old build. Pull latest — embeddings now run on a worker thread.                                   |

---

## Future work

See [DOCUMENTATION.md](DOCUMENTATION.md#roadmap) for the full roadmap. Highlights:

- Multi-camera support (entry + exit doors).
- Anti-spoofing (blink / depth / liveness checks).
- Web-based dashboard (FastAPI + a small front-end) so reports are visible
  outside the host machine.
- Class roster / timetable so sessions auto-start from a schedule.
- GPU inference path (`tensorflow[and-cuda]`) for sub-50 ms embeddings.
- Email / SMS notifications for absentees.

---

## License

No license has been declared on this project. Treat the code as **all rights
reserved** until a license is added.
