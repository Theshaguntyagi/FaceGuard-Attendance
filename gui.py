"""
gui.py — Tkinter GUI for FaceGuard Attendance.

Run with:
    python gui.py
"""

import csv
import os
import threading
from datetime import datetime
from tkinter import (BooleanVar, DoubleVar, IntVar, StringVar, Tk, filedialog,
                     messagebox, ttk)
from tkinter import font as tkfont

import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from database import (add_student, delete_student, end_session,
                      get_active_session, get_all_students, get_attendance,
                      get_distinct_dates, get_per_student_summary,
                      get_recent_sessions, get_student_names, init_db,
                      mark_attendance, start_session)
from face_engine import (detect_faces, get_embedding, load_encodings, match,
                         save_encodings, warm_up)


PREVIEW_W = 480
PREVIEW_H = 360


# ── webcam capture ────────────────────────────────────────────────────────────

# Backends tried in order. CAP_DSHOW is usually fastest on Windows, but some
# webcams only enumerate under CAP_MSMF or the default backend.
_CAPTURE_BACKENDS = [
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_MSMF",  cv2.CAP_MSMF),
    ("default",   0),
]


def _open_camera(index=0):
    """Try each backend until one returns a working capture. Returns (cap, name) or (None, error_str)."""
    errors = []
    for name, backend in _CAPTURE_BACKENDS:
        try:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        if cap.isOpened():
            # Make sure we can actually read a frame (some backends "open" but yield nothing)
            for _ in range(5):
                ok, _ = cap.read()
                if ok:
                    return cap, name
            cap.release()
            errors.append(f"{name}: opened but no frames")
        else:
            errors.append(f"{name}: not opened")
    return None, "; ".join(errors) if errors else "no backend available"


class CameraStream:
    """Background-thread webcam wrapper that can be started and stopped repeatedly."""

    def __init__(self):
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._backend_name = None
        self._last_error = None

    @property
    def running(self):
        return self._running

    @property
    def backend(self):
        return self._backend_name

    @property
    def last_error(self):
        return self._last_error

    def start(self, index=0):
        if self._running:
            return True
        cap, info = _open_camera(index)
        if cap is None:
            self._last_error = info
            return False
        self._cap = cap
        self._backend_name = info
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        t = self._thread
        self._thread = None
        if t and t.is_alive():
            t.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._frame = None
        self._backend_name = None

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _loop(self):
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame


# ── helpers ───────────────────────────────────────────────────────────────────

def cv2_to_tk(frame, max_w=PREVIEW_W, max_h=PREVIEW_H):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def status_for_time(now_str, late_after):
    if not late_after:
        return "present"
    return "late" if now_str[:5] > late_after else "present"


# ── tab: dashboard ────────────────────────────────────────────────────────────

class DashboardTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=16)
        self.app = app

        title = ttk.Label(self, text="Dashboard", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w")

        stats = ttk.Frame(self)
        stats.pack(fill="x", pady=(12, 4))

        self.lbl_today = ttk.Label(stats, text="Present today: 0", font=("Segoe UI", 11))
        self.lbl_late = ttk.Label(stats, text="Late today: 0", font=("Segoe UI", 11))
        self.lbl_absent = ttk.Label(stats, text="Absent today: 0", font=("Segoe UI", 11))
        self.lbl_total = ttk.Label(stats, text="Total registered: 0", font=("Segoe UI", 11))

        for w in (self.lbl_today, self.lbl_late, self.lbl_absent, self.lbl_total):
            w.pack(side="left", padx=14)

        self.lbl_session = ttk.Label(self, text="No active session.",
                                     font=("Segoe UI", 10, "italic"))
        self.lbl_session.pack(anchor="w", pady=(8, 12))

        ttk.Separator(self).pack(fill="x", pady=6)
        ttk.Label(self, text="Today's activity", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        cols = ("name", "time", "status")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=15)
        for c, w in zip(cols, (220, 120, 100)):
            self.tree.heading(c, text=c.title())
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=(6, 4))

        ttk.Button(self, text="Refresh", command=self.refresh).pack(anchor="e", pady=4)

    def refresh(self):
        today = datetime.now().strftime("%Y-%m-%d")
        rows = get_attendance(date=today)
        present = sum(1 for r in rows if r[3] == "present")
        late = sum(1 for r in rows if r[3] == "late")
        absent = sum(1 for r in rows if r[3] == "absent")
        total = len(get_all_students())

        self.lbl_today.config(text=f"Present today: {present}")
        self.lbl_late.config(text=f"Late today: {late}")
        self.lbl_absent.config(text=f"Absent today: {absent}")
        self.lbl_total.config(text=f"Total registered: {total}")

        active = get_active_session()
        if active:
            sid, sname, sdate, sstart, late_after = active
            self.lbl_session.config(
                text=f"Active session: '{sname}' (started {sstart}, late after {late_after or '—'})"
            )
        else:
            self.lbl_session.config(text="No active session.")

        self.tree.delete(*self.tree.get_children())
        for name, _date, t, status, _conf in rows:
            self.tree.insert("", "end", values=(name, t, status))


# ── tab: register ─────────────────────────────────────────────────────────────

class RegisterTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=16)
        self.app = app
        self.cam = CameraStream()
        self.capturing = False
        self.collected = []

        # Embedding worker — keeps DeepFace.represent off the Tk main thread
        # so the preview stays smooth during enrolment.
        self._emb_q = __import__("queue").Queue(maxsize=2)
        self._emb_thread = threading.Thread(target=self._embed_worker, daemon=True)
        self._emb_thread.start()
        self._model_ready = False

        ttk.Label(self, text="Register New Face", font=("Segoe UI", 16, "bold")).pack(anchor="w")

        form = ttk.Frame(self)
        form.pack(fill="x", pady=10)

        ttk.Label(form, text="Name:").pack(side="left")
        self.name_var = StringVar()
        ttk.Entry(form, textvariable=self.name_var, width=30).pack(side="left", padx=8)

        self.btn_start = ttk.Button(form, text="Start camera", command=self.toggle_camera)
        self.btn_start.pack(side="left", padx=4)
        self.btn_capture = ttk.Button(form, text="Capture", command=self.start_capture,
                                      state="disabled")
        self.btn_capture.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(form, text="Stop", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=4)

        preview_box = ttk.Frame(self, width=PREVIEW_W, height=PREVIEW_H)
        preview_box.pack(pady=10)
        preview_box.pack_propagate(False)
        self.preview = ttk.Label(preview_box, background="#222", anchor="center")
        self.preview.pack(fill="both", expand=True)

        self.progress = ttk.Progressbar(self, length=PREVIEW_W, mode="determinate",
                                        maximum=config.MIN_SAMPLES)
        self.progress.pack(pady=4)

        self.status = ttk.Label(self, text="Idle.", font=("Segoe UI", 10))
        self.status.pack(anchor="w", pady=4)

        self._tk_img = None  # keep a reference

    def toggle_camera(self):
        if self.cam.running:
            self.stop_camera()
        else:
            if not self.cam.start():
                messagebox.showerror(
                    "Camera",
                    f"Cannot open webcam.\n\n{self.cam.last_error or 'No backend worked.'}"
                )
                return
            self.btn_start.config(text="Stop camera")
            self.btn_capture.config(state="normal")
            self.btn_stop.config(state="normal")

            if not self._model_ready:
                self.status.config(
                    text=f"Camera running ({self.cam.backend}). Loading face model in background..."
                )
                threading.Thread(target=self._warmup_model, daemon=True).start()
            else:
                self.status.config(
                    text=f"Camera running ({self.cam.backend}). Click Capture to begin enrolment."
                )
            self._update_preview()

    def _warmup_model(self):
        warm_up()
        self._model_ready = True
        try:
            self.status.config(
                text=f"Camera running ({self.cam.backend}). Model ready — click Capture."
            )
        except Exception:
            pass

    def _embed_worker(self):
        """Background thread that consumes face crops from the queue and produces embeddings."""
        while True:
            crop = self._emb_q.get()
            if crop is None:
                break
            try:
                emb = get_embedding(crop)
            except Exception:
                emb = None
            if emb is not None and self.capturing and len(self.collected) < config.MIN_SAMPLES:
                self.collected.append(emb)

    def stop_camera(self):
        self.capturing = False
        self.cam.stop()
        self.btn_start.config(text="Start camera")
        self.btn_capture.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.status.config(text="Camera stopped.")

    def start_capture(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Name", "Please enter a name.")
            return
        existing = load_encodings()
        if name in existing["names"]:
            messagebox.showwarning("Already registered",
                                   f"'{name}' is already registered. Remove first.")
            return
        if not self._model_ready:
            self.status.config(text="Model still loading — please wait a moment, then try again.")
            return
        self.collected = []
        self.capturing = True
        self.progress.config(maximum=config.MIN_SAMPLES, value=0)
        self.status.config(text=f"Capturing {config.MIN_SAMPLES} samples for '{name}'...")

    def _update_preview(self):
        frame = self.cam.get_frame()
        if frame is not None:
            display = frame.copy()
            faces = detect_faces(frame)

            if self.capturing:
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    crop = frame[y:y + h, x:x + w].copy()
                    # Drop the crop if the worker is still busy — no point queueing
                    # frame after frame; ArcFace is the bottleneck.
                    try:
                        self._emb_q.put_nowait(crop)
                    except Exception:
                        pass
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)
                elif len(faces) > 1:
                    cv2.putText(display, "Multiple faces", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
                else:
                    cv2.putText(display, "No face", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

                count = len(self.collected)
                self.progress["value"] = count
                self.status.config(
                    text=f"Capturing... {count} / {config.MIN_SAMPLES} samples"
                )
                if count >= config.MIN_SAMPLES:
                    self._finish_capture()
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (180, 180, 180), 2)

            self._tk_img = cv2_to_tk(display)
            self.preview.config(image=self._tk_img)

        if self.cam.running:
            self.after(30, self._update_preview)

    def _finish_capture(self):
        self.capturing = False
        name = self.name_var.get().strip()
        if len(self.collected) < config.MIN_SAMPLES // 2:
            self.status.config(text=f"Aborted — only {len(self.collected)} samples.")
            return

        mean = np.mean(self.collected, axis=0)
        mean /= np.linalg.norm(mean) + 1e-9

        data = load_encodings()
        data["names"].append(name)
        data["encodings"].append(mean)
        save_encodings(data)
        add_student(name)

        self.status.config(text=f"OK — '{name}' registered ({len(self.collected)} samples).")
        messagebox.showinfo("Registered", f"'{name}' enrolled successfully.")
        self.name_var.set("")
        self.app.refresh_all()


# ── tab: live attendance ──────────────────────────────────────────────────────

class AttendanceTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=16)
        self.app = app
        self.cam = CameraStream()
        self.session_id = None
        self.late_after = config.DEFAULT_LATE_AFTER
        self.marked = {}      # name -> (time, status, dist)
        self.known = {"names": [], "encodings": []}
        self.frame_idx = 0
        self._tk_img = None

        ttk.Label(self, text="Live Attendance", font=("Segoe UI", 16, "bold")).pack(anchor="w")

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=10)

        ttk.Label(ctrl, text="Session:").pack(side="left")
        self.sess_name = StringVar(value=config.DEFAULT_SESSION_NAME)
        ttk.Entry(ctrl, textvariable=self.sess_name, width=18).pack(side="left", padx=4)

        ttk.Label(ctrl, text="Late after (HH:MM):").pack(side="left", padx=(10, 0))
        self.late_var = StringVar(value=config.DEFAULT_LATE_AFTER)
        ttk.Entry(ctrl, textvariable=self.late_var, width=8).pack(side="left", padx=4)

        self.btn_start = ttk.Button(ctrl, text="Start session", command=self.start_session)
        self.btn_start.pack(side="left", padx=6)
        self.btn_end = ttk.Button(ctrl, text="End session", command=self.end_session, state="disabled")
        self.btn_end.pack(side="left", padx=4)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=8)

        left = ttk.Frame(body)
        left.pack(side="left", padx=(0, 12))
        preview_box = ttk.Frame(left, width=PREVIEW_W, height=PREVIEW_H)
        preview_box.pack()
        preview_box.pack_propagate(False)
        self.preview = ttk.Label(preview_box, background="#222")
        self.preview.pack(fill="both", expand=True)
        self.status = ttk.Label(left, text="Not running.", font=("Segoe UI", 10))
        self.status.pack(anchor="w", pady=6)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Marked this session:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        cols = ("name", "time", "status", "dist")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=14)
        for c, w in zip(cols, (160, 100, 90, 80)):
            self.tree.heading(c, text=c.title())
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=6)

    def start_session(self):
        existing = get_active_session()
        if existing:
            messagebox.showinfo("Active session",
                                f"A session is already active (id={existing[0]}).")
            return
        data = load_encodings()
        if not data["encodings"]:
            messagebox.showerror("No faces", "No registered faces. Register someone first.")
            return
        self.known = data

        if not self.cam.start():
            messagebox.showerror(
                "Camera",
                f"Cannot open webcam.\n\n{self.cam.last_error or 'No backend worked.'}"
            )
            return

        warm_up()
        self.late_after = self.late_var.get().strip() or None
        self.session_id = start_session(self.sess_name.get().strip() or "Session",
                                        late_after=self.late_after)
        self.marked = {}
        self.tree.delete(*self.tree.get_children())
        self.btn_start.config(state="disabled")
        self.btn_end.config(state="normal")
        self.status.config(text=f"Session running ({self.cam.backend}, id={self.session_id}).")
        self._update_preview()
        self.app.refresh_all()

    def end_session(self):
        if self.session_id is None:
            return
        marked_absent = end_session(self.session_id)
        self.cam.stop()
        self.status.config(text=f"Session ended. Marked {marked_absent} absent.")
        messagebox.showinfo("Session ended",
                            f"Session ended.\n{len(self.marked)} attended, "
                            f"{marked_absent} marked absent.")
        self.session_id = None
        self.btn_start.config(state="normal")
        self.btn_end.config(state="disabled")
        self.app.refresh_all()

    def _update_preview(self):
        frame = self.cam.get_frame()
        if frame is not None:
            self.frame_idx += 1
            display = frame.copy()
            if self.frame_idx % 3 == 0:
                faces = detect_faces(frame, scale=config.FRAME_SCALE)
                for (x, y, w, h) in faces:
                    crop = frame[y:y + h, x:x + w]
                    emb = get_embedding(crop)
                    name, dist = ("Unknown", 1.0)
                    if emb is not None:
                        name, dist = match(emb, self.known["encodings"], self.known["names"])
                    color = (34, 180, 34) if name != "Unknown" else (50, 50, 210)
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display, name, (x, y + h + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if name != "Unknown" and name not in self.marked:
                        now_str = datetime.now().strftime("%H:%M:%S")
                        status = status_for_time(now_str, self.late_after)
                        if mark_attendance(name, status=status, confidence=dist,
                                           session_id=self.session_id):
                            self.marked[name] = (now_str, status, dist)
                            self.tree.insert("", "end", values=(name, now_str, status, f"{dist:.3f}"))

            self._tk_img = cv2_to_tk(display)
            self.preview.config(image=self._tk_img)

        if self.cam.running:
            self.after(30, self._update_preview)


# ── tab: reports ──────────────────────────────────────────────────────────────

class ReportsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=16)
        self.app = app

        ttk.Label(self, text="Reports", font=("Segoe UI", 16, "bold")).pack(anchor="w")

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=10)

        ttk.Label(ctrl, text="Date:").pack(side="left")
        self.date_var = StringVar(value="(all)")
        self.date_combo = ttk.Combobox(ctrl, textvariable=self.date_var, width=14, state="readonly")
        self.date_combo.pack(side="left", padx=4)

        ttk.Label(ctrl, text="Status:").pack(side="left", padx=(10, 0))
        self.status_var = StringVar(value="all")
        self.status_combo = ttk.Combobox(ctrl, textvariable=self.status_var, width=10,
                                         state="readonly",
                                         values=["all", "present", "late", "absent"])
        self.status_combo.pack(side="left", padx=4)

        ttk.Button(ctrl, text="Apply", command=self.refresh).pack(side="left", padx=6)
        ttk.Button(ctrl, text="Export CSV", command=self.export_csv).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Summary view", command=self.show_summary).pack(side="left", padx=4)

        cols = ("name", "date", "time", "status", "confidence")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=20)
        for c, w in zip(cols, (180, 110, 100, 90, 100)):
            self.tree.heading(c, text=c.title())
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=4)

        self.summary_mode = False
        self.current_rows = []

    def refresh(self):
        self.summary_mode = False
        dates = ["(all)"] + get_distinct_dates()
        self.date_combo["values"] = dates
        if self.date_var.get() not in dates:
            self.date_var.set("(all)")

        date = None if self.date_var.get() == "(all)" else self.date_var.get()
        rows = get_attendance(date=date)
        status = self.status_var.get()
        if status != "all":
            rows = [r for r in rows if r[3] == status]

        self.tree.delete(*self.tree.get_children())
        for c in ("name", "date", "time", "status", "confidence"):
            self.tree.heading(c, text=c.title())

        self.current_rows = rows
        for name, d, t, st, conf in rows:
            cstr = f"{conf:.3f}" if conf is not None else ""
            self.tree.insert("", "end", values=(name, d, t, st, cstr))

    def show_summary(self):
        self.summary_mode = True
        rows = get_per_student_summary()
        self.tree.delete(*self.tree.get_children())
        for c, label in zip(("name", "date", "time", "status", "confidence"),
                            ("Name", "Present", "Late", "Absent", "Total")):
            self.tree.heading(c, text=label)
        self.current_rows = rows
        for name, present, late, absent, total in rows:
            self.tree.insert("", "end", values=(name, present, late, absent, total))

    def export_csv(self):
        if not self.current_rows:
            messagebox.showinfo("Empty", "Nothing to export.")
            return
        default = ("summary" if self.summary_mode else "attendance") + \
                  "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default,
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if self.summary_mode:
                w.writerow(["Name", "Present", "Late", "Absent", "Total"])
            else:
                w.writerow(["Name", "Date", "Time", "Status", "Confidence"])
            for row in self.current_rows:
                w.writerow(row)
        messagebox.showinfo("Exported", f"Wrote {len(self.current_rows)} rows to:\n{path}")


# ── tab: students ─────────────────────────────────────────────────────────────

class StudentsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=16)
        self.app = app

        ttk.Label(self, text="Registered Students", font=("Segoe UI", 16, "bold")).pack(anchor="w")

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=10)
        ttk.Button(ctrl, text="Refresh", command=self.refresh).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Remove selected", command=self.remove_selected).pack(side="left", padx=4)

        cols = ("name", "registered_at")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=20)
        self.tree.heading("name", text="Name")
        self.tree.heading("registered_at", text="Registered at")
        self.tree.column("name", width=240)
        self.tree.column("registered_at", width=200)
        self.tree.pack(fill="both", expand=True, pady=4)

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        for name, reg in get_all_students():
            self.tree.insert("", "end", values=(name, reg))

    def remove_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        name = self.tree.item(sel[0])["values"][0]
        if not messagebox.askyesno("Remove", f"Remove '{name}' and their face encoding?"):
            return
        data = load_encodings()
        if name in data["names"]:
            idx = data["names"].index(name)
            data["names"].pop(idx)
            data["encodings"].pop(idx)
            save_encodings(data)
        delete_student(name)
        self.app.refresh_all()


# ── tab: settings ─────────────────────────────────────────────────────────────

class SettingsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=16)
        self.app = app

        ttk.Label(self, text="Settings", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(self,
                  text="Changes apply for this session (in-memory). "
                       "Edit config.py to persist.",
                  font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(0, 12))

        form = ttk.Frame(self)
        form.pack(anchor="w")

        row = 0
        self.cos_var = DoubleVar(value=config.COSINE_THRESHOLD)
        self._row(form, row, "Cosine threshold (lower = stricter)",
                  ttk.Spinbox(form, from_=0.20, to=0.80, increment=0.01,
                              textvariable=self.cos_var, width=8))
        row += 1
        self.scale_var = DoubleVar(value=config.FRAME_SCALE)
        self._row(form, row, "Frame scale (detection)",
                  ttk.Spinbox(form, from_=0.25, to=1.0, increment=0.05,
                              textvariable=self.scale_var, width=8))
        row += 1
        self.samples_var = IntVar(value=config.MIN_SAMPLES)
        self._row(form, row, "Samples per registration",
                  ttk.Spinbox(form, from_=5, to=100, increment=1,
                              textvariable=self.samples_var, width=8))
        row += 1
        self.late_var = StringVar(value=config.DEFAULT_LATE_AFTER)
        self._row(form, row, "Default late-after (HH:MM)",
                  ttk.Entry(form, textvariable=self.late_var, width=10))
        row += 1
        self.session_name_var = StringVar(value=config.DEFAULT_SESSION_NAME)
        self._row(form, row, "Default session name",
                  ttk.Entry(form, textvariable=self.session_name_var, width=18))
        row += 1

        ttk.Button(self, text="Apply", command=self.apply).pack(anchor="w", pady=12)

    def _row(self, parent, row, label, widget):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        widget.grid(row=row, column=1, sticky="w", padx=4, pady=4)

    def apply(self):
        config.COSINE_THRESHOLD = float(self.cos_var.get())
        config.FRAME_SCALE = float(self.scale_var.get())
        config.MIN_SAMPLES = int(self.samples_var.get())
        config.DEFAULT_LATE_AFTER = self.late_var.get().strip() or None
        config.DEFAULT_SESSION_NAME = self.session_name_var.get().strip() or "Session"
        messagebox.showinfo("Applied", "Settings applied for this session.")


# ── app shell ─────────────────────────────────────────────────────────────────

class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("FaceGuard Attendance")
        self.geometry("1080x720")
        self.minsize(900, 600)

        try:
            style = ttk.Style(self)
            style.theme_use("vista")
        except Exception:
            pass

        init_db()

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.dashboard = DashboardTab(self.nb, self)
        self.register_tab = RegisterTab(self.nb, self)
        self.attendance_tab = AttendanceTab(self.nb, self)
        self.reports = ReportsTab(self.nb, self)
        self.students = StudentsTab(self.nb, self)
        self.settings = SettingsTab(self.nb, self)

        self.nb.add(self.dashboard, text="Dashboard")
        self.nb.add(self.register_tab, text="Register")
        self.nb.add(self.attendance_tab, text="Live Attendance")
        self.nb.add(self.reports, text="Reports")
        self.nb.add(self.students, text="Students")
        self.nb.add(self.settings, text="Settings")

        self.refresh_all()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def refresh_all(self):
        self.dashboard.refresh()
        self.students.refresh()
        self.reports.refresh()

    def _on_close(self):
        try:
            self.register_tab.cam.stop()
            self.attendance_tab.cam.stop()
        except Exception:
            pass
        self.destroy()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
