from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from math import dist
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _optional_imports():
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None
    try:
        import face_recognition  # type: ignore
    except ImportError:
        face_recognition = None
    return cv2, face_recognition


@dataclass
class FaceGuardAttendance:
    known_names: List[str] = field(default_factory=list)
    known_encodings: List[Sequence[float]] = field(default_factory=list)
    attendance: dict[str, datetime] = field(default_factory=dict)

    def load_known_faces(self, known_faces_dir: Path) -> None:
        cv2, face_recognition = _optional_imports()
        if cv2 is None or face_recognition is None:
            raise ImportError(
                "OpenCV and face_recognition are required for loading known faces."
            )

        image_files = sorted(
            file
            for file in known_faces_dir.iterdir()
            if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for image_path in image_files:
            image = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                continue
            self.known_names.append(image_path.stem)
            self.known_encodings.append(encodings[0].tolist())

    def _find_best_match(self, encoding: Sequence[float], tolerance: float) -> str:
        if not self.known_encodings:
            return "Unknown"

        best_name = "Unknown"
        best_distance = float("inf")
        for name, known_encoding in zip(self.known_names, self.known_encodings):
            candidate_distance = dist(encoding, known_encoding)
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_name = name

        return best_name if best_distance <= tolerance else "Unknown"

    def mark_attendance(self, name: str, now: datetime | None = None) -> None:
        if name == "Unknown" or name in self.attendance:
            return
        self.attendance[name] = now or datetime.now()

    def recognize_frame(
        self,
        frame,
        tolerance: float = 0.45,
        face_locations: Iterable[Tuple[int, int, int, int]] | None = None,
        face_encodings: Iterable[Sequence[float]] | None = None,
    ) -> tuple[list[Tuple[int, int, int, int]], list[str]]:
        cv2, face_recognition = _optional_imports()

        if face_locations is None or face_encodings is None:
            if cv2 is None or face_recognition is None:
                raise ImportError(
                    "OpenCV and face_recognition are required for live recognition."
                )
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        normalized_locations = list(face_locations)
        recognized_names: list[str] = []
        for encoding in face_encodings:
            name = self._find_best_match(encoding, tolerance=tolerance)
            recognized_names.append(name)
            self.mark_attendance(name)

        return normalized_locations, recognized_names

    def write_attendance_report(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(self.attendance.items(), key=lambda item: item[1])
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Name", "Timestamp"])
            for name, timestamp in rows:
                writer.writerow([name, timestamp.isoformat(timespec="seconds")])

    def run_live_attendance(self, output_path: Path, camera_index: int = 0) -> None:
        cv2, face_recognition = _optional_imports()
        if cv2 is None or face_recognition is None:
            raise ImportError(
                "OpenCV and face_recognition are required for live webcam attendance."
            )

        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"Unable to open webcam at index {camera_index}.")

        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break

                locations, names = self.recognize_frame(frame)
                for (top, right, bottom, left), name in zip(locations, names):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        name,
                        (left, max(20, top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("FaceGuard Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()

        self.write_attendance_report(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FaceGuard attendance system using OpenCV + deep learning"
    )
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        default=Path("known_faces"),
        help="Directory containing known face images named by person",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("attendance.csv"),
        help="Output CSV path for attendance report",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system = FaceGuardAttendance()
    if args.known_faces_dir.exists():
        system.load_known_faces(args.known_faces_dir)
    else:
        print(f"Known faces directory not found: {args.known_faces_dir}")
        return 1

    system.run_live_attendance(args.output, camera_index=args.camera_index)
    print(f"Attendance report saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
