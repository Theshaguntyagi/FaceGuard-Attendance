import csv
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from faceguard_attendance import FaceGuardAttendance


class FaceGuardAttendanceTests(unittest.TestCase):
    def test_mark_attendance_only_once(self):
        system = FaceGuardAttendance()
        first_seen = datetime(2026, 5, 18, 7, 0, 0)
        second_seen = datetime(2026, 5, 18, 8, 0, 0)

        system.mark_attendance("Alice", now=first_seen)
        system.mark_attendance("Alice", now=second_seen)

        self.assertEqual(system.attendance["Alice"], first_seen)

    def test_recognize_frame_marks_known_faces(self):
        system = FaceGuardAttendance(
            known_names=["Alice", "Bob"],
            known_encodings=[[0.0, 0.0], [1.0, 1.0]],
        )

        locations, names = system.recognize_frame(
            frame=None,
            tolerance=0.1,
            face_locations=[(0, 1, 2, 3), (4, 5, 6, 7)],
            face_encodings=[[0.05, 0.02], [4.0, 4.0]],
        )

        self.assertEqual(locations, [(0, 1, 2, 3), (4, 5, 6, 7)])
        self.assertEqual(names, ["Alice", "Unknown"])
        self.assertIn("Alice", system.attendance)
        self.assertNotIn("Unknown", system.attendance)

    def test_write_attendance_report(self):
        system = FaceGuardAttendance()
        seen_at = datetime(2026, 5, 18, 7, 10, 0)
        system.mark_attendance("Charlie", now=seen_at)

        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "attendance.csv"
            system.write_attendance_report(report_path)

            with report_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(rows[0], ["Name", "Timestamp"])
        self.assertEqual(rows[1], ["Charlie", "2026-05-18T07:10:00"])


if __name__ == "__main__":
    unittest.main()
