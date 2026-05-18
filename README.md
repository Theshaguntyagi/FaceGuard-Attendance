# FaceGuard-Attendance
A system that uses OpenCV and deep learning to recognise faces and mark attendance automatically. Live webcam input, real-time recognition, and attendance reports. Demonstrates computer vision skills and real-time processing.

## Features
- Live webcam input with OpenCV
- Real-time face recognition using deep-learning encodings (`face_recognition`)
- Automatic attendance tracking (first recognition per person)
- CSV attendance report export with timestamps

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add known face images to `known_faces/` using the person name as the file name (for example `known_faces/Alice.jpg`).
3. Run:
   ```bash
   python faceguard_attendance.py --known-faces-dir known_faces --output attendance.csv
   ```
4. Press `q` in the video window to stop and save the report.
