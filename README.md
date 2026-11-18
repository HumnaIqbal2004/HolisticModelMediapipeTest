# MediaPipe Pose Extraction + Motion Replay

A project built with **Python**, **OpenCV**, **Mediapipe(HolisticModel)** to extract landmark coordinates from video 
keyframes, write them to a CSV file, and reconstruct the skeleton frame by frame.

---

# 📌 Project Overview
This project provides a lightweight motion-capture workflow:
1. **Pose & Holistic Detection**: Processes video input using Google's MediaPipe Holistic Landmarker(which is a combination of the Face model, Hand model and Pose Landmark Model).
2. **Selective Landmark Extraction**: Captures specific frames and isolates key landmarks (arms, mouth, hands, etc.).
3. **Data Serialization**: Saves normalized landmark coordinates `(x, y)` into structured CSV files (`video_pose_data.csv`).
4. **Standalone Reconstruction & Replay**: Re-reads the serialized CSV data independently to render customizable wireframe/stick-figure animations onto a clean canvas with interactive frame-by-frame stepping.

---

## 🚀 Current Features
- [x] **MediaPipe Holistic Integration**: Tracks body pose, face mesh landmarks, and hand knuckles from MP4 video. (This can also be changed to a Live Web Camera)
- [x] **Keyframe Data Extraction**: Captures landmark coordinates at specific timestamps/frame numbers (Using the `s` key).
- [x] **CSV Data Logging**: Exports normalized coordinates in a structured `DictReader` format for offline storage and processing.
- [x] **Custom Skeleton Renderer (`readdict.py`)**: Re-creates and draws connections (shoulders, arms, hands, mouth) onto a solid background without needing the original video or re-running the ML model.
- [x] **Interactive Frame Stepping**: Manually navigate through extracted keyframes using keyboard controls (`d` for next frame, `q` to quit).

---

## 📁 Repository Structure
```text
├── assets/
│   ├── dance3.mp4             # Sample input video for motion tracking
│   └── Solid_black.png        # Canvas background for skeleton re-rendering
├── holistic_landmarker.task   # MediaPipe Holistic pretrained model task
├── main.py                    # Runs landmark detection & writes coordinates to CSV
├── readdict.py                # Reads CSV and reconstructs the stick-figure animation
├── video_pose_data.csv        # Extracted landmark coordinate dataset
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```
---

## Future Goals
This project is still in development, the goal is to use the extracted poses as a pose template for someone to follow through a live webcamera.
This will be done by comparing pose angles (NumPY)


