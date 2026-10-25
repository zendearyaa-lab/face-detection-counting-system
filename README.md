# Real-Time Face Detection & Counting System

Detects and counts faces from a live webcam stream using OpenCV Haar Cascades, with a centroid tracker so each person is counted only once.

## Features
- Real-time face detection (Haar Cascade)
- Unique face counting with ID tracking
- Live FPS and count overlay
- Optimized for low latency (downscaled frames, detection on alternate frames)

## Tech
Python, OpenCV, NumPy, Computer Vision

## How to run
pip install -r requirements.txt
python face_counter.py

Press q to quit.
