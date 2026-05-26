# KITTI Visual Odometry Pipeline

Stereo visual odometry pipeline built on the KITTI dataset format.

## What it does
- Loads stereo calibration (P0, P1) and extracts K matrix and baseline B
- Computes per-pixel depth from stereo disparity (StereoBM)
- Detects and matches ORB features across consecutive frames (Lowe ratio test)
- Estimates frame-to-frame pose using solvePnPRansac
- Accumulates pose with homogeneous transform chaining
- Evaluates trajectory with ATE and plots bird's eye view

## Results
Mean ATE: 23.15 metres on synthetic KITTI-format dataset

## Stack
Python, OpenCV, NumPy, Matplotlib
## Kaggle Notebook
[View interactive notebook on Kaggle](https://www.kaggle.com/code/mesojack/kitti-visual-odometry-stereo-seun-akinwa?scriptVersionId=322373976)
