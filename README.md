ArUco Marker Detection and Augmented Reality Overlay
This project implements real-time ArUco marker detection using OpenCV, enabling augmented reality (AR) overlays on detected markers.

Features
Detects multiple ArUco markers (IDs 23 and 49) in live camera feed

Calculates marker positions and orientations

Applies perspective transformation to overlay videos/images precisely on marker surfaces

Supports dynamic video looping and real-time AR visualization

Technologies Used
Python 3.x

OpenCV with ArUco module

NumPy

How It Works
Capture video frames from a webcam.

Detect specified ArUco markers and extract their corner coordinates.

Compute the centers of markers to identify their relative positions.

Calculate a perspective transform matrix to warp the overlay video/image onto the quadrilateral defined by markers.

Generate a mask to blend the overlay seamlessly onto the camera frame.

Display the augmented frame in real-time.

Usage
Install dependencies:

bash
Kopyala
pip install opencv-python numpy
Run the main Python script:

bash
Kopyala
python your_script.py
Place ArUco markers with IDs 23 and 49 in the camera view to see the AR overlay applied.

Notes
Ensure that the markers are well-lit and clearly visible for accurate detection.

The overlay video (recep.webm) should be located in the same directory or adjust the path accordingly.



