# HumanSkeletonAR — Augmented Reality Human Skeleton Learning System

A Python and OpenCV project that uses an ArUco marker for a marker-based augmented-reality learning experience. The webcam feed is used to detect the marker and display the application's tracking output.

## Features

- Detects ArUco markers using OpenCV.
- Uses a webcam feed for marker detection.
- Displays marker-detection and tracking information.
- Includes the marker image and screenshots for demonstration reference.

## Project Structure

```text
HumanSkeletonAR/
├── main.py
├── aruco_marker.png
├── Screenshots/
├── README.md
└── .gitignore
```

## Requirements

- Python 3.11 or another compatible Python version
- A webcam
- OpenCV with the ArUco module
- NumPy

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/dhushanth05/HumanSkeletonAR.git
   cd HumanSkeletonAR
   ```

2. Create and activate a virtual environment (recommended).

   **Windows PowerShell:**

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the required packages:

   ```bash
   python -m pip install opencv-contrib-python numpy
   ```

   OpenCV's contrib package includes the ArUco module. Avoid installing conflicting OpenCV packages in the same environment.

## Run the Application

From the project root, run:

```bash
python main.py
```

Allow camera access if prompted. Hold the ArUco marker where the webcam can see it clearly. Use good lighting, keep the full marker in frame, and avoid glare or blur.

## Marker

The repository includes [aruco_marker.png](aruco_marker.png). Display or print the marker clearly for the camera to detect it. Keep its proportions intact when resizing or printing.

## Screenshots

Captured application screenshots are available in the [Screenshots folder](Screenshots/).

## Troubleshooting

- **Camera does not open:** Close other applications using the webcam and check camera permissions.
- **Marker is not detected:** Improve lighting, keep the whole marker in frame, and reduce glare or blur.
- **ArUco import or attribute errors:** Confirm that `opencv-contrib-python` is installed in the active environment and that conflicting OpenCV packages are not installed.
- **No AR output appears:** Check the terminal for errors and confirm the marker-detection status in the application.

## Tech Stack

Python · OpenCV · ArUco · NumPy

## License

No license has been specified for this repository yet.
