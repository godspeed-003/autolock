# Autolock V4 - Face Recognition System

## Overview
Autolock V4 is a face recognition system designed to enhance security by automatically locking and unlocking your computer based on user presence. The system utilizes a YOLOv8 model for face detection and K-Nearest Neighbors (KNN) for face recognition. It can manage a blacklist of faces and prevent unauthorized access.

## Features
- Real-time face detection and recognition
- Automatic system locking when the user is absent
- Blacklist management for unauthorized faces
- User-friendly interface for creating a master profile

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
   ```
   git clone <repository-url>
   cd autolock-v4
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the face recognition system, execute the following command:
```
python src/autolock_v4.py
```

Follow the on-screen instructions to create a master profile and start the system.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [YOLOv8](https://github.com/ultralytics/yolov5) for face detection
- [OpenCV](https://opencv.org/) for image processing
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms

For any issues or feature requests, please open an issue on the GitHub repository.