# Autolock V4 - Face Recognition System

## Overview
Autolock V4 is a face recognition system designed to enhance security by automatically locking and unlocking your computer based on user presence. The system utilizes a YOLOv8 model for face detection and K-Nearest Neighbors (KNN) for face recognition. It can manage a blacklist of faces and prevent unauthorized access.

## Features
- Real-time face detection and recognition
- Automatic system locking when user is absent
- Blacklist management for unauthorized faces 
- User-friendly interface for creating master profile
- Multi-platform support (Windows, macOS, Linux)

## Version History

### Autolock V1 (Initial Release)
- Basic face detection using YOLOv8
- Simple face recognition with KNN classifier
- Manual system lock when no face detected
- Basic webcam handling and UI display

### Autolock V2 (Security Update)
- Added blacklist system for unauthorized faces
- Improved face recognition accuracy with normalized features 
- Increased inactivity timeout to 10 seconds (from 5)
- Added validation during profile creation
- Enhanced error handling and logging

### Autolock V3 (Auto-Unlock Update) 
- Added automatic unlock feature using PIN
- Introduced system lock state tracking
- Added lock timestamp monitoring
- PIN-based security for unlock operations
- Improved recognition confidence handling
- Enhanced profile creation with real-time feedback

### Autolock V4 (Current - Performance Update)
- Improved recognition speed and accuracy
- Added rolling confidence average for stability
- Reduced false positives with stricter thresholds
- Enhanced feature extraction with HOG
- Added auto-recovery from failed recognition
- Improved multi-threading for lock monitoring
- Auto-switching between available cameras

## Project Structure

### Core Files
- `autolock_v4.py`: Main application file containing the face recognition system
- `requirements.txt`: Lists all Python dependencies
- `yolov8n-face.pt`: YOLOv8 face detection model (download separately)

### Configuration & Data
- `face_data/`: Directory storing face recognition data
  - `master_embeddings.pkl`: Stored face embeddings for master user
  - `master_knn_model.pkl`: Trained KNN model for face recognition
  - `blacklist.pkl`: List of blocked face embeddings
  - `master_feature_dim.txt`: Feature dimensions configuration

### Testing & Setup Files
- `tests/`
  - `test_face_recognition.py`: Unit tests for face recognition system
  - `test_system_lock.py`: Tests for system locking functionality
  - `test_blacklist.py`: Tests for blacklist management
  - `conftest.py`: PyTest configuration and fixtures

### Helper Scripts
- `scripts/`
  - `setup_camera.py`: Utility to test and configure webcam
  - `model_download.py`: Script to download YOLOv8 face model
  - `profile_cleanup.py`: Tool to reset master profile
  - `performance_test.py`: Benchmarking script

### Documentation
- `docs/`
  - `setup_guide.md`: Detailed installation instructions
  - `troubleshooting.md`: Common issues and solutions
  - `api_reference.md`: API documentation
  - `CONTRIBUTING.md`: Guidelines for contributors

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run specific test categories:
```bash
python -m pytest tests/test_face_recognition.py
python -m pytest tests/test_system_lock.py
```

Generate test coverage report:
```bash
python -m pytest --cov=src tests/
```

## Development Setup

1. Clone and install dependencies:
```bash
git clone <repository-url>
cd autolock-v4
pip install -r requirements.txt
```

2. Download YOLOv8 model:
```bash
python scripts/model_download.py
```

3. Test camera setup:
```bash
python scripts/setup_camera.py
```

4. Run performance tests:
```bash
python scripts/performance_test.py
```

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

To run the face recognition system:

```
python src/autolock_v4.py
```

Optional arguments:
- `--model`: Path to YOLOv8 face detection model
- `--data-dir`: Directory for face recognition data
- `--master`: Name for master user
- `--timeout`: Inactivity timeout before locking
- `--tolerance`: Recognition confidence tolerance
- `--retrain`: Force retraining of master profile

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [YOLOv8](https://github.com/ultralytics/yolov5) for face detection
- [OpenCV](https://opencv.org/) for image processing
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms

For any issues or feature requests, please open an issue on the GitHub repository.