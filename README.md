# Face Recognition System

A real-time face recognition system built with Python, offering efficient face detection, recognition, and clustering capabilities. The system provides live video processing, multiple detection algorithms, and automatic face clustering for unknown faces.

## Features

- **Real-time Video Processing**
  - Live video feed with face detection
  - Efficient frame processing with multi-threaded architecture
  - FPS counter and performance optimization
  - Configurable video resolution and camera settings

- **Face Detection & Recognition**
  - Multiple face detection algorithms (HOG and CNN)
  - Support for both Euclidean and Cosine similarity metrics
  - Configurable recognition tolerance and processing parameters
  - Real-time face matching with known profiles

- **Face Clustering**
  - Automatic grouping of similar faces
  - DBSCAN-based clustering algorithm
  - Profile management for unknown faces
  - Cluster optimization for accuracy

- **Profile Management**
  - Automatic profile directory monitoring
  - Dynamic profile updates without restart
  - Image quality assessment and cleanup
  - Duplicate detection and removal

- **Performance Optimization**
  - Intelligent frame dropping for smooth processing
  - Multi-threaded architecture for parallel processing
  - Caching system for improved performance
  - Configurable processing parameters

## System Requirements

- Python 3.7+
- OpenCV-compatible camera
- Sufficient CPU for real-time processing (GPU optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start Guide

1. **Prepare Profile Images**
   - Create profile images in the `data/profiles` directory
   - One clear face per image recommended
   - Supported formats: PNG, JPG, JPEG, GIF, BMP

2. **Configure Settings (Optional)**
   - Adjust camera settings in `src/config/general_config.py`
   - Modify recognition parameters in `src/config/models_config.py`

3. **Run the System**
```bash
python main.py
```

4. **Usage**
   - The system will automatically detect and recognize faces from profile images
   - Press 'q' or close the window to exit
   - New profiles are detected automatically - no restart needed

## Project Structure

```
face_recognition_system/
├── src/
│   ├── core/                    # Core recognition components
│   │   ├── face/
│   │   │   ├── detector/       # Face detection implementations
│   │   │   ├── encoders/       # Face encoding logic
│   │   │   └── matchers/       # Face matching algorithms
│   ├── utils/                   # Utility functions
│   ├── services/               # Main system services
│   ├── config/                 # System configuration
│   └── models/                 # Data models
├── data/                       # Data storage
│   ├── profiles/              # Profile images
│   └── cache/                 # System cache
├── requirements.txt
└── main.py
```

## Configuration

### Camera Settings (`general_config.py`)
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_INDEX = 0  # Default webcam
```

### Recognition Settings (`models_config.py`)
```python
FACE_DETECTION_MODEL = 'hog'  # or 'cnn' for GPU systems
RECOGNITION_TOLERANCE = 0.6
FRAME_SCALE_FACTOR = 0.25
```

## Components Overview

### Core Services

1. **Video Service**
   - Handles camera input and display
   - Manages frame buffering and processing
   - Provides FPS monitoring

2. **Recognition Service**
   - Processes frames for face detection
   - Matches detected faces against profiles
   - Manages real-time recognition

3. **Clustering Service**
   - Groups similar faces automatically
   - Manages unknown face profiles
   - Provides cluster analysis

4. **Watcher Service**
   - Monitors profile directory for changes
   - Handles dynamic profile updates
   - Manages profile cleanup

### Utilities

- **Image Processor**
  - Image quality assessment
  - Face cropping and resizing
  - Duplicate detection

- **Cache Manager**
  - Caches recognition results
  - Manages system performance
  - Handles cleanup

## Performance Optimization

1. **Frame Processing**
   - Configurable frame scaling
   - Intelligent frame dropping
   - Buffer management

2. **Recognition**
   - HOG-based detection for CPU
   - Optional CNN detection for GPU
   - Configurable recognition tolerance

3. **Caching**
   - Face encoding caching
   - Profile data caching
   - Automatic cache cleanup

## Troubleshooting

1. **System Performance**
   - Reduce `FRAME_SCALE_FACTOR` for better performance
   - Use 'hog' detection model on CPU systems
   - Adjust buffer sizes in `VideoService`

2. **Recognition Accuracy**
   - Increase `RECOGNITION_TOLERANCE` for stricter matching
   - Use higher quality profile images
   - Adjust `NUM_JITTERS` for better encoding

3. **Common Issues**
   - Camera not found: Check `CAMERA_INDEX`
   - High CPU usage: Reduce frame resolution
   - Recognition delays: Adjust buffer sizes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Face recognition powered by face-recognition library
- OpenCV for image processing
- NumPy for numerical operations
- Watchdog for file system monitoring
